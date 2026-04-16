import math

import einops
import elements
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

import embodied.jax
import embodied.jax.nets as nn

f32 = jnp.float32
sg = jax.lax.stop_gradient


class RSSM(nj.Module):
    # default hp values for world model learning
    # overridden by agent.dyn.rssm in configs.yaml

    # model state
    deter: int = 4096  # D
    stoch: int = 32    # S
    classes: int = 32  # C

    # MLP
    hidden: int = 2048  # H
    imglayers: int = 2
    obslayers: int = 1
    dynlayers: int = 1

    # layer
    act: str = 'gelu'
    norm: str = 'rms'
    outscale: float = 1.0

    # regularization
    unimix: float = 0.01
    free_nats: float = 1.0
    absolute: bool = False

    # GPU computation
    blocks: int = 8
    unroll: bool = False

    # data
    # overridden by defaults in configs.yaml
    # batch_size = 16              # B
    # batch_length = 64 (seq_len)  # T
    # nlast = batch_length         # K

    def __init__(self, act_space, **kw):
        assert self.deter % self.blocks == 0
        self.act_space = act_space
        self.kw = kw

    @property
    def entry_space(self):
        # model state s_t = (h_t, z_t)
        return dict(
            deter=elements.Space(np.float32, self.deter),                  # h_t (D,)
            stoch=elements.Space(np.float32, (self.stoch, self.classes)),  # z_t (S, C)
        )

    def initial(self, batch_size):
        """Initializes model state to (0, 0) when episode begins."""
        carry = nn.cast(dict(
            deter=jnp.zeros([batch_size, self.deter], f32),
            stoch=jnp.zeros([batch_size, self.stoch, self.classes], f32),
        ))
        return carry

    def truncate(self, entries, carry=None):
        """Takes last entries as carry for next batch initialization."""
        assert entries['deter'].ndim == 3, entries['deter'].shape
        carry = jax.tree.map(lambda x: x[:, -1], entries)
        return carry

    def starts(self, entries, carry, nlast):
        """Extracts last K model states from entries as starting points for imagination."""
        B = len(jax.tree.leaves(carry)[0])
        return jax.tree.map(
            lambda x: x[:, -nlast:].reshape((B * nlast, *x.shape[2:])),
            entries,
        )  # ((B, T, D), (B, T, S, C)) -> ((B*K, D), (B*K, S, C))

    def observe(self, carry, tokens, action, reset, training, single=False):
        """Runs posterior inference over the sequences.

        Returns:
            single=True:
                Returns a single _observe() call.
            single=False:
                carry: Last model states [s_T] = [(h_T, z_T)] of sequences.
                    Will be passed to next step.
                    Shape: ((B, D), (B, S, C)).
                entries: Sequences of model states.
                    Stack of each model state per timestep, will be saved in buffer.
                    Shape: ((B, T, D), (B, T, S, C)).
                feat: Sequences of model states and logits.
                    Will be used for loss calculation.
                    Shape: ((B, T, D), (B, T, S, C), (B, T, S, C)).
        """
        carry, tokens, action = nn.cast((carry, tokens, action))

        # computes only one posterior transition
        # used in policy() in agent.py for real-time interaction
        if single:
            carry, (entry, feat) = self._observe(carry, tokens, action, reset, training)
            return carry, entry, feat
        # computes posterior transitions from t=0 to t=T-1
        else:
            unroll = jax.tree.leaves(tokens)[0].shape[1] if self.unroll else 1
            carry, (entries, feat) = nj.scan(
                lambda carry, inputs: self._observe(carry, *inputs, training),
                carry,
                (tokens, action, reset),
                unroll=unroll,
                axis=1,
            )
            return carry, entries, feat

    def _observe(self, carry, tokens, action, reset, training):
        """Runs posterior inference for a single timestep.

        h_t = _core(h_{t-1}, z_{t-1}, a_{t-1}).
        z_t ~ q_phi(z_t | h_t, x_t).

        Returns:
            carry: Model states [s_t] = [(h_t, z_t)]. Shape: ((B, D), (B, S, C)).
            entry: Model states [s_t] = [(h_t, z_t)]. Shape: ((B, D), (B, S, C)).
            feat: Model states and logits. Shape: ((B, D), (B, S, C), (B, S, C)).
        """
        deter, stoch, action = nn.mask((carry['deter'], carry['stoch'], action), ~reset)
        action = nn.DictConcat(self.act_space, 1)(action)
        action = nn.mask(action, ~reset)  # reset: initialize (h, z, a) to 0 for episode boundaries handling

        deter = self._core(deter, stoch, action)
        tokens = tokens.reshape((*deter.shape[:-1], -1))
        x = tokens if self.absolute else jnp.concatenate([deter, tokens], -1)  # self.absolute: z_t ~ q_phi(z_t | x_t)

        # compute posterior logits q_phi(z_t | h_t, x_t) via MLP
        for i in range(self.obslayers):
            x = self.sub(f'obs{i}', nn.Linear, self.hidden, **self.kw)(x)
            x = nn.act(self.act)(self.sub(f'obs{i}norm', nn.Norm, self.norm)(x))
        logit = self._logit('obslogit', x)
        stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))
        assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, stoch, logit))

        carry = dict(deter=deter, stoch=stoch)
        feat = dict(deter=deter, stoch=stoch, logit=logit)
        entry = dict(deter=deter, stoch=stoch)

        return carry, (entry, feat)

    def imagine(self, carry, policy, length, training, single=False):
        """Runs prior inference over the imagined sequences.

        h_t = _core(h_{t-1}, z_{t-1}, a_{t-1}).
        z_t_hat ~ p_phi(z_t | h_t).
            Prior does not access x_t.

        Parameters:
            length: Number of imagination steps (used only when single=False).

        Returns:
            single=True:
                carry: Model states [s_t] = [(h_t, z_t)].
                    Shape: ((B, D), (B, S, C)).
                feat: Model states and logits.
                    Shape: ((B, D), (B, S, C), (B, S, C)).
                action: Single step actions.
                    Shape: (B, act_dim).
            single=False:
                carry: Last model states [s_T] = [(h_T, z_T)] of sequences.
                    Will be passed to next step.
                    Shape: ((B, D), (B, S, C)).
                feat: Sequences of model states and logits.
                    Will be used for loss calculation.
                    Shape: ((B, T, D), (B, T, S, C), (B, T, S, C)).
                action: Sequence of actions.
                    Shape: (B, T, act_dim).
        """
        # computes only one imagined transition
        if single:
            action = policy(sg(carry)) if callable(policy) else policy
            actemb = nn.DictConcat(self.act_space, 1)(action)

            deter = self._core(carry['deter'], carry['stoch'], actemb)
            logit = self._prior(deter)
            stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))
            assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, stoch, logit))

            carry = nn.cast(dict(deter=deter, stoch=stoch))
            feat = nn.cast(dict(deter=deter, stoch=stoch, logit=logit))

            return carry, (feat, action)
        # computes imagined transitions from t=0 to t=length-1
        else:
            unroll = length if self.unroll else 1
            if callable(policy):
                carry, (feat, action) = nj.scan(
                    lambda c, _: self.imagine(c, policy, 1, training, single=True),
                    nn.cast(carry),
                    (),
                    length,
                    unroll=unroll,
                    axis=1,
                )
            else:
                carry, (feat, action) = nj.scan(
                    lambda c, a: self.imagine(c, a, 1, training, single=True),
                    nn.cast(carry),
                    nn.cast(policy),
                    length,
                    unroll=unroll,
                    axis=1,
                )

            return carry, feat, action

    def loss(self, carry, tokens, acts, reset, training):
        """Calculates KL losses."""
        metrics = {}

        carry, entries, feat = self.observe(carry, tokens, acts, reset, training)

        prior = self._prior(feat['deter'])  # p_phi(z_t | h_t)
        post = feat['logit']                # q_phi(z_t | h_t, x_t)

        dyn = self._dist(sg(post)).kl(self._dist(prior))  # L_dyn
        rep = self._dist(post).kl(self._dist(sg(prior)))  # L_rep
        if self.free_nats:  # lower bound of KL losses
            dyn = jnp.maximum(dyn, self.free_nats)
            rep = jnp.maximum(rep, self.free_nats)
        losses = {'dyn': dyn, 'rep': rep}

        # training monitor metrics
        metrics['dyn_ent'] = self._dist(prior).entropy().mean()  # prior entropy
        metrics['rep_ent'] = self._dist(post).entropy().mean()   # posterior entropy

        return carry, entries, losses, feat, metrics

    def _core(self, deter, stoch, action):
        """Computes recurrent state via GRU."""
        stoch = stoch.reshape((stoch.shape[0], -1))
        action /= sg(jnp.maximum(1, jnp.abs(action)))

        g = self.blocks
        flat2group = lambda x: einops.rearrange(x, '... (g h) -> ... g h', g=g)
        group2flat = lambda x: einops.rearrange(x, '... g h -> ... (g h)', g=g)

        x0 = self.sub('dynin0', nn.Linear, self.hidden, **self.kw)(deter)
        x0 = nn.act(self.act)(self.sub('dynin0norm', nn.Norm, self.norm)(x0))
        x1 = self.sub('dynin1', nn.Linear, self.hidden, **self.kw)(stoch)
        x1 = nn.act(self.act)(self.sub('dynin1norm', nn.Norm, self.norm)(x1))
        x2 = self.sub('dynin2', nn.Linear, self.hidden, **self.kw)(action)
        x2 = nn.act(self.act)(self.sub('dynin2norm', nn.Norm, self.norm)(x2))

        x = jnp.concatenate([x0, x1, x2], -1)[..., None, :].repeat(g, -2)
        x = group2flat(jnp.concatenate([flat2group(deter), x], -1))
        for i in range(self.dynlayers):
            x = self.sub(f'dynhid{i}', nn.BlockLinear, self.deter, g, **self.kw)(x)
            x = nn.act(self.act)(self.sub(f'dynhid{i}norm', nn.Norm, self.norm)(x))

        x = self.sub('dyngru', nn.BlockLinear, 3 * self.deter, g, **self.kw)(x)
        gates = jnp.split(flat2group(x), 3, -1)
        reset, cand, update = [group2flat(x) for x in gates]
        reset = jax.nn.sigmoid(reset)
        cand = jnp.tanh(reset * cand)
        update = jax.nn.sigmoid(update - 1)
        deter = update * cand + (1 - update) * deter

        return deter  # h_t = f_phi(h_{t-1}, z_{t-1}, a_{t-1})

    def _prior(self, feat):
        """Computes prior logits via MLP."""
        x = feat
        for i in range(self.imglayers):
            x = self.sub(f'prior{i}', nn.Linear, self.hidden, **self.kw)(x)
            x = nn.act(self.act)(self.sub(f'prior{i}norm', nn.Norm, self.norm)(x))
        return self._logit('priorlogit', x)  # p_phi(z_t | h_t)

    def _logit(self, name, x):
        """Projects x to logits."""
        kw = dict(**self.kw, outscale=self.outscale)
        x = self.sub(name, nn.Linear, self.stoch * self.classes, **kw)(x)  # (..., S*C)
        return x.reshape(x.shape[:-1] + (self.stoch, self.classes))        # (..., S, C)

    def _dist(self, logits):
        """Wraps logits into a categorical distribution with unimix smoothing."""
        out = embodied.jax.outs.OneHot(logits, self.unimix)
        out = embodied.jax.outs.Agg(out, 1, jnp.sum)
        return out


class Encoder(nj.Module):
    """Encodes observations to tokens.

    Encoder only maps x_t to token.
    Posterior q_phi(z_t | h_t, x_t) construction and z_t sampling are performed in _observe().
    """
    # default hp values for encoder
    # overridden by agent.enc.simple in configs.yaml

    # MLP; for vector observation
    units: int = 1024
    layers: int = 3
    symlog: bool = True

    # CNN; for image observation
    depth: int = 64
    mults: tuple = (2, 3, 4, 4)
    kernel: int = 5
    outer: bool = False
    strided: bool = False

    # layer
    norm: str = 'rms'
    act: str = 'gelu'

    def __init__(self, obs_space, **kw):
        assert all(len(s.shape) <= 3 for s in obs_space.values()), obs_space
        self.obs_space = obs_space
        self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2]  # vector observation keys
        self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]  # image observation keys
        self.depths = tuple(self.depth * mult for mult in self.mults)
        self.kw = kw

    @property
    def entry_space(self):
        return {}  # encoder has no recurrent state

    def initial(self, batch_size):
        return {}  # encoder has no recurrent state

    def truncate(self, entries, carry=None):
        return {}  # encoder has no recurrent state

    def __call__(self, carry, obs, reset, training, single=False):
        bdims = 1 if single else 2
        outs = []
        bshape = reset.shape

        # encodes via MLP if observation is vector
        if self.veckeys:
            vspace = {k: self.obs_space[k] for k in self.veckeys}
            vecs = {k: obs[k] for k in self.veckeys}
            squish = nn.symlog if self.symlog else lambda x: x  # vector: symlog normalization
            x = nn.DictConcat(vspace, 1, squish=squish)(vecs)
            x = x.reshape((-1, *x.shape[bdims:]))
            for i in range(self.layers):
                x = self.sub(f'mlp{i}', nn.Linear, self.units, **self.kw)(x)
                x = nn.act(self.act)(self.sub(f'mlp{i}norm', nn.Norm, self.norm)(x))
            outs.append(x)
        # encodes via CNN if observation is image
        if self.imgkeys:
            K = self.kernel
            imgs = [obs[k] for k in sorted(self.imgkeys)]
            assert all(x.dtype == jnp.uint8 for x in imgs)
            x = nn.cast(jnp.concatenate(imgs, -1), force=True) / 255 - 0.5  # image: [0, 255] → [-0.5, 0.5]
            x = x.reshape((-1, *x.shape[bdims:]))
            for i, depth in enumerate(self.depths):
                if self.outer and i == 0:
                    x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, **self.kw)(x)
                elif self.strided:
                    x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, 2, **self.kw)(x)
                else:
                    x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, **self.kw)(x)
                    B, H, W, C = x.shape
                    x = x.reshape((B, H // 2, 2, W // 2, 2, C)).max((2, 4))
                x = nn.act(self.act)(self.sub(f'cnn{i}norm', nn.Norm, self.norm)(x))
            assert 3 <= x.shape[-3] <= 16, x.shape
            assert 3 <= x.shape[-2] <= 16, x.shape
            x = x.reshape((x.shape[0], -1))
            outs.append(x)

        x = jnp.concatenate(outs, -1)
        tokens = x.reshape((*bshape, *x.shape[1:]))

        entries = {}

        return carry, entries, tokens


class Decoder(nj.Module):
    """Reconstructs observations from model states.

    Decoder only maps s_t = (h_t, z_t) to x_t_hat.
    Prior p_phi(z_t | h_t) construction is performed in _prior().
    Reconstruction loss is computed in agent.py.
    """
    # default hp values for decoder
    # overridden by agent.dec.simple in configs.yaml

    # MLP; for vector observation
    units: int = 1024
    layers: int = 3
    symlog: bool = True

    # Transposed CNN; for image observation
    depth: int = 64
    mults: tuple = (2, 3, 4, 4)
    kernel: int = 5
    bspace: int = 8
    outer: bool = False
    strided: bool = False

    # layer
    norm: str = 'rms'
    act: str = 'gelu'
    outscale: float = 1.0

    def __init__(self, obs_space, **kw):
        assert all(len(s.shape) <= 3 for s in obs_space.values()), obs_space
        self.obs_space = obs_space
        self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2]  # vector observation keys
        self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]  # image observation keys
        self.depths = tuple(self.depth * mult for mult in self.mults)
        self.imgdep = sum(obs_space[k].shape[-1] for k in self.imgkeys)
        self.imgres = self.imgkeys and obs_space[self.imgkeys[0]].shape[:-1]
        self.kw = kw

    @property
    def entry_space(self):
        return {}  # decoder has no recurrent state

    def initial(self, batch_size):
        return {}  # decoder has no recurrent state

    def truncate(self, entries, carry=None):
        return {}  # decoder has no recurrent state

    def __call__(self, carry, feat, reset, training, single=False):
        assert feat['deter'].shape[-1] % self.bspace == 0
        K = self.kernel
        recons = {}

        bshape = reset.shape
        inp = [nn.cast(feat[k]) for k in ('stoch', 'deter')]
        inp = [x.reshape((math.prod(bshape), -1)) for x in inp]
        inp = jnp.concatenate(inp, -1)

        # reconstructs via MLP if observation is vector
        if self.veckeys:
            spaces = {k: self.obs_space[k] for k in self.veckeys}
            o1, o2 = 'categorical', ('symlog_mse' if self.symlog else 'mse')  # vector: symlog_mse loss
            outputs = {k: o1 if v.discrete else o2 for k, v in spaces.items()}
            kw = dict(**self.kw, act=self.act, norm=self.norm)
            x = self.sub('mlp', nn.MLP, self.layers, self.units, **kw)(inp)
            x = x.reshape((*bshape, *x.shape[1:]))
            kw = dict(**self.kw, outscale=self.outscale)
            outs = self.sub('vec', embodied.jax.DictHead, spaces, outputs, **kw)(x)
            recons.update(outs)
        # reconstructs via transposed CNN if observation is image
        if self.imgkeys:
            factor = 2 ** (len(self.depths) - int(bool(self.outer)))
            minres = [int(x // factor) for x in self.imgres]
            assert 3 <= minres[0] <= 16, minres
            assert 3 <= minres[1] <= 16, minres
            shape = (*minres, self.depths[-1])
            if self.bspace:
                u, g = math.prod(shape), self.bspace
                x0, x1 = nn.cast((feat['deter'], feat['stoch']))
                x1 = x1.reshape((*x1.shape[:-2], -1))
                x0 = x0.reshape((-1, x0.shape[-1]))
                x1 = x1.reshape((-1, x1.shape[-1]))
                x0 = self.sub('sp0', nn.BlockLinear, u, g, **self.kw)(x0)
                x0 = einops.rearrange(x0, '... (g h w c) -> ... h w (g c)', h=minres[0], w=minres[1], g=g)
                x1 = self.sub('sp1', nn.Linear, 2 * self.units, **self.kw)(x1)
                x1 = nn.act(self.act)(self.sub('sp1norm', nn.Norm, self.norm)(x1))
                x1 = self.sub('sp2', nn.Linear, shape, **self.kw)(x1)
                x = nn.act(self.act)(self.sub('spnorm', nn.Norm, self.norm)(x0 + x1))
            else:
                x = self.sub('space', nn.Linear, shape, **kw)(inp)
                x = nn.act(self.act)(self.sub('spacenorm', nn.Norm, self.norm)(x))
            for i, depth in reversed(list(enumerate(self.depths[:-1]))):
                if self.strided:
                    kw = dict(**self.kw, transp=True)
                    x = self.sub(f'conv{i}', nn.Conv2D, depth, K, 2, **kw)(x)
                else:
                    x = x.repeat(2, -2).repeat(2, -3)
                    x = self.sub(f'conv{i}', nn.Conv2D, depth, K, **self.kw)(x)
                x = nn.act(self.act)(self.sub(f'conv{i}norm', nn.Norm, self.norm)(x))
            if self.outer:
                kw = dict(**self.kw, outscale=self.outscale)
                x = self.sub('imgout', nn.Conv2D, self.imgdep, K, **kw)(x)
            elif self.strided:
                kw = dict(**self.kw, outscale=self.outscale, transp=True)
                x = self.sub('imgout', nn.Conv2D, self.imgdep, K, 2, **kw)(x)
            else:
                x = x.repeat(2, -2).repeat(2, -3)
                kw = dict(**self.kw, outscale=self.outscale)
                x = self.sub('imgout', nn.Conv2D, self.imgdep, K, **kw)(x)
            x = jax.nn.sigmoid(x)  # image: logits → [0, 1]
            x = x.reshape((*bshape, *x.shape[1:]))
            split = np.cumsum([self.obs_space[k].shape[-1] for k in self.imgkeys][:-1])
            for k, out in zip(self.imgkeys, jnp.split(x, split, -1)):
                out = embodied.jax.outs.MSE(out)
                out = embodied.jax.outs.Agg(out, 3, jnp.sum)
                recons[k] = out

        entries = {}

        return carry, entries, recons
