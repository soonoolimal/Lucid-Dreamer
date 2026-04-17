import re

import chex
import elements
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax

import embodied.jax
import embodied.jax.nets as nn

from . import rssm

f32 = jnp.float32
i32 = jnp.int32
sg = lambda xs, skip=False: xs if skip else jax.lax.stop_gradient(xs)
sample = lambda xs: jax.tree.map(lambda x: x.sample(nj.seed()), xs)
prefix = lambda xs, p: {f'{p}/{k}': v for k, v in xs.items()}
concat = lambda xs, a: jax.tree.map(lambda *x: jnp.concatenate(x, a), *xs)
isimage = lambda s: s.dtype == np.uint8 and len(s.shape) == 3


class Agent(embodied.jax.Agent):
    """Dreamerv3 agent."""
    banner = [
        r"---  ___                           __   ______ ---",
        r"--- |   \ _ _ ___ __ _ _ __  ___ _ \ \ / /__ / ---",
        r"--- | |) | '_/ -_) _` | '  \/ -_) '/\ V / |_ \ ---",
        r"--- |___/|_| \___\__,_|_|_|_\___|_|  \_/ |___/ ---",
    ]
    def __init__(self, obs_space, act_space, config):
        self.obs_space = obs_space
        self.act_space = act_space
        self.config = config

        exclude = ('is_first', 'is_last', 'is_terminal', 'reward')
        enc_space = {k: v for k, v in obs_space.items() if k not in exclude}
        dec_space = {k: v for k, v in obs_space.items() if k not in exclude}
        self.enc = {
            'simple': rssm.Encoder,
        }[config.enc.typ](enc_space, **config.enc[config.enc.typ], name='enc')
        self.dyn = {
            'rssm': rssm.RSSM,
        }[config.dyn.typ](act_space, **config.dyn[config.dyn.typ], name='dyn')
        self.dec = {
            'simple': rssm.Decoder,
        }[config.dec.typ](dec_space, **config.dec[config.dec.typ], name='dec')

        # sequence of model states
        #   from replay buffer: repfeat (B, T, D+S*C)
        #   from imagination: imgfeat (B*K, H+1, D+S*C)
        self.feat2tensor = lambda x: jnp.concatenate([
            nn.cast(x['deter']),                                        # h_t
            nn.cast(x['stoch'].reshape((*x['stoch'].shape[:-2], -1))),  # z_t
        ], -1)

        # r_t_hat and c_t_hat
        scalar = elements.Space(np.float32, ())
        binary = elements.Space(bool, (), 0, 2)
        self.rew = embodied.jax.MLPHead(scalar, **config.rewhead, name='rew')  # r_t_hat (B, T)
        self.con = embodied.jax.MLPHead(binary, **config.conhead, name='con')  # c_t_hat (B, T)

        # actor network pi_theta(a | s)
        d1, d2 = config.policy_dist_disc, config.policy_dist_cont
        outs = {k: d1 if v.discrete else d2 for k, v in act_space.items()}
        self.pol = embodied.jax.MLPHead(act_space, outs, **config.policy, name='pol')

        # critic network v_psi(s) and EMA critic network v_psi_bar(s)
        self.val = embodied.jax.MLPHead(scalar, **config.value, name='val')
        self.slowval = embodied.jax.SlowModel(
            embodied.jax.MLPHead(scalar, **config.value, name='slowval'),
            source=self.val,
            **config.slowvalue,  # slow-value EMA update params
        )

        # normalization layers
        self.retnorm = embodied.jax.Normalize(**config.retnorm, name='retnorm')  # lambda-return G_t^lambda
        self.valnorm = embodied.jax.Normalize(**config.valnorm, name='valnorm')  # value v(s_t)
        self.advnorm = embodied.jax.Normalize(**config.advnorm, name='advnorm')  # advantage G_t^lambda - v(s_t)

        # all neural networks of dreamer (slowval excluded; EMA copy, not a gradient target)
        self.modules = [self.dyn, self.enc, self.dec, self.rew, self.con, self.pol, self.val]

        # optimizer
        self.opt = embodied.jax.Optimizer(self.modules, self._make_opt(**config.opt), summary_depth=1, name='opt')

        # betas
        scales = self.config.loss_scales.copy()
        rec = scales.pop('rec')
        scales.update(dict.fromkeys(dec_space, rec))
        self.scales = scales

    @property
    def policy_keys(self):
        return '^(enc|dyn|dec|pol)/'  # params included in policy checkpoint

    @property
    def ext_space(self):
        """Returns extra fields to be stored in replay buffer."""
        spaces = {}
        spaces['consec'] = elements.Space(np.int32)
        spaces['stepid'] = elements.Space(np.uint8, 20)
        # also store enc/dyn/dec carries to reuse them as context for next batch
        if self.config.replay_context:
            spaces.update(elements.tree.flatdict(dict(
                enc=self.enc.entry_space,  # {} (encoder has no model state)
                dyn=self.dyn.entry_space,  # s_t = (h_t, z_t)
                dec=self.dec.entry_space,  # {} (decoder has no model state)
            )))
        return spaces

    def init_policy(self, batch_size):
        zeros = lambda x: jnp.zeros((batch_size, *x.shape), x.dtype)
        return (
            self.enc.initial(batch_size),
            self.dyn.initial(batch_size),
            self.dec.initial(batch_size),
            jax.tree.map(zeros, self.act_space)
        )  # (enc={}, dyn=(h_0=0, z_0=0), dec={}, a_0=0)

    def init_train(self, batch_size):
        return self.init_policy(batch_size)

    def init_report(self, batch_size):
        return self.init_policy(batch_size)

    def policy(self, carry, obs, mode='train'):
        """Runs a single agent-environment interaction.

        Computes model states:
            h_t = f_phi(h_{t-1}, z_{t-1}, a_{t-1}).
            z_t ~ q_phi(z_t | h_t, enc(x_t)).
        Samples actions:
            a_t ~ pi_theta(a_t | h_t, z_t).

        Returns:
            carry: (enc_carry, dyn_carry, dec_carry, act).
                dyn_carry: Model states.
                    Shape: ((B, D), (B, S, C)).
                act: Sampled actions, will be used as previous actions [a_{t-1}] at next timestep.
                    Shape: (B, act_dim).
            act: Sampled actions, will be passed to env.step().
                Shape: (B, act_dim).
            out: Extra info.
                finite: Per-element finiteness flags.
                enc/dyn/dec: Context entries for next batch initialization (only if replay_context is True).
        """
        (enc_carry, dyn_carry, dec_carry, prevact) = carry
        kw = dict(training=False, single=True)
        reset = obs['is_first']

        enc_carry, enc_entry, tokens = self.enc(enc_carry, obs, reset, **kw)
        dyn_carry, dyn_entry, feat = self.dyn.observe(
            dyn_carry,  # (h_{t-1}, z_{t-1})
            tokens,     # enc(x_t)
            prevact,    # a_{t-1}
            reset,
            **kw,
        )
        dec_entry = {}
        if dec_carry:
            dec_carry, dec_entry, _recons = self.dec(dec_carry, feat, reset, **kw)

        policy = self.pol(self.feat2tensor(feat), bdims=1)
        act = sample(policy)

        # extra info
        out = {}
        out['finite'] = elements.tree.flatdict(jax.tree.map(
            lambda x: jnp.isfinite(x).all(range(1, x.ndim)),
            dict(obs=obs, carry=carry, tokens=tokens, feat=feat, act=act),
        ))
        if self.config.replay_context:
            out.update(elements.tree.flatdict(dict(enc=enc_entry, dyn=dyn_entry, dec=dec_entry)))

        carry = (enc_carry, dyn_carry, dec_carry, act)

        return carry, act, out

    def train(self, carry, data):
        """Runs a single training step.

        Args:
            carry: Output from policy() call, i.e., single agent-environment interaction.
            data: Batch (from replay buffer).
                Should contain obs/act keys, stepid, enc/dyn/dec entries if replay_context is True.
                Shape: (B, R+T, ...)/(B, T, ...) if replay_context is True/False.
        """
        carry, obs, prevact, stepid = self._apply_replay_context(carry, data)

        # train all networks; world model and actor-critic
        metrics, (carry, entries, outs, mets) = self.opt(
            self.loss, carry, obs, prevact, training=True, has_aux=True,
        )
        metrics.update(mets)
        self.slowval.update()

        outs = {}
        if self.config.replay_context:
            updates = elements.tree.flatdict(dict(
                stepid=stepid, enc=entries[0], dyn=entries[1], dec=entries[2],
            ))
            B, T = obs['is_first'].shape
            assert all(x.shape[:2] == (B, T) for x in updates.values()), (
                (B, T), {k: v.shape for k, v in updates.items()}
            )
            outs['replay'] = updates
        # PER
        # if self.config.replay.fracs.priority > 0:
        #   outs['replay']['priority'] = losses['model']

        carry = (*carry, {k: data[k][:, -1] for k in self.act_space})

        return carry, outs, metrics

    def loss(self, carry, obs, prevact, training):
        """Compute all losses; world model and actor-critic."""

        # World model loss on real rollouts
        enc_carry, dyn_carry, dec_carry = carry
        reset = obs['is_first']
        B, T = reset.shape
        losses = {}
        metrics = {}

        # encode x_t to tokens
        enc_carry, enc_entries, tokens = self.enc(enc_carry, obs, reset, training)

        # L_dyn and L_rep
        dyn_carry, dyn_entries, los, repfeat, mets = self.dyn.loss(dyn_carry, tokens, prevact, reset, training)
        losses.update(los)
        metrics.update(mets)

        # decode s_t to x_t_hat
        dec_carry, dec_entries, recons = self.dec(dec_carry, repfeat, reset, training)

        # L_rew
        inp = sg(self.feat2tensor(repfeat), skip=self.config.reward_grad)
        losses['rew'] = self.rew(inp, 2).loss(obs['reward'])

        # L_con
        con = f32(~obs['is_terminal'])
        if self.config.contdisc:
            con *= 1 - 1 / self.config.horizon
        losses['con'] = self.con(self.feat2tensor(repfeat), 2).loss(con)

        # L_rec
        for key, recon in recons.items():
            space, value = self.obs_space[key], obs[key]
            assert value.dtype == space.dtype, (key, space, value.dtype)
            target = f32(value) / 255 if isimage(space) else value
            losses[key] = recon.loss(sg(target))

        shapes = {k: v.shape for k, v in losses.items()}
        assert all(x == (B, T) for x in shapes.values()), ((B, T), shapes)

        # Actor-Critic loss on imagination rollouts
        K = min(self.config.imag_last or T, T)
        H = self.config.imag_length
        policyfn = lambda feat: sample(self.pol(self.feat2tensor(feat), 1))

        # imagine H timesteps from K starting model states
        starts = self.dyn.starts(dyn_entries, dyn_carry, K)                       # sample K starts from repfeat
        _, imgfeat, imgprevact = self.dyn.imagine(starts, policyfn, H, training)  # imagine H timesteps

        # prepend real starts to imagined rollouts
        # since actor-critic loss requires H+1 steps starting from real model states
        first = jax.tree.map(lambda x: x[:, -K:].reshape((B * K, 1, *x.shape[2:])), repfeat)  # s_t
        imgfeat = concat([sg(first, skip=self.config.ac_grads), sg(imgfeat)], 1)
        lastact = policyfn(jax.tree.map(lambda x: x[:, -1], imgfeat))
        lastact = jax.tree.map(lambda x: x[:, None], lastact)
        imgact = concat([imgprevact, lastact], 1)
        assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgfeat))  # (B*K, H+1)
        assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgact))   # (B*K, H+1)

        # L_policy, L_value: actor-critic loss on imagined rollouts
        inp = self.feat2tensor(imgfeat)
        los, imgloss_out, mets = imag_loss(
            imgact,
            self.rew(inp, 2).pred(),
            self.con(inp, 2).prob(1),
            self.pol(inp, 2),
            self.val(inp, 2),
            self.slowval(inp, 2),
            self.retnorm, self.valnorm, self.advnorm,
            update=training,
            contdisc=self.config.contdisc,
            horizon=self.config.horizon,
            **self.config.imag_loss,
        )
        losses.update({k: v.mean(1).reshape((B, K)) for k, v in los.items()})
        metrics.update(mets)

        # L_repval: critic loss on replays
        if self.config.repval_loss:
            feat = sg(repfeat, skip=self.config.repval_grad)
            last, term, rew = [obs[k] for k in ('is_last', 'is_terminal', 'reward')]
            boot = imgloss_out['ret'][:, 0].reshape(B, K)
            feat, last, term, rew, boot = jax.tree.map(
                lambda x: x[:, -K:], (feat, last, term, rew, boot)  # last K steps of real rollouts
            )
            inp = self.feat2tensor(feat)
            los, _reploss_out, mets = repl_loss(
                last, term, rew, boot,
                self.val(inp, 2),
                self.slowval(inp, 2),
                self.valnorm,
                update=training,
                horizon=self.config.horizon,
                **self.config.repl_loss,
            )
            losses.update(los)
            metrics.update(prefix(mets, 'reploss'))

        assert set(losses.keys()) == set(self.scales.keys()), (
            sorted(losses.keys()), sorted(self.scales.keys())
        )
        metrics.update({f'loss/{k}': v.mean() for k, v in losses.items()})
        loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])

        carry = (enc_carry, dyn_carry, dec_carry)
        entries = (enc_entries, dyn_entries, dec_entries)
        outs = {'tokens': tokens, 'repfeat': repfeat, 'losses': losses}

        return loss, (carry, entries, outs, metrics)

    def report(self, carry, data):
        """Generates evaluation metrics and open-loop video predictions for logging."""
        if not self.config.report:
            return carry, {}

        carry, obs, prevact, _ = self._apply_replay_context(carry, data)
        (enc_carry, dyn_carry, dec_carry) = carry
        B, T = obs['is_first'].shape
        RB = min(6, B)
        metrics = {}

        # train metrics
        _, (new_carry, _entries, outs, mets) = self.loss(carry, obs, prevact, training=False)
        metrics.update(mets)

        # grad norms
        if self.config.report_gradnorms:
            for key in self.scales:
                try:
                    lossfn = lambda data, carry, key=key: self.loss(
                        carry, obs, prevact, training=False,
                    )[1][2]['losses'][key].mean()
                    grad = nj.grad(lossfn, self.modules)(data, carry)[-1]
                    metrics[f'gradnorm/{key}'] = optax.global_norm(grad)
                except KeyError:
                    print(f'Skipping gradnorm summary for missing loss: {key}')

        # open loop: world model monitoring (observe first half, imagine second half)
        firsthalf = lambda xs: jax.tree.map(lambda x: x[:RB, :T // 2], xs)
        secondhalf = lambda xs: jax.tree.map(lambda x: x[:RB, T // 2:], xs)
        dyn_carry = jax.tree.map(lambda x: x[:RB], dyn_carry)
        dec_carry = jax.tree.map(lambda x: x[:RB], dec_carry)
        dyn_carry, _, obsfeat = self.dyn.observe(
            dyn_carry, firsthalf(outs['tokens']), firsthalf(prevact), firsthalf(obs['is_first']), training=False,
        )
        _, imgfeat, _ = self.dyn.imagine(
            dyn_carry, secondhalf(prevact), length=T - T // 2, training=False,
        )
        dec_carry, _, obsrecons = self.dec(
            dec_carry, obsfeat, firsthalf(obs['is_first']), training=False,
        )
        dec_carry, _, imgrecons = self.dec(
            dec_carry, imgfeat, jnp.zeros_like(secondhalf(obs['is_first'])), training=False,
        )
        # visualize open-loop results
        for key in self.dec.imgkeys:
            assert obs[key].dtype == jnp.uint8

            # assemble [true | pred | error] frames
            true = obs[key][:RB]
            pred = jnp.concatenate([obsrecons[key].pred(), imgrecons[key].pred()], 1)
            pred = jnp.clip(pred * 255, 0, 255).astype(jnp.uint8)
            error = ((i32(pred) - i32(true) + 255) / 2).astype(np.uint8)
            video = jnp.concatenate([true, pred, error], 2)

            # add colored border (green: observe, red: imagine) and black padding
            video = jnp.pad(video, [[0, 0], [0, 0], [2, 2], [2, 2], [0, 0]])
            mask = jnp.zeros(video.shape, bool).at[:, :, 2:-2, 2:-2, :].set(True)
            border = jnp.full((T, 3), jnp.array([0, 255, 0]), jnp.uint8)
            border = border.at[T // 2:].set(jnp.array([255, 0, 0], jnp.uint8))
            video = jnp.where(mask, video, border[None, :, None, None, :])
            video = jnp.concatenate([video, 0 * video[:, :10]], 1)

            # reshape to (T, H, RB*W, C) grid and store
            B, T, H, W, C = video.shape
            grid = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
            metrics[f'openloop/{key}'] = grid

        carry = (*new_carry, {k: data[k][:, -1] for k in self.act_space})

        return carry, metrics

    def _apply_replay_context(self, carry, data):
        """Initializes starting carries in training loop with optional recovery logic to keep context.

        Uses the first R stored timesteps to recover a warm carry via truncate(),
        replacing cold zero-carry at episode starts.

        Previous actions are formatted via prepend.
        """
        (enc_carry, dyn_carry, dec_carry, prevact) = carry
        carry = (enc_carry, dyn_carry, dec_carry)
        stepid = data['stepid']
        obs = {k: data[k] for k in self.obs_space}
        prepend = lambda x, y: jnp.concatenate([x[:, None], y[:, :-1]], 1)
        prevact = {k: prepend(prevact[k], data[k]) for k in self.act_space}

        if not self.config.replay_context:
            return carry, obs, prevact, stepid

        # use first R timesteps as context to recover carry via truncate()
        # so that remaining T timesteps are trained
        R = self.config.replay_context
        nested = elements.tree.nestdict(data)
        entries = [nested.get(k, {}) for k in ('enc', 'dyn', 'dec')]
        lhs = lambda xs: jax.tree.map(lambda x: x[:, :R], xs)  # first R timesteps
        rhs = lambda xs: jax.tree.map(lambda x: x[:, R:], xs)  # remaining T timesteps
        rep_carry = (
            self.enc.truncate(lhs(entries[0]), enc_carry),
            self.dyn.truncate(lhs(entries[1]), dyn_carry),
            self.dec.truncate(lhs(entries[2]), dec_carry),
        )
        rep_obs = {k: rhs(data[k]) for k in self.obs_space}
        rep_prevact = {k: data[k][:, R - 1: -1] for k in self.act_space}  # a_{t-1} for rhs window
        rep_stepid = rhs(stepid)

        first_chunk = (data['consec'][:, 0] == 0)
        # first_chunk=True (episode start): use recovered carry
        # first_chunk=False (mid-episode): use normal carry
        carry, obs, prevact, stepid = jax.tree.map(
            lambda normal, replay: nn.where(first_chunk, replay, normal),
            (carry, rhs(obs), rhs(prevact), rhs(stepid)),
            (rep_carry, rep_obs, rep_prevact, rep_stepid),
        )

        return carry, obs, prevact, stepid

    def _make_opt(
        self,
        lr: float = 4e-5,
        agc: float = 0.3,
        eps: float = 1e-20,
        beta1: float = 0.9,
        beta2: float = 0.999,
        momentum: bool = True,
        nesterov: bool = False,
        wd: float = 0.0,
        wdregex: str = r'/kernel$',
        schedule: str = 'const',
        warmup: int = 1000,
        anneal: int = 0,
    ):
        chain = []
        chain.append(embodied.jax.opt.clip_by_agc(agc))
        chain.append(embodied.jax.opt.scale_by_rms(beta2, eps))
        chain.append(embodied.jax.opt.scale_by_momentum(beta1, nesterov))

        if wd:
            assert not wdregex[0].isnumeric(), wdregex
            pattern = re.compile(wdregex)
            wdmask = lambda params: {k: bool(pattern.search(k)) for k in params}
            chain.append(optax.add_decayed_weights(wd, wdmask))

        assert anneal > 0 or schedule == 'const'
        if schedule == 'const':
            sched = optax.constant_schedule(lr)
        elif schedule == 'linear':
            sched = optax.linear_schedule(lr, 0.1 * lr, anneal - warmup)
        elif schedule == 'cosine':
            sched = optax.cosine_decay_schedule(lr, anneal - warmup, 0.1 * lr)
        else:
            raise NotImplementedError(schedule)

        if warmup:
            ramp = optax.linear_schedule(0.0, lr, warmup)
            sched = optax.join_schedules([ramp, sched], [warmup])

        chain.append(optax.scale_by_learning_rate(sched))

        return optax.chain(*chain)


def imag_loss(
    act, rew, con,
    policy, value, slowvalue,
    retnorm, valnorm, advnorm,
    update,
    contdisc=True,
    slowtar=True,
    horizon=333,
    lam=0.95,
    actent=3e-4,
    slowreg=1.0,
):
    """Compute actor-critic loss on imagined rollouts."""
    losses = {}
    metrics = {}

    voffset, vscale = valnorm.stats()
    val = value.pred() * vscale + voffset
    slowval = slowvalue.pred() * vscale + voffset
    tarval = slowval if slowtar else val

    disc = 1 if contdisc else 1 - 1 / horizon
    weight = jnp.cumprod(disc * con, 1) / disc
    last = jnp.zeros_like(con)
    term = 1 - con
    ret = lambda_return(last, term, rew, tarval, tarval, disc, lam)

    roffset, rscale = retnorm(ret, update)
    adv = (ret - tarval[:, :-1]) / rscale
    aoffset, ascale = advnorm(adv, update)
    adv_normed = (adv - aoffset) / ascale

    logpi = sum([v.logp(sg(act[k]))[:, :-1] for k, v in policy.items()])
    ents = {k: v.entropy()[:, :-1] for k, v in policy.items()}
    policy_loss = sg(weight[:, :-1]) * -(logpi * sg(adv_normed) + actent * sum(ents.values()))
    losses['policy'] = policy_loss

    voffset, vscale = valnorm(ret, update)
    tar_normed = (ret - voffset) / vscale
    tar_padded = jnp.concatenate([tar_normed, 0 * tar_normed[:, -1:]], 1)
    losses['value'] = sg(weight[:, :-1]) * (
        value.loss(sg(tar_padded)) + slowreg * value.loss(sg(slowvalue.pred()))
    )[:, :-1]

    ret_normed = (ret - roffset) / rscale
    metrics['adv'] = adv.mean()
    metrics['adv_std'] = adv.std()
    metrics['adv_mag'] = jnp.abs(adv).mean()
    metrics['rew'] = rew.mean()
    metrics['con'] = con.mean()
    metrics['ret'] = ret_normed.mean()
    metrics['val'] = val.mean()
    metrics['tar'] = tar_normed.mean()
    metrics['weight'] = weight.mean()
    metrics['slowval'] = slowval.mean()
    metrics['ret_min'] = ret_normed.min()
    metrics['ret_max'] = ret_normed.max()
    metrics['ret_rate'] = (jnp.abs(ret_normed) >= 1.0).mean()

    for k in act:
        metrics[f'ent/{k}'] = ents[k].mean()
        if hasattr(policy[k], 'minent'):
            lo, hi = policy[k].minent, policy[k].maxent
            metrics[f'rand/{k}'] = (ents[k].mean() - lo) / (hi - lo)

    outs = {}
    outs['ret'] = ret

    return losses, outs, metrics


def repl_loss(
    last, term, rew, boot,
    value, slowvalue, valnorm,
    update=True,
    slowreg=1.0,
    slowtar=True,
    horizon=333,
    lam=0.95,
):
    """Compute critic loss on real (replay) rollouts."""
    losses = {}

    voffset, vscale = valnorm.stats()
    val = value.pred() * vscale + voffset
    slowval = slowvalue.pred() * vscale + voffset
    tarval = slowval if slowtar else val

    disc = 1 - 1 / horizon
    weight = f32(~last)
    ret = lambda_return(last, term, rew, tarval, boot, disc, lam)

    voffset, vscale = valnorm(ret, update)
    ret_normed = (ret - voffset) / vscale
    ret_padded = jnp.concatenate([ret_normed, 0 * ret_normed[:, -1:]], 1)
    losses['repval'] = weight[:, :-1] * (
        value.loss(sg(ret_padded)) + slowreg * value.loss(sg(slowvalue.pred()))
    )[:, :-1]

    outs = {}
    outs['ret'] = ret

    metrics = {}

    return losses, outs, metrics


def lambda_return(last, term, rew, val, boot, disc, lam):
    chex.assert_equal_shape((last, term, rew, val, boot))
    rets = [boot[:, -1]]
    live = (1 - f32(term))[:, 1:] * disc
    cont = (1 - f32(last))[:, 1:] * lam
    interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
    for t in reversed(range(live.shape[1])):
        rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
    return jnp.stack(list(reversed(rets))[:-1], 1)
