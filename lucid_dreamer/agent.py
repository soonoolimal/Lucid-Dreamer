import elements
import jax
import jax.numpy as jnp

from dreamerv3.agent import (
    Agent,
    imag_loss, repl_loss, lambda_return,
    f32, sg, sample, prefix, concat, isimage,
)


class LucidDreamerAgent(Agent):
    """Inception-based Dreamerv3 agent."""
    def train(self, carry, data, real=False):
        carry, obs, prevact, stepid = self._apply_replay_context(carry, data)

        metrics, (carry, entries, outs, mets) = self.opt(
            self.loss, carry, obs, prevact, training=True, real=real, has_aux=True,
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

        carry = (*carry, {k: data[k][:, -1] for k in self.act_space})

        return carry, outs, metrics

    def loss(self, carry, obs, prevact, training, real=False):
        """Compute all losses.

        World model loss is always computed on real rollouts.
        Actor-Critic loss is computed on real/imagined transitions if real=True/False.
        """
        enc_carry, dyn_carry, dec_carry = carry
        reset = obs['is_first']
        B, T = reset.shape
        losses = {}
        metrics = {}

        enc_carry, enc_entries, tokens = self.enc(enc_carry, obs, reset, training)
        dyn_carry, dyn_entries, los, repfeat, mets = self.dyn.loss(dyn_carry, tokens, prevact, reset, training)
        losses.update(los)
        metrics.update(mets)
        dec_carry, dec_entries, recons = self.dec(dec_carry, repfeat, reset, training)

        inp = sg(self.feat2tensor(repfeat), skip=self.config.reward_grad)
        losses['rew'] = self.rew(inp, 2).loss(obs['reward'])

        con = f32(~obs['is_terminal'])
        if self.config.contdisc:
            con *= 1 - 1 / self.config.horizon
        losses['con'] = self.con(self.feat2tensor(repfeat), 2).loss(con)

        for key, recon in recons.items():
            space, value = self.obs_space[key], obs[key]
            assert value.dtype == space.dtype, (key, space, value.dtype)
            target = f32(value) / 255 if isimage(space) else value
            losses[key] = recon.loss(sg(target))

        shapes = {k: v.shape for k, v in losses.items()}
        assert all(x == (B, T) for x in shapes.values()), ((B, T), shapes)

        K = min(self.config.imag_last or T, T)

        if not real:
            H = self.config.imag_length
            policyfn = lambda feat: sample(self.pol(self.feat2tensor(feat), 1))
            starts = self.dyn.starts(dyn_entries, dyn_carry, K)
            _, imgfeat, imgprevact = self.dyn.imagine(starts, policyfn, H, training)

            first = jax.tree.map(lambda x: x[:, -K:].reshape((B * K, 1, *x.shape[2:])), repfeat)
            imgfeat = concat([sg(first, skip=self.config.ac_grads), sg(imgfeat)], 1)
            lastact = policyfn(jax.tree.map(lambda x: x[:, -1], imgfeat))
            lastact = jax.tree.map(lambda x: x[:, None], lastact)
            imgact = concat([imgprevact, lastact], 1)

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
            boot_ret = imgloss_out['ret'][:, 0].reshape(B, K)
        else:
            los, mets, boot_ret = _real_ac_loss(
                repfeat, prevact, obs,
                self.pol, self.val, self.slowval,
                self.feat2tensor,
                self.retnorm, self.valnorm, self.advnorm,
                K=K, B=B,
                update=training,
                contdisc=self.config.contdisc,
                horizon=self.config.horizon,
                ac_grads=self.config.ac_grads,
                **self.config.imag_loss,
            )
            losses.update(los)
            metrics.update(mets)

        if self.config.repval_loss:
            feat = sg(repfeat, skip=self.config.repval_grad)
            last, term, rew = [obs[k] for k in ('is_last', 'is_terminal', 'reward')]
            boot = boot_ret
            feat, last, term, rew, boot = jax.tree.map(
                lambda x: x[:, -K:], (feat, last, term, rew, boot)
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


def _real_ac_loss(
    repfeat, prevact, obs,
    pol, val, slowval,
    feat2tensor,
    retnorm, valnorm, advnorm,
    K, B,
    update,
    contdisc,
    horizon,
    slowtar,
    lam,
    actent,
    slowreg,
    ac_grads=False,
):
    """Compute actor-critic loss directly on real replay transitions.

    Real action at state t is prevact[:, t+1];
        prevact[:, t] holds the action taken at t-1,
        so the action actually executed at state t sits one step ahead.

    Boot_ret uses slow critic values on all K states;
        each position can be an episode boundary (is_last=True),
        so boot[:, t] must be available at every t for repval_loss lambda-return.

    Args:
        B: Number of real sequences sampled replay buffer.
        K: Uses K-1 real transitions per sequence.
            The Kth state is bootstrap-only (used as tarval[:, -1] for lambda-return computation);
            policy and value are evaluated on the preceding K-1 states.

    Returns:
        losses: Shape: (B, K-1).
        boot_ret: For repval_loss. Shape: (B, K).
    """
    feat = sg(repfeat) if not ac_grads else repfeat
    feat_all = jax.tree.map(lambda x: x[:, -K:], feat)  # (B, K, D)

    inp_all = feat2tensor(feat_all)  # (B, K, feat_dim)
    inp_states = inp_all[:, :-1]     # (B, K-1, feat_dim); used for both actor and critic loss
    # inp_all[:, -1] is bootstrap-only (no action available for the last state)

    shifted_act = {k: prevact[k][:, -(K - 1):] for k in prevact}  # (B, K-1)
    rew = obs['reward'][:, -(K - 1):]                             # (B, K-1)
    is_terminal = obs['is_terminal'][:, -(K - 1):]
    is_last = obs['is_last'][:, -(K - 1):]

    # value predictions on all K states (K-1 policy states + 1 bootstrap)
    voffset, vscale = valnorm.stats()
    slowval_pred = slowval(inp_all, 2).pred() * vscale + voffset  # (B, K)
    tarval = slowval_pred if slowtar else val(inp_all, 2).pred() * vscale + voffset

    # lambda-return over K-1 real steps with bootstrap at step K
    disc = 1 if contdisc else 1 - 1 / horizon
    last_pad = jnp.concatenate([f32(is_last), jnp.ones((B, 1))], 1)      # (B, K)
    term_pad = jnp.concatenate([f32(is_terminal), jnp.ones((B, 1))], 1)  # (B, K)
    rew_pad = jnp.concatenate([rew, jnp.zeros((B, 1))], 1)               # (B, K)
    ret = lambda_return(last_pad, term_pad, rew_pad, tarval, tarval, disc, lam)  # (B, K-1)

    roffset, rscale = retnorm(ret, update)
    ret_normed = (ret - roffset) / rscale

    # advantage
    adv = (ret - tarval[:, :-1]) / rscale
    aoffset, ascale = advnorm(adv, update)
    adv_normed = (adv - aoffset) / ascale

    # actor loss
    # no cumulative discount weight (unlike imag_loss): real transitions are equally reliable
    policy = pol(inp_states, 2)
    logpi = sum([v.logp(sg(shifted_act[k])) for k, v in policy.items()])   # (B, K-1)
    ents = {k: v.entropy() for k, v in policy.items()}
    policy_loss = -(logpi * sg(adv_normed) + actent * sum(ents.values()))  # (B, K-1)

    # critic distributions; used for value_loss and val_metric
    # must be computed before valnorm update (val_metric uses pre-update scale)
    val_dist = val(inp_states, 2)
    slowval_dist = slowval(inp_states, 2)
    val_metric = (val_dist.pred() * vscale + voffset).mean()

    # critic loss
    voffset, vscale = valnorm(ret, update)
    tar_normed = (ret - voffset) / vscale  # (B, K-1)
    value_loss = (
        val_dist.loss(sg(tar_normed)) +
        slowreg * val_dist.loss(sg(slowval_dist.pred()))
    )  # (B, K-1)

    losses = {'policy': policy_loss, 'value': value_loss}

    metrics = {
        'real_ac/adv': adv.mean(),
        'real_ac/adv_std': adv.std(),
        'real_ac/rew': rew.mean(),
        'real_ac/ret': ret_normed.mean(),
        'real_ac/val': val_metric,
    }
    for k in shifted_act:
        metrics[f'real_ac/ent/{k}'] = ents[k].mean()

    boot_ret = slowval_pred  # (B, K); slow critic bootstrap for repval_loss

    return losses, metrics, boot_ret
