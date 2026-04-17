import torch
import torch.nn.functional as F
from base_trainer import BaseTrainer
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from inception.models.dyn_encoder import DTOutput


class DTEncTrainer(BaseTrainer):
    def __init__(self, dye, train_loader, valid_loader, device, cfg):
        cfg['hidden_size'] = dye.gpt2.config.n_embd
        cfg['wandb_config'] = {k: cfg[k] for k in (
            'hidden_size', 'lr', 'weight_decay', 'betas', 'loss_mode',
            'lambda_act', 'lambda_rtg', 'lambda_obs', 'lambda_var',
            'grad_clip_act', 'grad_clip_rtg', 'grad_clip_obs',
            'log_grad_norm', 'stop_gradient', 'num_epochs', 'patience',
        )}
        super().__init__(type(dye).__name__, train_loader, valid_loader, device, cfg)

        self.dye = dye.to(device)
        self._bind(cfg, [
            'loss_mode',
            'lambda_act', 'lambda_rtg', 'lambda_obs', 'lambda_var',
            'grad_clip_act', 'grad_clip_rtg', 'grad_clip_obs',
            'log_grad_norm', 'stop_gradient',
        ])
        self.optimizer = self._configure_optimizer(cfg['lr'], cfg['weight_decay'], cfg['betas'])

    @property
    def tag(self):
        return 'DTEnc'

    @property
    def val_loss_key(self):
        return 'total_loss'

    @property
    def model(self):
        return self.dye

    def _build_log(self, epoch, train_metrics, valid_metrics):
        return {
            'epoch': epoch,
            **{f'train/{k}': v for k, v in train_metrics.items()},
            **{f'valid/{k}': v for k, v in valid_metrics.items()},
        }

    def _extra_postfix(self, best_valid_metrics):
        return {
            f'valid_{k}_at_best': f'{best_valid_metrics[k]:.4f}'
            for k in ('act_loss', 'rtg_loss', 'obs_loss', 'obs_var_loss')
            if k in best_valid_metrics
        }

    def _run_epoch(self, loader, train):
        self.dye.train(train)
        context = torch.enable_grad() if train else torch.no_grad()

        total_loss_sum = act_loss_sum = rtg_loss_sum = obs_loss_sum = obs_var_loss_sum = 0.0
        act_gn_sum = rtg_gn_sum = obs_gn_sum = 0.0
        n_batches = 0

        batch_pbar = tqdm(
            loader,
            desc='Train' if train else 'Valid',
            leave=False,
            dynamic_ncols=True,
        )

        with context:
            for batch in batch_pbar:
                observations = batch['observations'].to(self.device)
                actions = batch['actions'].to(self.device)
                rewards = batch['rewards'].to(self.device)
                returns_to_go = batch['returns_to_go'].to(self.device)
                timesteps = batch['timesteps'].to(self.device)
                mask = batch['mask'].to(self.device)

                enc_out: DTOutput = self.dye(observations, actions, rewards, returns_to_go, timesteps, mask)

                obs_enc = enc_out.obs_enc.detach() if self.stop_gradient else enc_out.obs_enc

                total_loss, act_loss, rtg_loss, obs_loss, obs_var_loss = self._compute_loss(
                    enc_out.ac_logits, enc_out.rtg_preds, enc_out.obs_preds,
                    obs_enc, actions, returns_to_go,
                )

                if train:
                    self.optimizer.zero_grad()

                    # joint: single backward pass
                    if self.loss_mode == 'joint':
                        total_loss.backward()
                    # separate: per-loss backward + grad clip, then combine
                    else:
                        act_loss.backward(retain_graph=True)
                        act_gn = clip_grad_norm_(self.dye.parameters(), max_norm=self.grad_clip_act)
                        act_grads = {n: p.grad.clone() for n, p in self.dye.named_parameters() if p.grad is not None}

                        self.optimizer.zero_grad()
                        rtg_loss.backward(retain_graph=True)
                        rtg_gn = clip_grad_norm_(self.dye.parameters(), max_norm=self.grad_clip_rtg)
                        rtg_grads = {n: p.grad.clone() for n, p in self.dye.named_parameters() if p.grad is not None}

                        self.optimizer.zero_grad()
                        (obs_loss + obs_var_loss).backward()
                        obs_gn = clip_grad_norm_(self.dye.parameters(), max_norm=self.grad_clip_obs)

                        for n, p in self.dye.named_parameters():
                            act_g = act_grads.get(n)
                            rtg_g = rtg_grads.get(n)
                            if act_g is None and rtg_g is None:
                                continue
                            combined = (
                                (act_g if act_g is not None else torch.zeros_like(p))
                                + (rtg_g if rtg_g is not None else torch.zeros_like(p))
                            )
                            if p.grad is not None:
                                p.grad += combined
                            else:
                                p.grad = combined

                        if self.log_grad_norm:
                            act_gn_sum += act_gn.item()
                            rtg_gn_sum += rtg_gn.item()
                            obs_gn_sum += obs_gn.item()

                    self.optimizer.step()

                total_loss_sum += total_loss.item()
                act_loss_sum += act_loss.item()
                rtg_loss_sum += rtg_loss.item()
                obs_loss_sum += obs_loss.item()
                obs_var_loss_sum += obs_var_loss.item()
                n_batches += 1
                batch_pbar.set_postfix(dict(
                    total_loss=f'{total_loss.item():.4f}',
                    act_loss=f'{act_loss.item():.4f}',
                    rtg_loss=f'{rtg_loss.item():.4f}',
                    obs_loss=f'{obs_loss.item():.4f}',
                    var_loss=f'{obs_var_loss.item():.4f}',
                ))

        metrics = dict(
            total_loss=total_loss_sum / n_batches,
            act_loss=act_loss_sum / n_batches,
            rtg_loss=rtg_loss_sum / n_batches,
            obs_loss=obs_loss_sum / n_batches,
            obs_var_loss=obs_var_loss_sum / n_batches,
        )
        if train and self.log_grad_norm and self.loss_mode == 'separate':
            metrics['grad_norm_act'] = act_gn_sum / n_batches
            metrics['grad_norm_rtg'] = rtg_gn_sum / n_batches
            metrics['grad_norm_obs'] = obs_gn_sum / n_batches

        return metrics

    def _compute_loss(self, ac_logits, rtg_preds, obs_preds, obs_enc, actions, returns_to_go):
        # act_loss: CE/MSE loss if discrete/continuous on on a_t predicted from h(obs_t)
        act_preds = ac_logits.reshape(-1, ac_logits.size(-1))
        if self.dye.is_discrete:
            act_trues = actions.reshape(-1)
            act_loss = F.cross_entropy(act_preds, act_trues) * self.lambda_act
        else:
            act_trues = actions.reshape(-1, actions.size(-1))
            act_loss = F.mse_loss(act_preds, act_trues) * self.lambda_act

        # rtg_loss: MSE loss on R_{t+1} predicted from h(act_t), normalized by std
        rtg_trues = returns_to_go[:, 1:, :]
        rtg_std = rtg_trues.std().clamp(min=1.0)
        rtg_loss = F.mse_loss(rtg_preds[:, :-1, :] / rtg_std, rtg_trues / rtg_std) * self.lambda_rtg

        # obs_loss: MSE loss on obs_enc[:, 1:] predicted from h(act_t)
        obs_loss = F.mse_loss(obs_preds[:, :-1, :], obs_enc[:, 1:, :]) * self.lambda_obs

        # obs_var_loss: VICReg-style variance regularization to prevent obs_enc collapse
        obs_flat = obs_enc.reshape(-1, obs_enc.size(-1))
        std = torch.sqrt(obs_flat.var(dim=0) + 1e-4)
        obs_var_loss = F.relu(1 - std).mean() * self.lambda_var

        total_loss = act_loss + rtg_loss + obs_loss + obs_var_loss

        return total_loss, act_loss, rtg_loss, obs_loss, obs_var_loss

    def _configure_optimizer(self, lr, weight_decay, betas):
        decay_params, no_decay_params = [], []
        for p in self.dye.parameters():
            if not p.requires_grad:
                continue
            (decay_params if p.dim() >= 2 else no_decay_params).append(p)
        return torch.optim.AdamW(
            [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': no_decay_params, 'weight_decay': 0.0},
            ],
            lr=lr,
            weight_decay=0.0,
            betas=betas,
        )


class DTPredTrainer(BaseTrainer):
    def __init__(self, dye, dyp, train_loader, valid_loader, device, cfg):
        for p in dye.parameters():
            if p.requires_grad:
                raise ValueError(
                    'DyE parameters must be frozen before passing to DTPredTrainer. '
                    'Call dye.requires_grad_(False) first.'
                )

        cfg['wandb_config'] = {k: cfg[k] for k in (
            'n_dynamics', 'lr', 'weight_decay', 'betas', 'num_epochs', 'patience',
        )}
        super().__init__(type(dyp).__name__, train_loader, valid_loader, device, cfg)

        self.n_dynamics = cfg['n_dynamics']
        self.dye = dye.to(device)
        self.dyp = dyp.to(device)
        self.optimizer = torch.optim.AdamW(
            self.dyp.parameters(),
            lr=cfg['lr'],
            weight_decay=cfg['weight_decay'],
            betas=cfg['betas'],
        )

    @property
    def tag(self):
        return 'DTPred'

    @property
    def val_loss_key(self):
        return 'ce_loss'

    @property
    def model(self):
        return self.dyp

    def _build_log(self, epoch, train_metrics, valid_metrics):
        def to_log(metrics, phase):
            d = {f'{phase}/ce_loss': metrics['ce_loss'], f'{phase}/acc': metrics['acc']}
            for c in range(self.n_dynamics):
                d[f'{phase}/acc_class{c}'] = metrics[f'acc_class{c}']
            return d

        return {'epoch': epoch, **to_log(train_metrics, 'train'), **to_log(valid_metrics, 'valid')}

    def _extra_postfix(self, best_valid_metrics):
        if not best_valid_metrics:
            return {}
        return {'valid_acc_at_best': f'{best_valid_metrics["acc"]:.4f}'}

    def _run_epoch(self, loader, train):
        self.dye.eval()
        self.dyp.train(train)

        ce_loss_sum = 0.0
        n_batches = 0

        correct = 0
        n_samples = 0
        class_correct = torch.zeros(self.n_dynamics, dtype=torch.long)
        class_total = torch.zeros(self.n_dynamics, dtype=torch.long)

        batch_pbar = tqdm(
            loader,
            desc='Train' if train else 'Valid',
            leave=False,
            dynamic_ncols=True,
        )

        for batch in batch_pbar:
            observations = batch['observations'].to(self.device)
            actions = batch['actions'].to(self.device)
            rewards = batch['rewards'].to(self.device)
            returns_to_go = batch['returns_to_go'].to(self.device)
            timesteps = batch['timesteps'].to(self.device)
            mask = batch['mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            label = labels[:, -1]  # (B,) all timesteps share the same label

            with torch.no_grad():
                enc_out: DTOutput = self.dye(observations, actions, rewards, returns_to_go, timesteps, mask)

            if train:
                logits = self.dyp(enc_out)  # (B, n_dynamics)
                loss = F.cross_entropy(logits, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    logits = self.dyp(enc_out)
                    loss = F.cross_entropy(logits, label)

            ce_loss_sum += loss.item()
            n_batches += 1
            batch_pbar.set_postfix({'ce_loss': f'{loss.item():.4f}'})

            preds = logits.argmax(dim=-1)  # (B,)
            correct += (preds == label).sum().item()
            n_samples += label.size(0)

            for c in range(self.n_dynamics):
                mask_c = (label == c)
                class_correct[c] += (preds[mask_c] == c).sum().item()
                class_total[c] += mask_c.sum().item()

        per_class_acc = {
            f'acc_class{c}': (
                (class_correct[c] / class_total[c]).item() if class_total[c] > 0 else 0.0
            )
            for c in range(self.n_dynamics)
        }

        return dict(
            ce_loss=ce_loss_sum / n_batches,
            acc=correct / n_samples,
            **per_class_acc,
        )
