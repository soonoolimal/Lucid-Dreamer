from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

import wandb


class BaseTrainer(ABC):
    def __init__(self, model_name, train_loader, valid_loader, device, cfg):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self._bind(cfg, ['num_epochs', 'patience', 'epoch_save_interval', 'load_only'])

        ckpt_dir = Path(cfg['ckpt_dir'])
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        prefix = f"{model_name}_s{cfg['seed']}"
        self.best_ckpt_path = ckpt_dir / f'{prefix}_best.pt'
        self._epoch_ckpt_fmt = str(ckpt_dir / f'{prefix}_epoch{{}}.pt')

        if not self.load_only:
            if self.best_ckpt_path.exists():
                raise FileExistsError(f'Checkpoint already exists: {self.best_ckpt_path}')

            Path(cfg['log_dir']).mkdir(parents=True, exist_ok=True)

            wandb.init(
                project=cfg['wandb_project'],
                group=cfg['wandb_group'],
                name=cfg['wandb_run_name'],
                config=cfg['wandb_config'],
            )

    @property
    @abstractmethod
    def tag(self) -> str:
        """Short prefix used in tqdm.write, e.g., 'DyE' or 'DyP'."""

    @property
    @abstractmethod
    def val_loss_key(self) -> str:
        """Key in metrics dict used for early stopping."""

    @property
    @abstractmethod
    def model(self) -> nn.Module:
        """Model whose state_dict is saved as checkpoint."""

    @abstractmethod
    def _run_epoch(self, loader, train) -> dict:
        """Run one epoch, return metrics dict."""

    @abstractmethod
    def _build_log(self, epoch, train_metrics, valid_metrics) -> dict:
        """Build wandb log dict for this epoch."""

    def train(self):
        best_val_loss = float('inf')
        best_valid_metrics = {}
        epochs_no_improve = 0

        epoch_pbar = tqdm(
            range(1, self.num_epochs + 1),
            desc=f'[{self.tag}]',
            dynamic_ncols=True,
        )

        for epoch in epoch_pbar:
            train_metrics = self._run_epoch(self.train_loader, train=True)
            valid_metrics = self._run_epoch(self.valid_loader, train=False)

            if not self.load_only:
                wandb.log(self._build_log(epoch, train_metrics, valid_metrics), step=epoch)

            val_loss = valid_metrics[self.val_loss_key]

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_valid_metrics = valid_metrics
                epochs_no_improve = 0
                self._save_best()
                tqdm.write(
                    f'[{self.tag}] Epoch {epoch:02d} | Valid loss improved -> {val_loss:.4f}\n'
                    f'Saved: {self.best_ckpt_path}'
                )
            else:
                epochs_no_improve += 1
                tqdm.write(
                    f'[{self.tag}] Epoch {epoch:02d} | Valid loss {val_loss:.4f} | '
                    f'(Best {best_val_loss:.4f}, no improve {epochs_no_improve}/{self.patience})'
                )
                if epochs_no_improve >= self.patience:
                    tqdm.write(f'[{self.tag}] Early stopping triggered at epoch {epoch}')
                    break

            if epoch % self.epoch_save_interval == 0:
                self._save_epoch(epoch)

            epoch_pbar.set_postfix(dict(
                train_loss=f'{train_metrics[self.val_loss_key]:.4f}',
                valid_loss=f'{val_loss:.4f}',
                best=f'{best_val_loss:.4f}',
                **self._extra_postfix(best_valid_metrics),
            ))

        if not self.load_only:
            wandb.finish()

        tqdm.write(f'[{self.tag}] Done. Best valid loss: {best_val_loss:.4f}')

    def load_best(self):
        state = torch.load(self.best_ckpt_path, map_location=self.device)
        self.model.load_state_dict(state)
        print(f'[{self.tag}] Loaded: {self.best_ckpt_path}')

    def _extra_postfix(self, best_valid_metrics) -> dict:
        """Extra items to add to the epoch progress bar. Override in subclasses."""
        return {}

    def _bind(self, cfg, keys):
        for k in keys:
            setattr(self, k, cfg[k])

    def _save_best(self):
        torch.save(self.model.state_dict(), self.best_ckpt_path)

    def _save_epoch(self, epoch):
        torch.save(self.model.state_dict(), self._epoch_ckpt_fmt.format(epoch))
