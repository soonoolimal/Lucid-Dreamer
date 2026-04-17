import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).parents[2]
SCRIPTS_INC = pathlib.Path(__file__).parent

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS_INC))

import torch
import torch.nn.functional as F
import wandb
import yaml
from tqdm import tqdm

import inception.models  # register all subclasses
import inception.utils.load as ldu
from inception.models.base import DynEncModel, DynPredModel, ObsEncModel

OBS_ENC_REGISTRY = {cls.__name__: cls for cls in ObsEncModel.__subclasses__()}
DYN_ENC_REGISTRY = {cls.__name__: cls for cls in DynEncModel.__subclasses__()}
DYN_PRED_REGISTRY = {cls.__name__: cls for cls in DynPredModel.__subclasses__()}

CKPT_DIR_TPL = 'logs/inception/{scn}/{ds_type}/{timestamp}'


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--scn', required=True)
    parser.add_argument('--ds_type', required=True, choices=['dreamer', 'random'])
    parser.add_argument('--timestamp', required=True, help='Training run timestamp (e.g. 260417_120000)')
    parser.add_argument('--timestamps', nargs='+', default=None, help='Per-dy_type HDF5 timestamps. Auto-detect if omitted.')
    parser.add_argument('--device', default=None)
    return parser.parse_args(argv)


def _compute_cm(preds, labels, n_dynamics):
    cm = torch.zeros(n_dynamics, n_dynamics, dtype=torch.long)
    for p, l in zip(preds, labels):
        cm[l, p] += 1
    return cm


def _print_results(cm, n_dynamics):
    acc = cm.diagonal().sum().item() / cm.sum().item()
    print(f'\nTest accuracy: {acc * 100:.2f}%')

    per_class = {}
    for c in range(n_dynamics):
        total = cm[c].sum().item()
        per_class[c] = cm[c, c].item() / total if total > 0 else 0.0
        print(f'  class {c}: {per_class[c] * 100:.2f}%')

    print('\nConfusion matrix (row=gt, col=pred):')
    print(cm)

    header = f"{'class':<8} {'precision':>10} {'recall':>10} {'f1':>10}"
    print('\n' + header)
    print('-' * len(header))

    precisions, recalls, f1s = [], [], []
    for c in range(n_dynamics):
        tp = cm[c, c].item()
        fp = cm[:, c].sum().item() - tp
        fn = cm[c, :].sum().item() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        print(f'{c:<8} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f}')

    macro_p = sum(precisions) / n_dynamics
    macro_r = sum(recalls) / n_dynamics
    macro_f1 = sum(f1s) / n_dynamics
    print(f"{'macro':<8} {macro_p:>10.4f} {macro_r:>10.4f} {macro_f1:>10.4f}")


def run_test(dye, dyp, test_loader, device, n_dynamics, wandb_cfg=None, logdir=None):
    """Run test inference and print results. Models must already be loaded."""
    dye.to(device).eval()
    dyp.to(device).eval()

    all_preds = []
    all_labels = []
    ce_loss_sum = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Test', dynamic_ncols=True):
            observations = batch['observations'].to(device)
            actions = batch['actions'].to(device)
            rewards = batch['rewards'].to(device)
            returns_to_go = batch['returns_to_go'].to(device)
            timesteps = batch['timesteps'].to(device)
            mask = batch['mask'].to(device)
            labels = batch['labels'][:, -1]

            enc_out = dye(observations, actions, rewards, returns_to_go, timesteps, mask)
            logits = dyp(enc_out)  # (B, n_dynamics)
            loss = F.cross_entropy(logits, labels.to(device))

            ce_loss_sum += loss.item()
            n_batches += 1
            all_preds.append(logits.argmax(dim=-1).cpu())
            all_labels.append(labels)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    ce_loss = ce_loss_sum / n_batches

    print(f'\nTest CE loss: {ce_loss:.4f}')
    cm = _compute_cm(all_preds, all_labels, n_dynamics)
    _print_results(cm, n_dynamics)

    acc = cm.diagonal().sum().item() / cm.sum().item()
    per_class_acc = {}
    for c in range(n_dynamics):
        total = cm[c].sum().item()
        per_class_acc[c] = cm[c, c].item() / total if total > 0 else 0.0

    if logdir is not None:
        metrics = {'ce_loss': ce_loss, 'acc': acc, **{f'acc_class{c}': v for c, v in per_class_acc.items()}}
        with open(pathlib.Path(logdir) / 'test_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

    if wandb_cfg is not None:
        log = {'test/ce_loss': ce_loss, 'test/acc': acc,
               **{f'test/acc_class{c}': v for c, v in per_class_acc.items()}}
        wandb.init(
            project=wandb_cfg['project'],
            group=wandb_cfg['group'],
            name=wandb_cfg['run_name'],
        )
        wandb.log(log)
        wandb.finish()


def main(argv=None):
    args = _parse_args(argv)
    run_dir = ROOT / CKPT_DIR_TPL.format(scn=args.scn, ds_type=args.ds_type, timestamp=args.timestamp)

    with open(ROOT / 'configs/inception.yaml') as f:
        cfg = yaml.safe_load(f)['inception']

    device = torch.device(args.device or cfg['device'])
    print(f'Device: {device}')
    print(f'Run dir: {run_dir}')

    # Offline data (test split only)
    data_cfg = cfg['data']
    _, _, test_sets = ldu.load_hdf5_datasets(
        paths=ldu.make_hdf5_paths(args.scn, args.ds_type, cfg['seed'], cfg['n_dynamics'], args.timestamps),
        gamma=data_cfg['gamma'],
        train_ratio=data_cfg['train_ratio'],
        valid_ratio=data_cfg['valid_ratio'],
    )

    max_ep_len = test_sets[0].max_ep_len
    n_act = test_sets[0].n_act
    is_discrete = test_sets[0].is_discrete

    test_loader = ldu.make_dataloader(
        test_sets, data_cfg['seq_len'], data_cfg['batch_size'],
        shuffle=False, drop_last=False, num_workers=data_cfg['num_workers'],
    )

    # Models
    obs_enc_cls_name = cfg['obs_enc']['select']
    obs_enc = OBS_ENC_REGISTRY[obs_enc_cls_name](**cfg['obs_enc'][obs_enc_cls_name])

    dyn_enc_cls_name = cfg['dyn_enc']['select']
    dyn_enc_kw = cfg['dyn_enc'][dyn_enc_cls_name]
    dye = DYN_ENC_REGISTRY[dyn_enc_cls_name](
        obs_encoder=obs_enc,
        act_dim=n_act,
        is_discrete=is_discrete,
        hidden_size=obs_enc.enc_dim,
        seq_len=data_cfg['seq_len'],
        max_ep_len=max_ep_len,
        **dyn_enc_kw,
    )

    dyn_pred_cls_name = cfg['dyn_pred']['select']
    dyn_pred_kw = cfg['dyn_pred'][dyn_pred_cls_name]
    dyp = DYN_PRED_REGISTRY[dyn_pred_cls_name](
        hidden_size=dye.gpt2.config.n_embd,
        last_k=dyn_enc_kw['last_k'],
        pass_h=dyn_enc_kw['pass_h'],
        n_dynamics=cfg['n_dynamics'],
        **dyn_pred_kw,
    )

    # Load checkpoints
    seed = cfg['seed']
    dye_ckpt = run_dir / f'{dyn_enc_cls_name}_s{seed}_best.pt'
    dyp_ckpt = run_dir / f'{dyn_pred_cls_name}_s{seed}_best.pt'

    dye.load_state_dict(torch.load(dye_ckpt, map_location=device))
    dyp.load_state_dict(torch.load(dyp_ckpt, map_location=device))
    print(f'Loaded DyE: {dye_ckpt}')
    print(f'Loaded DyP: {dyp_ckpt}')

    run_test(dye, dyp, test_loader, device, cfg['n_dynamics'],
        wandb_cfg={
            'project': cfg['wandb']['project'], 'group': cfg['wandb']['group'],
            'run_name': f'inception_test/{args.scn}/{args.ds_type}/{args.timestamp}'
        },
        logdir=run_dir,
    )


if __name__ == '__main__':
    main()
