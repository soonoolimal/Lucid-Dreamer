import argparse
import pathlib
import sys
from datetime import datetime

ROOT = pathlib.Path(__file__).parents[2]
SCRIPTS_INC = pathlib.Path(__file__).parent

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS_INC))

import torch
import yaml
from trainer import DTEncTrainer, DTPredTrainer

import inception.models  # register all subclasses
import inception.utils.load as ldu
from inception.models.base import DynEncModel, DynPredModel, ObsEncModel

OBS_ENC_REGISTRY = {cls.__name__: cls for cls in ObsEncModel.__subclasses__()}
DYN_ENC_REGISTRY = {cls.__name__: cls for cls in DynEncModel.__subclasses__()}
DYN_PRED_REGISTRY = {cls.__name__: cls for cls in DynPredModel.__subclasses__()}

CKPT_DIR_TPL = 'logs/inception/{scn}/{ds_type}/{timestamp}'


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--scn', required=True, help='Scenario name (e.g., DeadlyCorridor)')
    parser.add_argument('--ds_type', required=True, choices=['dreamer', 'random'])
    parser.add_argument('--timestamps', nargs='+', default=None,
                        help='Per-dy_type HDF5 timestamps (n_dynamics values). Auto-detect if omitted.')
    parser.add_argument('--device', default=None)
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    with open(ROOT / 'configs/inception.yaml') as f:
        cfg = yaml.safe_load(f)['inception']

    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    run_dir = str(ROOT / CKPT_DIR_TPL.format(scn=args.scn, ds_type=args.ds_type, timestamp=timestamp))
    device = torch.device(args.device or cfg['device'])

    print(f'Device: {device}')
    print(f'Run dir: {run_dir}')

    # Offline Data
    data_cfg = cfg['data']
    train_sets, valid_sets, _ = ldu.load_hdf5_datasets(
        paths=ldu.make_hdf5_paths(args.scn, args.ds_type, cfg['seed'], cfg['n_dynamics'], args.timestamps),
        gamma=data_cfg['gamma'],
        train_ratio=data_cfg['train_ratio'],
        valid_ratio=data_cfg['valid_ratio'],
    )

    max_ep_len = train_sets[0].max_ep_len
    n_act = train_sets[0].n_act
    is_discrete = train_sets[0].is_discrete

    train_loader = ldu.make_dataloader(
        train_sets, data_cfg['seq_len'], data_cfg['batch_size'],
        shuffle=True, drop_last=True, num_workers=data_cfg['num_workers'],
    )
    valid_loader = ldu.make_dataloader(
        valid_sets, data_cfg['seq_len'], data_cfg['batch_size'],
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

    # Shared training configs
    if args.debug:
        cfg['dyn_enc']['train_kw'].update(cfg['debug']['dyn_enc']['train_kw'])
        cfg['dyn_pred']['train_kw'].update(cfg['debug']['dyn_pred']['train_kw'])

    run_name_base = f"{args.scn}/{args.ds_type}/{timestamp}"
    wandb_project = cfg['wandb']['project']
    wandb_group = cfg['wandb']['group']
    if args.debug:
        run_name_base = 'debug_' + run_name_base
        wandb_project = 'Lucid-Dreamer-Debug'
        wandb_group = 'debug_' + wandb_group

    shared_cfg = {
        'n_dynamics': cfg['n_dynamics'],
        'seed': cfg['seed'],
        'ckpt_dir': run_dir,
        'log_dir': run_dir,
        'wandb_project': wandb_project,
        'wandb_group': wandb_group,
    }

    # Train dynamics encoder
    enc_trainer = DTEncTrainer(dye, train_loader, valid_loader, device, {
        **cfg['dyn_enc']['train_kw'],
        **shared_cfg,
        'wandb_run_name': f'inception_enc/{run_name_base}',
    })
    enc_trainer.train()
    enc_trainer.load_best()

    # Freeze dynamics encoder and train dynamics predictor
    dye.requires_grad_(False)

    pred_trainer = DTPredTrainer(dye, dyp, train_loader, valid_loader, device, {
        **cfg['dyn_pred']['train_kw'],
        **shared_cfg,
        'wandb_run_name': f'inception_pred/{run_name_base}',
    })
    pred_trainer.train()


if __name__ == '__main__':
    main()
