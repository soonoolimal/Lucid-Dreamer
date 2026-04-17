import importlib
import sys

from scripts.common.banner import print_banner

SCRIPTS = {
    'continual_baseline': {
        'train': 'scripts.continual_baseline.train_eval',
        'eval':  'scripts.continual_baseline.eval_only',
    },
    'per_dy_dreamer': {
        'train':  'scripts.per_dy_dreamer.train',
        'sample': 'scripts.per_dy_dreamer.sample_only',
    },
    'pretrain_inception': {
        'train': 'scripts.pretrain_inception.train_eval',
        'test':  'scripts.pretrain_inception.test',
    },
    'lucid_dreamer': {
        'train': 'scripts.lucid_dreamer.train_eval',
    },
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in SCRIPTS:
        print(f'Usage: python main.py [{"|".join(SCRIPTS)}] [--eval|--sample] [...]')
        sys.exit(1)

    task = sys.argv[1]
    argv = sys.argv[2:]
    modes = SCRIPTS[task]

    if '--eval' in argv:
        argv = [a for a in argv if a != '--eval']
        mode = 'eval'
    elif '--sample' in argv:
        argv = [a for a in argv if a != '--sample']
        mode = 'sample'
    elif '--test' in argv:
        argv = [a for a in argv if a != '--test']
        mode = 'test'
    else:
        mode = next(iter(modes))

    if mode not in modes:
        valid = ', '.join(f'--{m}' for m in modes if m != next(iter(modes)))
        print(f"Error: '{task}' does not support --{mode}. Valid flags: {valid or 'none'}")
        sys.exit(1)

    banner_key = task
    if task == 'pretrain_inception':
        banner_key = 'inception'
    elif task == 'per_dy_dreamer' and '--random_agent' in argv:
        idx = argv.index('--random_agent')
        if idx + 1 < len(argv) and argv[idx + 1].lower() == 'true':
            banner_key = 'random_agent'
    print_banner(banner_key)
    importlib.import_module(modes[mode]).main(argv)


if __name__ == '__main__':
    main()
