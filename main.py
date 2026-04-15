import importlib
import sys

SCRIPTS = {
    'continual_baseline': {
        'train': 'scripts.continual_baseline.train_eval',
        'eval':  'scripts.continual_baseline.eval_only',
    },

}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in SCRIPTS:
        print(f'Usage: python main.py [{"|".join(SCRIPTS)}] [--eval] [...]')
        sys.exit(1)

    task = sys.argv[1]
    argv = sys.argv[2:]
    modes = SCRIPTS[task]

    if '--eval' in argv:
        argv = [a for a in argv if a != '--eval']
        mode = 'eval'
    else:
        mode = next(iter(modes))

    importlib.import_module(modes[mode]).main(argv)


if __name__ == '__main__':
    main()
