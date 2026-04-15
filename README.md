# Lucid Dreamer

## Setup

```bash
python -m embodied.envs.vizdoom.build_scenarios
```

## 0. Baseline

Baseline performance of DreamerV3 under non-stationary VizDoom dynamics.

### 0.1. Training & Evaluation

```bash
python main.py continual_baseline --task vizdoom_DeadlyCorridor
python main.py continual_baseline --task vizdoom_DeadlyCorridor --seed 1
```

For debug: `python main.py continual_baseline --configs defaults vizdoom_continual debug --task vizdoom_DeadlyCorridor`

To run multiple seeds in parallel across GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py continual_baseline --task vizdoom_DeadlyCorridor --seed 0 &
CUDA_VISIBLE_DEVICES=1 python main.py continual_baseline --task vizdoom_DeadlyCorridor --seed 1 &
CUDA_VISIBLE_DEVICES=2 python main.py continual_baseline --task vizdoom_DeadlyCorridor --seed 2 &
```
### 0.2. Evaluation Only

Requires a saved checkpoint from a training run.

```bash
python main.py continual_baseline --eval \
    --task vizdoom_DeadlyCorridor \
    --run.from_checkpoint logs/continual_baseline/DeadlyCorridor/{timestamp}/ckpt
```
