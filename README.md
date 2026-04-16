# Lucid Dreamer

## Setup

Requires CUDA 12.8 (cu128).

```bash
conda env create -f conda_env.yaml
conda activate lucid-dreamer
```

```
python -m embodied.envs.vizdoom.build_scenarios
```

## 0. Baseline

Baseline performance of DreamerV3 under non-stationary VizDoom dynamics.

### 0.1. Training & Evaluation

```bash
python main.py continual_baseline --task vizdoom_DeadlyCorridor
python main.py continual_baseline --task vizdoom_DeadlyCorridor --seed 1
```

For debug:
```
python main.py continual_baseline --configs defaults vizdoom_continual debug --task vizdoom_DeadlyCorridor
```

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

---

## 1. Per-Dynamics Dreamer

Trains DreamerV3 on a fixed dynamics type and periodically collects offline HDF5 data for Alarm training.
HDF5 samples are saved to `logs/per_dy_dreamer/{scn}/dy{dy_type}/{timestamp}/samples/type-{dy_type}_s{seed}.hdf5`.

### 1.1. Training

```bash
python main.py per_dy_dreamer --task vizdoom_DeadlyCorridor --env.vizdoom.dy_type 0
python main.py per_dy_dreamer --task vizdoom_DeadlyCorridor --env.vizdoom.dy_type 1
python main.py per_dy_dreamer --task vizdoom_DeadlyCorridor --env.vizdoom.dy_type 2
python main.py per_dy_dreamer --task vizdoom_DeadlyCorridor --env.vizdoom.dy_type 3
```

For debug:

```bash
python main.py per_dy_dreamer --configs defaults vizdoom_fixed debug --task vizdoom_DeadlyCorridor --env.vizdoom.dy_type 0
```

To run all dy_types in parallel across GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py per_dy_dreamer --task vizdoom_DeadlyCorridor --env.vizdoom.dy_type 0 &
CUDA_VISIBLE_DEVICES=1 python main.py per_dy_dreamer --task vizdoom_DeadlyCorridor --env.vizdoom.dy_type 1 &
CUDA_VISIBLE_DEVICES=2 python main.py per_dy_dreamer --task vizdoom_DeadlyCorridor --env.vizdoom.dy_type 2 &
CUDA_VISIBLE_DEVICES=3 python main.py per_dy_dreamer --task vizdoom_DeadlyCorridor --env.vizdoom.dy_type 3 &
```

### 1.2. Random Agent Sampling

Collects offline data with a random agent (no training). Cheat mode is enabled automatically.

```bash
python main.py per_dy_dreamer --task vizdoom_DeadlyCorridor --env.vizdoom.dy_type 0 --random_agent True
```

### 1.3. Sample Only

Loads a saved checkpoint and collects HDF5 samples without training.

```bash
python main.py per_dy_dreamer --sample \
    --task vizdoom_DeadlyCorridor \
    --env.vizdoom.dy_type 0 \
    --run.from_checkpoint logs/per_dy_dreamer/DeadlyCorridor/dy0/{timestamp}/ckpt \
    --run.n_sample_eps 50
```
