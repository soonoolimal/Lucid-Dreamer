DeadlyCorridor dynamics:
- dy0: obs_shift=null, rew_shift=null            (distance + kill_base=25)
- dy1: obs_shift=null, rew_shift=survive         (distance + kill_base=25 + survive)
- dy2: obs_shift=BIGBRIK1-to-TEKWALL4, rew_shift=null    (distance + kill_base=25)
- dy3: obs_shift=BIGBRIK1-to-TEKWALL4, rew_shift=survive (distance + kill_base=25 + survive)

HP adjustments (vizdoom.yaml &vizdoom_agent anchor, deadly-corridor branch only adds imag_length=30):
- agent.imag_loss.actent: 3e-3
- agent.policy.unimix: 0.03
- agent.loss_scales.rew: 2.0
- agent.loss_scales.con: 2.0
- agent.dyn.rssm.unimix: 0.03
- agent.imag_length: 30

---

## 0. Continual Baseline

```bash
python main.py continual_baseline --task vizdoom_DeadlyCorridor --logger.wandb_project Lucid-Dreamer-DeadlyCorridor
```

Resume:
```bash
python main.py continual_baseline --task vizdoom_DeadlyCorridor --logger.wandb_project Lucid-Dreamer-DeadlyCorridor \
    --run.resume_timestamp {timestamp} \
    --run.wandb_id {wandb_run_id}
```

Debug:
```bash
python main.py continual_baseline --configs defaults vizdoom_continual debug --task vizdoom_DeadlyCorridor --logger.wandb_project Lucid-Dreamer-DeadlyCorridor
```

---

## 1. Per-Dynamics Dreamer

### 1-1. Training (dy0~3)

```bash
python main.py per_dy_dreamer --task vizdoom_DeadlyCorridor --logger.wandb_project Lucid-Dreamer-DeadlyCorridor --env.vizdoom.dy_type 0
python main.py per_dy_dreamer --task vizdoom_DeadlyCorridor --logger.wandb_project Lucid-Dreamer-DeadlyCorridor --env.vizdoom.dy_type 1
python main.py per_dy_dreamer --task vizdoom_DeadlyCorridor --logger.wandb_project Lucid-Dreamer-DeadlyCorridor --env.vizdoom.dy_type 2
python main.py per_dy_dreamer --task vizdoom_DeadlyCorridor --logger.wandb_project Lucid-Dreamer-DeadlyCorridor --env.vizdoom.dy_type 3
```

Resume:
```bash
python main.py per_dy_dreamer --task vizdoom_DeadlyCorridor --logger.wandb_project Lucid-Dreamer-DeadlyCorridor --env.vizdoom.dy_type 0 \
    --run.resume_timestamp {timestamp} \
    --run.wandb_id {wandb_run_id}
```

Debug:
```bash
python main.py per_dy_dreamer --configs defaults vizdoom_fixed debug --task vizdoom_DeadlyCorridor --logger.wandb_project Lucid-Dreamer-DeadlyCorridor --env.vizdoom.dy_type 0
```

### 1-2. Random Agent Sampling

```bash
python main.py per_dy_dreamer --task vizdoom_DeadlyCorridor --logger.wandb_project Lucid-Dreamer-DeadlyCorridor --env.vizdoom.dy_type 0 --random_agent True && \
python main.py per_dy_dreamer --task vizdoom_DeadlyCorridor --logger.wandb_project Lucid-Dreamer-DeadlyCorridor --env.vizdoom.dy_type 1 --random_agent True && \
python main.py per_dy_dreamer --task vizdoom_DeadlyCorridor --logger.wandb_project Lucid-Dreamer-DeadlyCorridor --env.vizdoom.dy_type 2 --random_agent True && \
python main.py per_dy_dreamer --task vizdoom_DeadlyCorridor --logger.wandb_project Lucid-Dreamer-DeadlyCorridor --env.vizdoom.dy_type 3 --random_agent True
```

### 1-3. Sample Only

Dreamer checkpoint:
```bash
python main.py per_dy_dreamer --sample \
    --task vizdoom_DeadlyCorridor \
    --logger.wandb_project Lucid-Dreamer-DeadlyCorridor \
    --env.vizdoom.dy_type 0 \
    --timestamp {timestamp} \
    --run.n_sample_eps 1000
```

Random agent checkpoint:
```bash
python main.py per_dy_dreamer --sample \
    --task vizdoom_DeadlyCorridor \
    --logger.wandb_project Lucid-Dreamer-DeadlyCorridor \
    --env.vizdoom.dy_type 0 \
    --random_agent True \
    --timestamp {timestamp} \
    --run.n_sample_eps 1000
```

---

## 2. Inception Pretraining

```bash
python main.py pretrain_inception --scn DeadlyCorridor --ds_type dreamer --wandb_project Lucid-Dreamer-DeadlyCorridor
python main.py pretrain_inception --scn DeadlyCorridor --ds_type random --wandb_project Lucid-Dreamer-DeadlyCorridor
```

Debug:
```bash
python main.py pretrain_inception --scn DeadlyCorridor --ds_type random --debug
```

Test only:
```bash
python main.py pretrain_inception --test \
    --scn DeadlyCorridor --ds_type dreamer \
    --timestamp {timestamp}
```

---

## 3. Lucid Dreamer

```bash
python main.py lucid_dreamer \
    --task vizdoom_DeadlyCorridor \
    --logger.wandb_project Lucid-Dreamer-DeadlyCorridor \
    --ds_type dreamer \
    --inc_timestamp {timestamp}
```

Resume:
```bash
python main.py lucid_dreamer \
    --task vizdoom_DeadlyCorridor \
    --logger.wandb_project Lucid-Dreamer-DeadlyCorridor \
    --ds_type dreamer \
    --inc_timestamp {timestamp} \
    --run.resume_timestamp {timestamp} \
    --run.wandb_id {wandb_run_id}
```

Debug:
```bash
python main.py lucid_dreamer \
    --configs defaults vizdoom_lucid lucid_dreamer debug \
    --task vizdoom_DeadlyCorridor \
    --logger.wandb_project Lucid-Dreamer-DeadlyCorridor \
    --ds_type dreamer \
    --inc_timestamp {timestamp}
```
