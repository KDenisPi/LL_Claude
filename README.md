# LunarLander-v3 — Double DQN

A from-scratch Double DQN (DDQN) agent for Gymnasium's `LunarLander-v3`, built on TF-Agents/TensorFlow. The project **solved** the environment (avg return ≥ 200 over 30 episodes) using an iterative *warm-start chain*: 23 runs, ~40 hours of training, ~2.86M environment steps, final avg return **249.2**.

See [`LL_result.md`](LL_result.md) for the full run-by-run training report and [`LL_QStd.md`](LL_QStd.md) for a walkthrough of how the Q-value-spread (QStd) signal in TensorBoard evolved from an untrained network to a converged policy.

## How it works

- **Agent:** `SelectiveClipDqnAgent`, a `tf_agents.agents.dqn.dqn_agent.DdqnAgent` subclass that supports per-layer gradient-norm clipping (`ModelTrain.py`).
- **Network:** fully connected, 2 hidden layers of 256 units, `GlorotNormal` init.
- **Environment:** `LunarLander-v3` (Gymnasium) wrapped as a TF-Agents `PyEnvironment` via `GymnasiumWrapper` (`gym_wrap.py`), since TF-Agents' bundled `suite_gym` targets the older `gym` API.
- **Replay buffer:** `TFUniformReplayBuffer`, or a prioritized (PER) sampler — configurable.
- **Warm-start chain:** instead of always training from scratch, later runs restore only the agent's Q-network/target-network weights (and step counter) from a prior run's checkpoint, then continue training with a fresh replay buffer. This let the policy improve incrementally across dozens of sessions instead of restarting exploration each time.
- **Config:** all hyperparameters live in `ModelCfg.py` (`ModelCfg` class) and are overridden per-run directly in `ModelTrain.py`'s `__main__` block — this is a research sandbox, not a CLI tool with stable flags.

## Repository layout

| Path | Purpose |
|---|---|
| `ModelTrain.py` | Main training/evaluation driver: agent, training loop, checkpointing, warm-start logic, `__main__` entry point |
| `ModelTrainMin.py` | Minimal/simplified variant of the training loop |
| `ModelCfg.py` | `ModelCfg` — all hyperparameters (learning rate, epsilon schedule, network shape, PER params, checkpoint/eval intervals, early stopping, etc.) |
| `ModelUtils.py` | Small shared helpers |
| `gym_wrap.py` | `GymnasiumWrapper` — adapts a Gymnasium env to the TF-Agents `PyEnvironment` interface |
| `generate_video.py` | Loads a saved checkpoint and records an mp4 of the greedy policy via `RecordVideo` |
| `plot_runs.py` | Compares eval returns / loss / gradient norms across runs from the CSVs `ModelTrain.py` writes |
| `plot_lr_schedule.py` | Plots the warmup + cosine-decay learning-rate schedule for chosen parameter combos |
| `test_env.py` | Smoke test for the Gymnasium → TF-Agents environment wrapper |
| `docker/` | `DockerfileGPU1` (based on `nvcr.io/nvidia/tensorflow:24.01-tf2-py3`) + pinned pip requirements |
| `data/` | Per-run outputs: checkpoints (`multi_checkpoint_<label>[_best]/`), eval returns, loss/gradient/Q-value/learning-rate CSVs, `parameters.csv` |
| `logs/` | Per-run TensorBoard event logs |
| `images/`, `tb_images/` | Exported plots (learning-rate schedules, gradient norms, TensorBoard QStd screenshots) |
| `docs/` | Analysis notes (`Diff.txt`: comparison against a reference DQN implementation), a sample rendered episode video, and Docker/TensorFlow setup notes |
| `LL_result.md` | Full training report, run-by-run (LL_01–LL_23) |
| `LL_QStd.md` | Analysis of the QStd (Q-value spread) convergence signal across the campaign |

## Setup

Build and run inside the provided Docker image (GPU, TensorFlow 24.01, pybox2d built from source for Box2D support):

```bash
docker build -t lunarlander-ddqn -f docker/DockerfileGPU1 docker/
```

Run training, mounting the repo and data/log directories:

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $(pwd):/app/src -v $(pwd)/data:/app/data:rw -v $(pwd)/logs:/app/logs:rw \
  -u 1000:1000 -it -w /app/src --rm lunarlander-ddqn \
  python ModelTrain.py --label=LL_24
```

Watch TensorBoard:

```bash
docker run --gpus all -v $(pwd)/logs:/app/logs:rw -p 6006:6006 -u 1000:1000 -it --rm \
  lunarlander-ddqn tensorboard --logdir=/app/logs
```

## Usage

Training runs are configured by editing the hyperparameter block at the bottom of `ModelTrain.py` (learning rate, epsilon schedule, iterations, clipping, etc.) before each run, then launched with a run label:

```bash
python ModelTrain.py --label=LL_24
```

Warm-start from a prior run's checkpoint:

```bash
python ModelTrain.py --label=LL_24 --warm_start=LL_23_best --warm_start_ckpt=31
```

Evaluate a saved checkpoint:

```bash
python ModelTrain.py --label=LL_23 --step=LL_23_best/ckpt-31
```

Record a video of the greedy policy from a checkpoint:

```bash
python generate_video.py --label LL_23_best --ckpt 31 --episodes 3
```

Compare eval returns / loss / gradients across runs:

```bash
python plot_runs.py LL_21 LL_22 LL_23
```

## Result

| Metric | Value |
|---|---|
| Average return (30 eval episodes) | **249.2** |
| Episodes above 200 | 29 / 30 |
| Solve threshold | 200 |

Full progression, hyperparameters, and lessons learned from the 23-run campaign are documented in [`LL_result.md`](LL_result.md).
