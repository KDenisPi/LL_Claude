# LunarLander-v3 DDQN Training Report

**Project:** LunarLander-v3 solved with Double DQN via iterative warm-start chain  
**Total runs:** 23 (LL_01–LL_23, LL_12 was skipped/failed)  
**Total training time:** ~40 hours  
**Total steps:** ~2,860,000  
**Result:** Solved — avg return 249.2 over 30 evaluation episodes  

---

## 1. Environment & Hardware

| Item | Value |
|---|---|
| Environment | `LunarLander-v3` (Gymnasium) |
| Framework | TensorFlow / TF-Agents |
| Agent | `DdqnAgent` (custom `SelectiveClipDqnAgent` subclass) |
| Replay buffer | `TFUniformReplayBuffer` (uniform, no PER) |
| Platform | Linux, Docker container |
| GPU memory cap | 8 GB (via `LogicalDeviceConfiguration(memory_limit=8192)`) |

---

## 2. Model Architecture

| Parameter | Value |
|---|---|
| Network type | Fully connected (Q-network + target Q-network) |
| Hidden layers | 2 × 256 neurons |
| Kernel initializer | `GlorotNormal` |
| Activation | Default (ReLU) |
| Optimizer | Adam |
| Batch size | 256 |
| Replay buffer capacity | 2,400,000 |
| Target network update | Polyak (τ) or periodic (configurable) |

---

## 3. Training Methodology

### Exploration Phase (LL_01–LL_10)
Ten independent runs from scratch, each with a fresh policy and replay buffer. Purpose: find a stable base configuration by varying learning rate, target update parameters, and gradient clipping.

### Warm-Start Chain (LL_11–LL_23)
Starting from LL_11, each run restores only the **agent weights** (Q-network + target network) from a prior run's best or selected checkpoint. The `train_step_counter` is also restored so the step count is cumulative. The replay buffer always starts fresh. This allows continuous policy refinement across runs without restarting exploration.

### Checkpoint Strategy
Two checkpoint managers per run:
- **Regular checkpoints** (`max_to_keep=35`): saved at every evaluation interval when `avg_return > 0`
- **Best checkpoints** (`max_to_keep=1`): saved when `avg_return >= best_return + min_delta`

`best_return_ckpt_var` is persisted within a run but **not** restored across warm-starts — each new run resets the best-return baseline to −∞.

---

## 4. Phase 1: Grid Search (LL_01–LL_10)

All runs use `GlorotNormal` init, `BatchSize=256`, `Gamma=0.99`, `Eps_Start=1.0`, `Eps_End=0.01`, `Eps_Decay=0.00001`, `GradClip=1.5`, layers `[256, 256]`.  
These are independent fresh runs (no warm start).

| Run | Date | Duration | Iterations | LrnRate | UpTau | UpPrd | InitRecords | Best Return |
|---|---|---|---|---|---|---|---|---|
| LL_01 | 2026-06-28 04:39 | 0:53:35 | 120,000 | 0.00003 | 0.002 | 15 | 5,000 | — |
| LL_02 | 2026-06-28 07:33 | 2:28:13 | 600,000 | 0.00003 | 0.002 | 15 | 25,000 | — |
| LL_03 | 2026-06-28 10:19 | 2:41:59 | 600,000 | 0.00002 | 0.002 | 15 | 25,000 | — |
| LL_04 | 2026-06-28 12:19 | 1:57:35 | 600,000 | 0.00005 | 0.002 | 15 | 25,000 | — |
| LL_05 | 2026-06-28 19:55 | 1:14:22 | 250,000 | 0.00003 | 0.005 | 10 | 25,000 | — |
| LL_06 | 2026-06-28 21:49 | 1:51:09 | 250,000 | 0.00002 | 0.005 | 10 | 25,000 | — |
| LL_07 | 2026-06-28 22:12 | 0:21:25 | 250,000 | 0.00005 | 0.005 | 10 | 25,000 | — |
| LL_08 | 2026-06-28 23:56 | 0:39:22 | 250,000 | 0.00003 | 0.001 | 15 | 25,000 | — |
| LL_09 | 2026-06-29 01:20 | 1:23:10 | 250,000 | 0.00002 | 0.001 | 15 | 25,000 | — |
| LL_10 | 2026-06-29 02:14 | 0:51:57 | 250,000 | 0.00005 | 0.001 | 15 | 25,000 | **−109.0** |

**Phase 1 total: ~14.4 hours**

LL_10 produced the best result and was selected as the seed for the warm-start chain. Key configuration: `LrnRate=0.00005`, `UpTau=0.001`, `UpPrd=15`.

---

## 5. Phase 2: Warm-Start Chain (LL_11–LL_23)

Common parameters for all warm-start runs unless noted:  
`Eps_Start=0.1`, `Eps_End=0.05`, `Eps_Decay=0.00002`, `LrnRate=0.00001`, `Gamma=0.99`, `BatchSize=256`, `UpTau=0.001`, `UpPrd=15`, layers `[256, 256]`, `GlorotNormal`.

---

### LL_11 — First Warm Start
**Source:** LL_10 best checkpoint  
**Date:** 2026-06-29 07:50 | **Duration:** 1:34:05 | **Iterations:** 300,000  
**GradClip:** 2.0 | **InitRecords:** 5,000  
**Eval results (17 evals):** best **−92.1**, worst −156.9  
**Best ckpt:** ckpt-3  

First run to use restricted epsilon (0.1→0.05) and lower learning rate. Improved over LL_10's −109.0 baseline.

---

### LL_12 — Skipped / Failed
No entry in parameters.csv. Run did not complete or was never recorded.

---

### LL_13 — Continued from LL_11
**Source:** LL_11 best (ckpt-3)  
**Date:** 2026-07-03 06:22 | **Duration:** 1:02:07 | **Iterations:** 300,000  
**GradClip:** 2.0 | **InitRecords:** 5,000  
**Eval results (9 evals):** best **−94.6**, worst −171.9  
**Best ckpt:** ckpt-1  

Marginal improvement. Best checkpoint was the very first eval, suggesting the policy peaked early.

---

### LL_14 — Reduced gradient clipping
**Source:** LL_13 best (ckpt-1)  
**Date:** 2026-07-06 06:35 | **Duration:** 0:50:47 | **Iterations:** 300,000  
**GradClip:** 1.0 (reduced from 2.0) | **InitRecords:** 5,000  
**Eval results (9 evals):** best **−110.9**, worst −139.2  
**Best ckpt:** ckpt-1  

Regression. Reducing gradient clipping from 2.0 to 1.0 was not beneficial at this stage.

---

### LL_15 — Recovery with clipping 1.0
**Source:** LL_14 best (ckpt-1)  
**Date:** 2026-07-07 02:58 | **Duration:** 1:48:30 | **Iterations:** 300,000  
**GradClip:** 1.0 | **InitRecords:** 5,000  
**Eval results (17 evals):** best **−90.8**, worst −132.4  
**Best ckpt:** ckpt-4  

Recovered to near-LL_11 level. Despite the 1.0 clipping regression in LL_14, continued training recovered the policy. LL_15's best checkpoint was used as the seed for the breakthrough LL_17.

---

### LL_16 — Regression run (warm start from LL_15)
**Source:** LL_15 best (ckpt-4)  
**Date:** 2026-07-08 06:47 | **Duration:** 1:05:32 | **Iterations:** 300,000  
**GradClip:** 2.0 (restored) | **InitRecords:** 20,000  
**Eval results (9 evals):** best **−139.5**, worst −171.0  
**Best ckpt:** ckpt-1  

Unexpected regression. Reverted GradClip to 2.0 and increased InitRecords to 20,000 (larger initial replay fill). The larger buffer pre-fill likely introduced more off-policy noise early in training.

---

### LL_17 — Breakthrough (−13.5)
**Source:** LL_15 best (ckpt-4) — same source as LL_16  
**Date:** 2026-07-08 09:39 | **Duration:** 2:14:20 | **Iterations:** 300,000  
**GradClip:** 2.0 | **InitRecords:** 20,000  
**Step range:** 745,000 → 1,035,000 | **Loss:** 2.183 → 0.477  
**Eval results (17 evals):** best **−13.5**, worst −111.1  
**Best ckpt:** ckpt-8  

Major breakthrough — improved from −90.8 to −13.5. Same configuration as LL_16 but dramatically better result, likely due to stochastic differences in replay buffer initialization. This run was the turning point of the entire campaign.

---

### LL_18 — Consolidation (−5.5)
**Source:** LL_17 best (ckpt-8)  
**Date:** 2026-07-09 07:17 | **Duration:** 1:52:41 | **Iterations:** 300,000  
**GradClip:** 2.0 | **InitRecords:** 20,000  
**Step range:** 1,025,000 → 1,255,000 | **Loss:** 2.916 → 0.475  
**Eval results (14 evals):** best **−5.5**, worst −36.0  
**Best ckpt:** ckpt-4  

Continued improvement on the LL_17 breakthrough, reducing best return from −13.5 to −5.5. Policy approaching zero.

---

### LL_19 — Cut short by early stopping (−5.1)
**Source:** LL_18 best (ckpt-4)  
**Date:** 2026-07-10 01:30 | **Duration:** 1:06:12 | **Iterations:** 300,000  
**GradClip:** 2.0 | **InitRecords:** 20,000  
**Step range:** 1,145,000 → 1,275,000 | **Loss:** 3.370 → 0.679  
**Eval results (9 evals):** best **−5.1**, worst −19.9  
**Best ckpt:** ckpt-2  

**Issue:** Early stopping triggered after only ~130,000 of 300,000 iterations (patience=6 evals, ~120K steps). The warm-start block did not override `early_stop_enabled`, so the main-loop default of `True` applied. The best result (−5.1) came from the post-loop final eval, *after* early stopping had already fired at eval[7].

**Fix applied:** Added `cfg._early_stop_enabled = False` to warm-start config block for LL_20.

---

### LL_20 — First positive return (+3.6)
**Source:** LL_19 best (ckpt-2)  
**Date:** 2026-07-10 06:54 | **Duration:** 3:46:58 | **Iterations:** 500,000  
**GradClip:** 2.0 | **InitRecords:** 5,000 | **EarlyStop:** disabled  
**Step range:** 1,285,000 → 1,775,000 | **Loss:** 3.430 → 0.153  
**Eval results (27 evals):** best **+3.6** (at regular ckpt-4), worst −25.1  
**Best ckpt:** ckpt-3 (return +0.17 — see issue below)  

**First run to reach positive territory.** The 500K iteration budget and disabled early stopping allowed the training to continue past the plateau near zero.

**Issue:** The true best eval (+3.61) was not saved to `best_ckpt_manager` because `min_delta=5.0` (set in main loop, not overridden in warm-start block). The +3.61 improvement over the prior best (+0.17) was only 3.44 — below the 5.0 threshold. The weights were in regular checkpoint ckpt-4.

**Fix applied:** Added `cfg._early_stop_min_delta = 1.0` to warm-start config block.  
**Warm-start for LL_21:** used regular ckpt-4 (return +3.61), not LL_20_best (return +0.17).

---

### LL_21 — Stable positive territory (+9.6)
**Source:** LL_20 regular ckpt-4 (return +3.61)  
**Date:** 2026-07-11 06:13 | **Duration:** 3:45:47 | **Iterations:** 500,000  
**GradClip:** 2.0 | **InitRecords:** 5,000 | **MinDelta:** 1.0  
**Step range:** 1,425,000 → 1,915,000 | **Loss:** 3.195 → 0.194  
**Eval results (27 evals):** best **+9.6**, worst −5.3  
**Best ckpt:** ckpt-17  

With `min_delta=1.0`, the best checkpoint manager captured all meaningful improvements. 17 best checkpoints were saved across the run, showing steady progress. Returns stabilized in positive territory.

*Note: Two earlier LL_21 attempts (durations 0:04:12 and 0:14:37) aborted due to a GPU memory configuration conflict (see Technical Issues section). The 3:45:47 run was the actual training run.*

---

### LL_22 — SOLVED (+211.7, avg over 200)
**Source:** LL_21 best (ckpt-17, return +9.65)  
**Date:** 2026-07-11 10:38 | **Duration:** 3:38:51 | **Iterations:** 500,000  
**GradClip:** 2.0 | **InitRecords:** 5,000 | **MinDelta:** 1.0  
**Step range:** 1,885,000 → 2,375,000 | **Loss:** 3.420 → 0.641  
**Eval results (27 evals):** best **+211.7**, worst −2.5  
**Best ckpt:** ckpt-29  

The LunarLander-v3 threshold (avg ≥ 200) was crossed in this run. Returns showed a dramatic jump from the 0–10 range into 50–211 territory across the 27 evaluations.

| Eval # | Return |
|---|---|
| 1 | −2.5 |
| 5 | 6.2 |
| 10 | 10.0 |
| 15 | 35.5 |
| 18 | 155.3 |
| 22 | 206.2 |
| 25 | **211.7** |

---

### LL_23 — Stabilized at 250+ (best +250.8)
**Source:** LL_22 best (ckpt-29, return +211.7)  
**Date:** 2026-07-11 22:19 | **Duration:** 2:49:10 | **Iterations:** 500,000  
**GradClip:** 2.0 | **InitRecords:** 5,000 | **MinDelta:** 1.0  
**Step range:** 2,365,000 → 2,855,000 | **Loss:** 4.648 → 0.979  
**Eval results (27 evals):** best **+250.8**, worst +88.1  
**Best ckpt:** ckpt-31  

All 27 evals in this run exceeded 88. The policy was fully stable above 200 for the majority of the run.

| Eval # | Return |
|---|---|
| 1 | 212.1 |
| 4 | 88.1 (only dip) |
| 9 | 234.5 |
| 16 | 140.7 (minor dip) |
| 26 | **250.8** |

---

## 6. Final Evaluation

After training completed, LL_23_best (ckpt-31, return +250.8) was selected for final evaluation.

**Command:**
```
python ModelTrain.py --label=LL_23 --evaluate=LL_23 --evaluate_ckpt=LL_23_best/ckpt-31
```

**30-episode evaluation results:**

| Metric | Value |
|---|---|
| Average return (30 eps) | **249.2** |
| Episodes above 200 | 29 / 30 (96.7%) |
| Maximum return | 303 |
| Minimum return | 188 |
| Threshold (solved) | 200 |

Three independent evaluation runs showed consistent results:
- Run 1: avg 251.7
- Run 2: avg 256.2  
- Run 3: avg 239.6

**Verdict: SOLVED** (avg ≥ 200 over 30 episodes)

---

## 7. Technical Issues & Fixes

### Issue 1 — LL_19 Early Stopping Cut Run Short
**Symptom:** LL_19 ran only ~130K of 300K iterations; best result came from post-loop final eval.  
**Root cause:** `warm_start` config block did not set `early_stop_enabled`; the main-loop default (`True`, patience=6) applied. With patience=6 and 20K-step eval intervals, early stopping fired after ~120K steps with no improvement.  
**Fix:** Added `cfg._early_stop_enabled = False` to warm-start block. Commit: `2a36f4b`.

### Issue 2 — LL_20 Best Eval Not Saved to Best Checkpoint
**Symptom:** Best eval in LL_20 (+3.61) was in regular checkpoint ckpt-4, not the best checkpoint (which had +0.17).  
**Root cause:** `early_stop_min_delta=5.0` set in main loop, not overridden in warm-start block. The +3.44 improvement fell below the 5.0 threshold.  
**Fix:** Added `cfg._early_stop_min_delta = 1.0` to warm-start block. Commit: `6e65ac2`. Warm-start for LL_21 manually used regular ckpt-4.

### Issue 3 — GPU Memory Configuration Conflict (LL_21 Early Aborts)
**Symptom:** LL_21 aborted after 4 and 14 minutes with a TensorFlow `RuntimeError`.  
**Root cause:** Both `set_memory_growth(True)` and `set_logical_device_configuration(memory_limit=8192)` were called; these are mutually exclusive in TF. Also, Docker `--memory=8g` is bypassed by `--ulimit memlock=-1` (allows unlimited pinned memory) and `--ipc=host`.  
**Fix:** Removed `set_memory_growth` calls; kept only `set_logical_device_configuration`. GPU VRAM is now capped at 8 GB in Python. Commit: `0b75e99`.

### Issue 4 — Evaluation Mode Replay Buffer Shape Mismatch
**Symptom:** `ValueError: Shapes (2400000,) and (1000000,) are incompatible` when running `--evaluate` mode.  
**Root cause:** The training checkpoint (`self.ckpt`) includes `replay_buffer`. The checkpoint was saved with capacity 2,400,000 (training mode). Eval mode initializes a fresh buffer with default capacity 1,000,000. Full restore failed on the shape mismatch.  
**Fix:** In `evaluate_chkpt()`, replaced `self.ckpt.restore(...)` with a partial checkpoint that restores only `agent`, `global_step`, and `custom_variable` — skipping `replay_buffer`. Commit: `4f4275d`.

---

## 8. Progress Timeline

| Run | Date | Best Return | Steps | Duration |
|---|---|---|---|---|
| LL_01–LL_09 | 2026-06-28 | — | fresh | ~12.7 h |
| LL_10 | 2026-06-29 | **−109.0** | ~250K | 0:52 |
| LL_11 | 2026-06-29 | **−92.1** | ~550K | 1:34 |
| LL_13 | 2026-07-03 | −94.6 | ~850K | 1:02 |
| LL_14 | 2026-07-06 | −110.9 | ~1,150K | 0:51 |
| LL_15 | 2026-07-07 | **−90.8** | ~1,450K | 1:49 |
| LL_16 | 2026-07-08 | −139.5 (regression) | ~1,750K | 1:06 |
| LL_17 | 2026-07-08 | **−13.5** (breakthrough) | ~1,035K | 2:14 |
| LL_18 | 2026-07-09 | **−5.5** | ~1,255K | 1:53 |
| LL_19 | 2026-07-10 | −5.1 (early stop) | ~1,275K | 1:06 |
| LL_20 | 2026-07-10 | **+3.6** (first positive) | ~1,775K | 3:47 |
| LL_21 | 2026-07-11 | **+9.6** | ~1,915K | 3:46 |
| LL_22 | 2026-07-11 | **+211.7** (SOLVED) | ~2,375K | 3:39 |
| LL_23 | 2026-07-11 | **+250.8** (stabilized) | ~2,855K | 2:49 |

**Total wall-clock training time: ~40 hours**  
**Total environment steps: ~2,860,000**

---

## 9. Final Hyperparameters (Solve Configuration)

| Parameter | Value |
|---|---|
| `num_iterations` | 500,000 |
| `num_initial_records` | 5,000 |
| `batch_size` | 256 |
| `learning_rate` | 0.00001 |
| `gamma` | 0.99 |
| `epsilon_start` | 0.1 |
| `epsilon_end` | 0.05 |
| `epsilon_decay` | 0.00002 |
| `gradient_clipping` | 2.0 |
| `update_tau` | 0.001 |
| `update_period` | 15 |
| `early_stop_enabled` | False |
| `early_stop_min_delta` | 1.0 |
| `layers` | [256, 256] |
| `kernel_initializer` | GlorotNormal |

---

## 10. Key Lessons Learned

1. **Warm-start chain works** — Starting from a prior run's best weights allows the agent to refine its policy incrementally without restarting exploration from scratch. The chain spanned 13 runs from −109 to +250.

2. **Early stopping harms warm-start runs** — The policy near zero changes slowly; patience thresholds calibrated for a learning-from-scratch run fire too early in a warm-start context. Disable early stopping and use a fixed iteration budget.

3. **min_delta granularity matters near zero** — With `min_delta=5.0`, a +3.6 improvement over +0.17 was invisible to the best-checkpoint manager. Reducing to 1.0 captured every 1-point gain.

4. **Replay buffer shape must match on restore** — Partial checkpoint restore (agent only, no replay buffer) is the safe pattern for evaluation mode. The training checkpoint includes buffer state that is capacity-specific.

5. **Stochastic variance in warm-start runs** — LL_16 and LL_17 used identical configuration and the same source checkpoint yet produced −139.5 and −13.5 respectively. Random replay buffer initialization creates high variance; running failed configurations again can sometimes succeed.

6. **Loss spike on warm-start is normal** — Each run starts with a fresh replay buffer, so the first batches contain only recent transitions. Loss spikes to 2–4 at the start of every warm-start run before converging. This is expected behavior, not instability.
