# Experiment Tracker

**Last Updated:** 2025-01-13

## Overview: Planned Results

- **[Point A]** FIATS outperforms unimodal on fidel-ts test sets (except CAISO). Show efficiency and performance on big datasets, compare to other multimodal methods, show loss curves.
- **[Point B]** Beat unimodal consistently on time mmd and ttc (where multimodal typically can't). Show we do better than time mmd method on big datasets.
- **[Point C]** Show that learning a residual can be damaging; learning the full signal is better.
- **[Point D]** Show prediction level ensembles are useful, and present other methods for ensuring text is useful but doesn't pollute time series signal when alternative modality isn't useful.

---

## High Priority Experiments

### Priority 1: iTransformer Pretraining (Baselines)

**Status:** 🟡 In Progress  
**Purpose:** Unimodal baselines needed for all datasets before multimodal comparisons

#### Fidel-TS Datasets
| Dataset | Status | Suite ID | Experiment ID | Location | Compute |
|---------|--------|----------|---------------|----------|---------|
| Canada Photovoltaics | ✅ Complete | `itransformer_canada_photovoltaics_20260113_101629` | `20260113-101629_d36af4d594fb` | `output/itransformer_canada_photovoltaics_20260113_101629/` | RunPod (pod:canada) |
| Germany Renewable | 🟡 Running | - | - | - | RunPod (pod:germany) |
| NYC Traffic Speed | ⏳ Pending | - | - | - | RunPod (pod:nyc_traffic) |
| California ISO | ⏳ Pending | - | - | - | RunPod (pod:caiso) |
| Bear Room | ⏳ Pending | - | - | - | RunPod (pod:bear_room) |
| Jena Atmospheric Physics | ⏳ Pending | - | - | - | RunPod (pod:jena) |

#### Time MMD & TTC
| Dataset | Status | Suite ID | Experiment ID | Location | Compute |
|---------|--------|----------|---------------|----------|---------|
| Time MMD & TTC | ✅ Complete | - | - | - | MIT (engaging) |
| Evaluation | ⏳ Pending | - | - | - | - |

---

### Priority 2: TGTSF (FIATS) Loss Curves on Fidel-TS

**Status:** ⏳ Pending  
**Purpose:** Generate loss curves for comparison with other multimodal methods

| Dataset | Status | Suite ID | Experiment ID | Location | Compute |
|---------|--------|----------|---------------|----------|---------|
| All Fidel-TS | ⏳ Pending | - | - | - | RunPod (A40 x 6) |

---

### Priority 3: Benchmark Methods on Fidel-TS

**Status:** 🟡 In Progress  
**Purpose:** Compare against other multimodal methods

#### Time-LLM
| Dataset | Status | Suite ID | Experiment ID | Location | Compute |
|---------|--------|----------|---------------|----------|---------|
| California ISO | ✅ Complete | `timellm_california_iso_20260112_141144` | `20260112-141144_a05ebe165526` | `output/timellm_california_iso_20260112_141144/` | MIT Sloan (A100) |
| NYC Traffic Speed | 🟡 Running | `timellm_nyc_traffic_speed_20260112_185045` | - | - | MIT Sloan (A100) |
| Canada Photovoltaics | ⏳ Pending | - | - | - | MIT Sloan (A100) |
| Germany Renewable | ⏳ Pending | - | - | - | MIT Sloan (A100) |
| Bear Room | ⏳ Pending | - | - | - | MIT Sloan (A100) |
| Jena Atmospheric Physics | ⏳ Pending | - | - | - | MIT Sloan (A100) |

#### TimeCMA
| Dataset | Status | Suite ID | Experiment ID | Location | Compute |
|---------|--------|----------|---------------|----------|---------|
| Time MMD & TTC | ✅ Complete | - | - | - | MIT Sloan (A100) |
| Bear Room | ✅ Complete | - | - | - | MIT Sloan (A100) |
| California ISO | 🟡 Running (OOM issues) | - | - | - | MIT Sloan (A100) |
| Other Fidel-TS | ⏳ Pending | - | - | - | MIT Sloan (A100) |

#### LeRet
| Dataset | Status | Suite ID | Experiment ID | Location | Compute |
|---------|--------|----------|---------------|----------|---------|
| All Fidel-TS | ⏳ Pending | - | - | - | RunPod (RTX 4000 Ada x 5) |

#### Other Benchmarks (Pending Experimentation)
- **ZhangHanBest-PatchTST**: ⏳ Pending (needs experimentation setup)
- **ZhangHanBest-DLinear**: ⏳ Pending (needs experimentation setup)
- **MMTSFLib-FedFormer**: ⏳ Pending (needs experimentation setup)

---

### Priority 4: Lynx Film Raw on Fidel-TS

**Status:** 🟡 In Progress  
**Purpose:** Initial runs before hyperparameter tuning complete

#### Hyperparameter Sweep (NYC Traffic)
| Status | Suite ID | Experiment ID | Location | Compute |
|--------|----------|---------------|----------|---------|
| 🟡 Running | `sweep_v2dy5lr5` | - | - | MIT Preemptable (L40s x 4) |

#### Initial Runs (Other Datasets)
| Dataset | Status | Suite ID | Experiment ID | Location | Compute |
|---------|--------|----------|---------------|----------|---------|
| NYC Traffic Speed | ✅ Complete (sweep) | `sweep_v2dy5lr5` | - | - | MIT Preemptable |
| Canada Photovoltaics | ⏳ Pending | - | - | - | RunPod (L40s x 5) |
| Germany Renewable | ⏳ Pending | - | - | - | RunPod (L40s x 5) |
| California ISO | ⏳ Pending | - | - | - | RunPod (L40s x 5) |
| Bear Room | ⏳ Pending | - | - | - | RunPod (L40s x 5) |
| Jena Atmospheric Physics | ⏳ Pending | - | - | - | RunPod (L40s x 5) |

---

### Priority 5: Lynx Film on Fidel-TS

**Status:** 🟡 In Progress  
**Purpose:** Compare with residual learning (Point C)

| Dataset | Status | Suite ID | Experiment ID | Location | Compute |
|---------|--------|----------|---------------|----------|---------|
| Canada Photovoltaics | ✅ Complete | - | - | - | RunPod |
| Germany Renewable | ⏳ Pending (testing tensor_cache) | - | - | - | RunPod |
| NYC Traffic Speed | ⏳ Pending | - | - | - | RunPod |
| California ISO | ⏳ Pending | - | - | - | RunPod |
| Bear Room | ⏳ Pending | - | - | - | RunPod |
| Jena Atmospheric Physics | ⏳ Pending | - | - | - | RunPod |

**Note:** Requires iTransformer pretraining to complete first.

---

### Priority 6: Lynx Film Enhanced (Select Configurations)

**Status:** ⏳ Pending  
**Purpose:** Ablations on various ways to mix with gating (Point D)

**Prerequisites:**
- ✅ Testing complete
- ✅ Low compute ablations complete
- ✅ iTransformer pretraining complete

| Configuration | Status | Suite ID | Experiment ID | Location | Compute |
|---------------|--------|----------|---------------|----------|---------|
| TBD | ⏳ Pending | - | - | - | TBD (x6 x nconfigurations) |

---

### Priority 7: MMVision Best Configuration

**Status:** ⏳ Pending  
**Purpose:** Final best configuration selection

**Prerequisites:**
- ✅ Lynx Film Enhanced configurations tested
- ✅ Best configuration identified

| Status | Suite ID | Experiment ID | Location | Compute |
|--------|----------|---------------|----------|---------|
| ⏳ Pending | - | - | - | TBD (x1) |

---

## Low Priority Experiments

### Quick Runs on Time MMD & TTC

**Status:** 🟡 In Progress  
**Purpose:** Validate multimodal performance on smaller datasets (Point B)

#### iTransformer Pretraining
| Dataset | Status | Suite ID | Experiment ID | Location | Compute |
|---------|--------|----------|---------------|----------|---------|
| Time MMD & TTC | ✅ Complete | - | - | - | MIT (engaging) |

#### Lynx Film
| Dataset | Status | Suite ID | Experiment ID | Location | Compute |
|---------|--------|----------|---------------|----------|---------|
| Time MMD & TTC | 🟡 Running | - | - | - | MIT Normal GPU (L40s) |

#### Lynx Film Raw
| Dataset | Status | Suite ID | Experiment ID | Location | Compute |
|---------|--------|----------|---------------|----------|---------|
| Time MMD & TTC | ✅ Complete | `lynx_film_raw_time_mmd_ttc_20260112_165034` | - | `output/lynx_film_raw_time_mmd_ttc_20260112_165034/` | MIT Normal GPU |
| Evaluation | ✅ Complete | - | - | `context/logs/lynx_film_raw_time_mmd_ttc_evaluation.log` | - |

#### Lynx Film Enhanced Ablations
| Status | Suite ID | Experiment ID | Location | Compute |
|--------|----------|---------------|----------|---------|
| ⏳ Pending | - | - | - | MIT Normal GPU (L40s) |

---

## Experimentation & Development

**Status:** 🟡 In Progress

| Task | Status | Notes |
|------|--------|-------|
| GPU Utilization Optimization | 🟡 In Progress | Currently 3-4% GPU utilization, need to increase |
| Lynx Film Enhanced Configurations | 🟡 In Progress | Testing all configurations |
| MMTSFLib-FedFormer Setup | ⏳ Pending | Make performant and able to run on fidel-ts |
| ZhangHanBest-PatchTST Setup | ⏳ Pending | Make performant and able to run on fidel-ts |
| ZhangHanBest-DLinear Setup | ⏳ Pending | Make performant and able to run on fidel-ts |
| MMVision Best Configuration | ⏳ Pending | Select best from lynx_film_enhanced or lynx_film_raw |

---

## Completed Test Suites

| Suite ID | Method | Description | Results |
|----------|--------|-------------|---------|
| `lynx_test_20251208_134724` | Lynx | Initial test on single dataset/horizon | Underperformed; stopped early at 15 epochs |
| `tgtsf_test_20251207_145025` | TGTSF (FIATS) | Initial test on single dataset/horizon | Performed well; 21 epochs |
| `itransformer_pretraining_test_20251205_163130` | iTransformer | Initial test; used as frozen base for lynx_test | Baseline established |
| `film_test_20251209_143628` | Lynx Film | Initial test on single dataset/horizon | Better than TGTSF and Lynx, but not as good as lynx_film_raw |
| `lynx_film_raw_test_20251209_174125` | Lynx Film Raw | Initial test on single dataset/horizon | Excellent performance; all gains from first epoch |

---

## Compute Resource Allocation

### High Compute Resources

| Resource | Allocation | Status |
|----------|------------|--------|
| MIT Preemptable (L40s x 4) | Lynx Film Raw sweep (NYC Traffic) | 🟡 Running |
| MIT Sloan (A100) | Time-LLM on Fidel-TS | 🟡 Running |
| MIT Sloan (A100) | TimeCMA on Fidel-TS | 🟡 Running |
| RunPod (L4 x 6) | iTransformer pretraining on Fidel-TS | ⏳ Need to spin up |
| RunPod (A40 x 6) | TGTSF on Fidel-TS | ⏳ Pending |
| RunPod (RTX 4000 Ada x 5) | LeRet on Fidel-TS | ⏳ Pending |
| RunPod (L40s x 5) | Lynx Film Raw initial on Fidel-TS | ⏳ Pending |
| TBD | Lynx Film on Fidel-TS | ⏳ Pending |
| TBD | Lynx Film Enhanced configurations | ⏳ Pending |
| TBD | ZhangHanBest-PatchTST | ⏳ Pending |
| TBD | ZhangHanBest-DLinear | ⏳ Pending |
| TBD | MMTSFLib-FedFormer | ⏳ Pending |
| TBD | MMVision best configuration | ⏳ Pending |

### Low Compute Resources

| Resource | Allocation | Status |
|----------|------------|--------|
| MIT Normal GPU (L40s) | Low compute quick runs | 🟡 Running |
| MIT Normal GPU (L40s) | Experimentation | 🟡 In Progress |
| RunPod (1x) | GPU utilization testing | ⏳ Pending |

---

## Notes & Workflows

### RunPod → Engaging Workflow
- RunPod jobs complete on pod, then results must be SCP-ed to engaging (output subfolder recursive)
- Storage volume contains persistent storage for all RunPod jobs (attached to all pods)
- Spin pods on community cloud for pricing and availability
- One pod per Fidel-TS dataset (run everything on that dataset on that pod)
- Logging: Use `| tee /workspace/command.log` for command output
- Save tmux session: `capture-pane -S - -E -; save-buffer /workspace/session.log` after Ctrl-B
- SCP command: `scp -P <port> -i ~/.ssh/id_ed25519 root@<IP>:/workspace/session.log ~/Downloads/`

### Experiment Workflow
- When running experiments (not sweeps): Init only first manually, then put requeue on preemptable with resumption-id and resume-suite-id in configs before submitting sbatch job

### Known Issues
- **TimeCMA on CAISO**: OOM issues, reduced chunk size to 50k from 500k (flag might not be working, may need to adjust actual config)
- **GPU Utilization**: Currently 3-4% on average (CPU bound, not GPU bound). Need to optimize for better utilization.

---

## Status Legend

- ✅ Complete
- 🟡 In Progress / Running
- ⏳ Pending
- ❌ Failed / Blocked
