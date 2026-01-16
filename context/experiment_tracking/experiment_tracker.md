╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ EXPERIMENT TRACKER                                                                                                                         ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

**Last Updated:** 2025-01-13

**Hetero stride note:** All multimodal experiments that use text need to be rerun without hetero striding (full-resolution text). TimeCMA LLM embeddings are not affected and remain valid.

┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ OVERVIEW: PLANNED RESULTS                                                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

- **[Point A]** FIATS outperforms unimodal on fidel-ts test sets (except CAISO). Show efficiency and performance on big datasets, compare to other multimodal methods, show loss curves.
- **[Point B]** Beat unimodal consistently on time mmd and ttc (where multimodal typically can't). Show we do better than time mmd method on big datasets.
- **[Point C]** Show that learning a residual can be damaging; learning the full signal is better.
- **[Point D]** Show prediction level ensembles are useful, and present other methods for ensuring text is useful but doesn't pollute time series signal when alternative modality isn't useful.

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ PRIORITY SUMMARY                                                                                                                           │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

─── High Compute Priorities ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
- (1) iTransformer pretraining on fidel-ts
- (2) TGTSF loss curves on fidel-ts
- (3) Other benchmarks on fidel-ts (leret, timecma, time-llm, zhanghanbest-patchtst, zhanghanbest-dlinear, mmtsflib-fedformer)
- (4) lynx_film_raw initial on fidel-ts
- (5) lynx_film on fidel-ts (after pretraining complete)
- (6) lynx_film_enhanced select configurations on fidel-ts (after testing, low compute ablations, and itransformer pretraining complete)
- (7) mmvision best configuration of lynx film enhanced or lynx film raw

─── Low Compute Priorities ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
- (1) itransformer pretraining on time mmd and ttc [training complete, evaluation pending]
- (2) lynx_film on time mmd and ttc [complete]
- (3) lynx_film_enhanced ablations on time mmd and ttc [pending]

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ CURRENT WORK                                                                                                                               │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

> - HETERO STRIDE: Plan rerun of all text-using experiments without hetero striding (full-resolution text); only TimeCMA LLM embeddings are kept as-is.
> ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── rerunning time mmd ttc lynx_film_raw with full resolution
> - tensor_cache test (small dataset first, then large dataset, both on cpu interactive) -> cache on engaging cpu -> speed ups on high compute -> spin up high compute jobs with tensor cache scp to pods -- testing on jena_atmospheric... 
> - once it's tested and works, add a tcache column to all of the tracking in this document -> checkmark when tensor cache generated, which should always be done before running the experiment
> -- tensor cache works, but the cacheing takes up too much space (over 1TB just for bear room) due to lots of duplication. Looking into solutions. Once implemented, test again on Bear room (on a non-gpu node) then if it works properly, we proceed with high compute jobs.
> - lynx_film_enhanced working -> low compute -> analysis
> - eval and tables on low compute -> analysis

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ TENSOR CACHE GENERATION                                                                                                                    │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

**Status:** [~]  
**Purpose:** Generate tensor caches for all fidel-ts datasets to speed up training runs

| Dataset | Status | Compute |
|--------|----------|--------|
| Canada Photovoltaics | [ ] | RunPod (pod:canada) |
| Germany Renewable | [~] | RunPod (pod:germany) |
| NYC Traffic Speed | [~] | RunPod (pod:nyc_traffic) |
| California ISO | [ ] | RunPod (pod:caiso) |
| Bear Room | [ ] | RunPod (pod:bear_room) |
| Jena Atmospheric Physics | [✓] | RunPod (pod:jena) |

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ HIGH PRIORITY EXPERIMENTS                                                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

─── Priority 1: iTransformer Pretraining (Baselines) ─────────────────────────────────────────────────────────────────────────────────────────

**Status:** [~]  
**Purpose:** Unimodal baselines needed for all datasets before multimodal comparisons

#### Fidel-TS Datasets
| Dataset | Status | Suite ID | Experiment ID | Eval | Tables | Compute |
|--------|----------|---------|--------|--------|----------|--------|
| Canada Photovoltaics | [✓] | `itransformer_canada_photovoltaics_20260113_101629` | `20260113-101629_d36af4d594fb` | - | - | RunPod (pod:canada) |
| Germany Renewable | [>>] | - | - | - | - | RunPod (pod:germany) |
| NYC Traffic Speed | [ ] | - | - | - | - | RunPod (pod:nyc_traffic) |
| California ISO | [ ] | - | - | - | - | RunPod (pod:caiso) |
| Bear Room | [ ] | - | - | - | - | RunPod (pod:bear_room) |
| Jena Atmospheric Physics | [ ] | - | - | - | - | RunPod (pod:jena) |

#### Time MMD & TTC
| Dataset | Status | Suite ID | Experiment ID | Eval | Tables | Compute |
|--------|----------|---------|--------|--------|----------|--------|
| Time MMD & TTC | [✓] | `itransformer_time_mmd_ttc_20260113_111324` | - | [✓] | - | MIT (engaging) |

**Note:** Evaluation log at `context/logs/itransformer_time_mmd_ttc_evaluation.log` (should be moved to `configs/experiment_suites/itransformer/time_mmd_ttc.eval`)

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

─── Priority 2: TGTSF (FIATS) Loss Curves on Fidel-TS ────────────────────────────────────────────────────────────────────────────────────────

**Status:** [ ]  
**Purpose:** Generate loss curves for comparison with other multimodal methods

| Dataset | Status | Suite ID | Experiment ID | Eval | Tables | Compute |
|--------|----------|---------|--------|--------|----------|--------|
| Canada Photovoltaics | [ ] | - | - | - | - | RunPod (pod:canada) |
| Germany Renewable | [ ] | - | - | - | - | RunPod (pod:germany) |
| NYC Traffic Speed | [ ] | - | - | - | - | RunPod (pod:nyc_traffic) |
| California ISO | [ ] | - | - | - | - | RunPod (pod:caiso) |
| Bear Room | [ ] | - | - | - | - | RunPod (pod:bear_room) |
| Jena Atmospheric Physics | [ ] | - | - | - | - | RunPod (pod:jena) |

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

─── Priority 3: Benchmark Methods on Fidel-TS ────────────────────────────────────────────────────────────────────────────────────────────────

**Status:** [~]  
**Purpose:** Compare against other multimodal methods

#### Time-LLM
| Dataset | Status | Suite ID | Experiment ID | Eval | Tables | Compute |
|--------|----------|---------|--------|--------|----------|--------|
| California ISO | [ ] | `timellm_california_iso_20260112_141144` | `20260112-141144_a05ebe165526` | - | - | MIT Sloan (A100) |
| NYC Traffic Speed | [ERR] (needs resume) | `timellm_nyc_traffic_speed_20260112_185045` | - | - | - | MIT Sloan (A100) |
| Canada Photovoltaics | [ ] | - | - | - | - | MIT Sloan (A100) |
| Germany Renewable | [ ] | - | - | - | - | MIT Sloan (A100) |
| Bear Room | [ ] | - | - | - | - | MIT Sloan (A100) |
| Jena Atmospheric Physics | [ ] | - | - | - | - | MIT Sloan (A100) |

#### TimeCMA
| Dataset | Status | LLM Embed | Suite ID | Experiment ID | Eval | Tables | Compute |
|--------|----------|----------|---------|--------|--------|----------|--------|
| Time MMD & TTC | [ ] | [✓] | - | - | - | - | MIT Sloan (A100) |
| Bear Room | [ ] | [✓] | - | - | - | - | MIT Sloan (A100) |
| California ISO | [ ] | [✓] | - | - | - | - | MIT Sloan (A100) |
| Canada Photovoltaics | [ ] | [✓]  | - | - | - | - | MIT Sloan (A100) |
| Germany Renewable | [ ] | [✓] | - | - | - | - | MIT Sloan (A100) |
| NYC Traffic Speed | [ ] | [✓] | - | - | - | - | MIT Sloan (A100) |
| Jena Atmospheric Physics | [ ] | [✓] | - | - | - | - | MIT Sloan (A100) |

**Note:** TimeCMA has two steps: (1) LLM embedding generation, (2) model training. Suite ID and training status are all pending for Fidel-TS datasets.

#### LeRet
| Dataset | Status | Suite ID | Experiment ID | Eval | Tables | Compute |
|--------|----------|---------|--------|--------|----------|--------|
| Canada Photovoltaics | [ ] | - | - | - | - | RunPod (pod:canada) |
| Germany Renewable | [ ] | - | - | - | - | RunPod (pod:germany) |
| NYC Traffic Speed | [ ] | - | - | - | - | RunPod (pod:nyc_traffic) |
| California ISO | [ ] | - | - | - | - | RunPod (pod:caiso) |
| Bear Room | [ ] | - | - | - | - | RunPod (pod:bear_room) |
| Jena Atmospheric Physics | [ ] | - | - | - | - | RunPod (pod:jena) |

#### MMTSFLib
| Dataset | Status | LLM Embed | Suite ID | Experiment ID | Eval | Tables | Compute |
|--------|----------|----------|---------|--------|--------|----------|--------|
| Time MMD & TTC | [ ] | [ ] | - | - | - | - | - |
| Canada Photovoltaics | [ ] | [ ] | - | - | - | - | - |
| Germany Renewable | [ ] | [ ] | - | - | - | - | - |
| NYC Traffic Speed | [ ] | [ ] | - | - | - | - | - |
| California ISO | [ ] | [ ] | - | - | - | - | - |
| Bear Room | [ ] | [ ] | - | - | - | - | - |
| Jena Atmospheric Physics | [ ] | [ ] | - | - | - | - | - |

**Note:** MMTSFlib requires precomputed LLM embeddings. All experiments are pending setup and execution.

#### Other Benchmarks (Pending Experimentation)
- **ZhangHanBest-PatchTST**: [ ] (needs experimentation setup)
- **ZhangHanBest-DLinear**: [ ] (needs experimentation setup)
- **MMTSFLib-FedFormer**: [ ] (needs experimentation setup)

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

─── Priority 4: Lynx Film Raw on Fidel-TS ────────────────────────────────────────────────────────────────────────────────────────────────────

**Status:** [~]  
**Purpose:** Initial runs before hyperparameter tuning complete

#### Hyperparameter Sweep (NYC Traffic)
| Status | Suite ID | Experiment ID | Eval | Tables | Compute |
|----------|---------|--------|--------|----------|--------|
| [||] Paused (new job submission) | `sweep_v2dy5lr5` | - | - | - | MIT Preemptable (L40s x 4) |

**Notes:**
- New job submission paused until tensor cache is working to speed up runs
- Findings: Learning rate 0.0004 with d_model 768, e_layers 2, n_heads 4 shows good generalization
- Every single run gets best val loss after a single epoch → set max_epochs to 1
- Will likely create a new sweep that uses tensor cache, sweeps closer around these optimal points, and only uses a single epoch

#### Initial Runs (Other Datasets)
| Dataset | Status | Suite ID | Experiment ID | Eval | Tables | Compute |
|--------|----------|---------|--------|--------|----------|--------|
| NYC Traffic Speed | [ ] | - | - | - | - | RunPod (pod:nyc_traffic) |
| Canada Photovoltaics | [ ] | - | - | - | - | RunPod (pod:canada) |
| Germany Renewable | [ ] | - | - | - | - | RunPod (pod:germany) |
| California ISO | [ ] | - | - | - | - | RunPod (pod:caiso) |
| Bear Room | [ ] | - | - | - | - | RunPod (pod:bear_room) |
| Jena Atmospheric Physics | [ ] | - | - | - | - | RunPod (pod:jena) |

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

─── Priority 5: Lynx Film on Fidel-TS ────────────────────────────────────────────────────────────────────────────────────────────────────────

**Status:** [~]  
**Purpose:** Compare with residual learning (Point C)

| Dataset | Status | Suite ID | Experiment ID | Eval | Tables | Compute |
|--------|----------|---------|--------|--------|----------|--------|
| Canada Photovoltaics | [ ] | `lynx_film_canada_photovoltaics_20260113_105228` | `20260113-105229_286069803489` | - | - | RunPod (pod:canada) |
| Germany Renewable | [ ] (testing tensor_cache) | - | - | - | - | RunPod (pod:germany) |
| NYC Traffic Speed | [ ] | - | - | - | - | RunPod (pod:nyc_traffic) |
| California ISO | [ ] | - | - | - | - | RunPod (pod:caiso) |
| Bear Room | [ ] | - | - | - | - | RunPod (pod:bear_room) |
| Jena Atmospheric Physics | [ ] | - | - | - | - | RunPod (pod:jena) |

**Note:** Requires iTransformer pretraining to complete first.

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

─── Priority 6: Lynx Film Enhanced (Select Configurations) ───────────────────────────────────────────────────────────────────────────────────

**Status:** [ ]  
**Purpose:** Ablations on various ways to mix with gating (Point D)

**Prerequisites:**
- [✓] Testing complete
- [✓] Low compute ablations complete
- [✓] iTransformer pretraining complete

| Configuration | Status | Suite ID | Experiment ID | Eval | Tables | Compute |
|--------|----------|---------|--------|--------|----------|--------|
| TBD | [ ] | - | - | - | - | TBD (x6 x nconfigurations) |

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

─── Priority 7: MMVision Best Configuration ──────────────────────────────────────────────────────────────────────────────────────────────────

**Status:** [ ]  
**Purpose:** Final best configuration selection

**Prerequisites:**
- [✓] Lynx Film Enhanced configurations tested
- [✓] Best configuration identified

| Status | Suite ID | Experiment ID | Eval | Tables | Compute |
|----------|---------|--------|--------|----------|--------|
| [ ] | - | - | - | - | TBD (x1) |

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ LOW COMPUTE EXPERIMENTS                                                                                                                    │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

─── Quick Runs on Time MMD & TTC ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

**Status:** [~]  
**Purpose:** Validate multimodal performance on smaller datasets (Point B)

#### iTransformer Pretraining
| Dataset | Status | Suite ID | Experiment ID | Eval | Tables | Compute |
|--------|----------|---------|--------|--------|----------|--------|
| Time MMD & TTC | [✓] | `itransformer_time_mmd_ttc_20260113_111324` | - | [✓] | - | MIT (engaging) |

#### Lynx Film
| Dataset | Status | Suite ID | Experiment ID | Eval | Tables | Compute |
|--------|----------|---------|--------|--------|----------|--------|
| Time MMD & TTC | [✓] | `lynx_film_time_mmd_ttc_20260113_120342` | - | - | - | MIT Normal GPU (L40s) |

#### Lynx Film Raw
| Dataset | Status | Suite ID | Experiment ID | Eval | Tables | Compute |
|--------|----------|---------|--------|--------|----------|--------|
| Time MMD & TTC | [✓] | `lynx_film_raw_time_mmd_ttc_20260112_165034` | - | [✓] | - | MIT Normal GPU |

#### Lynx Film Enhanced Ablations
| Status | Suite ID | Experiment ID | Eval | Tables | Compute |
|----------|---------|--------|--------|----------|--------|
| [ ] | - | - | - | - | MIT Normal GPU (L40s) |

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ EXPERIMENTATION & DEVELOPMENT                                                                                                              │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

**Status:** [~]

| Task | Status | Notes |
|--------|----------|---------|
| GPU Utilization Optimization | [~] | Currently 3-4% GPU utilization, need to increase |
| Lynx Film Enhanced Configurations | [~] | Testing all configurations |
| MMTSFLib-FedFormer Setup | [ ] | Make performant and able to run on fidel-ts |
| ZhangHanBest-PatchTST Setup | [ ] | Make performant and able to run on fidel-ts |
| ZhangHanBest-DLinear Setup | [ ] | Make performant and able to run on fidel-ts |
| MMVision Best Configuration | [ ] | Select best from lynx_film_enhanced or lynx_film_raw |

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ LEGACY COMPLETED TEST SUITES                                                                                                               │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

| Suite ID | Method | Description | Results |
|---------|----------|---------|--------|
| `lynx_test_20251208_134724` | Lynx | Initial test on single dataset/horizon | Underperformed; stopped early at 15 epochs |
| `tgtsf_test_20251207_145025` | TGTSF (FIATS) | Initial test on single dataset/horizon | Performed well; 21 epochs |
| `itransformer_pretraining_test_20251205_163130` | iTransformer | Initial test; used as frozen base for lynx_test | Baseline established |
| `film_test_20251209_143628` | Lynx Film | Initial test on single dataset/horizon | Better than TGTSF and Lynx, but not as good as lynx_film_raw |
| `lynx_film_raw_test_20251209_174125` | Lynx Film Raw | Initial test on single dataset/horizon | Excellent performance; all gains from first epoch |

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ COMPUTE RESOURCE ALLOCATION                                                                                                                │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

─── High Compute Resources ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────

| Resource | Allocation | Status |
|---------|--------|----------|
| MIT Preemptable (L40s x 4) | Lynx Film Raw sweep (NYC Traffic) | [||] Paused (new job submission) |
| MIT Sloan (A100) | Time-LLM on Fidel-TS | [>>] |
| MIT Sloan (A100) | TimeCMA on Fidel-TS | [✗] Crashed |
| RunPod-Canada | iTransformer pretraining, TGTSF, LeRet, Lynx Film Raw, Lynx Film | [ ] |
| RunPod-Germany | iTransformer pretraining, TGTSF, LeRet, Lynx Film Raw, Lynx Film | [ ] |
| RunPod-NYC Traffic | iTransformer pretraining, TGTSF, LeRet, Lynx Film Raw, Lynx Film | [ ] |
| RunPod-CAISO | iTransformer pretraining, TGTSF, LeRet, Lynx Film Raw, Lynx Film | [ ] |
| RunPod-Bear Room | iTransformer pretraining, TGTSF, LeRet, Lynx Film Raw, Lynx Film | [ ] |
| RunPod-Jena | iTransformer pretraining, TGTSF, LeRet, Lynx Film Raw, Lynx Film | [ ] |

─── Low Compute Resources ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

| Resource | Allocation | Status |
|---------|--------|----------|
| MIT Normal GPU (L40s) | Low compute quick runs | [>>] |
| MIT Normal GPU (L40s) | Experimentation | [~] |
| RunPod (1x) | GPU utilization testing | [ ] |

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ NOTES & WORKFLOWS                                                                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

─── RunPod → Engaging Workflow ───────────────────────────────────────────────────────────────────────────────────────────────────────────────
- RunPod jobs complete on pod, then results must be SCP-ed to engaging (output subfolder recursive)
- Storage volume contains persistent storage for all RunPod jobs (attached to all pods)
- Spin pods on community cloud for pricing and availability
- One pod per Fidel-TS dataset (run everything on that dataset on that pod)
- Logging: Use `| tee /workspace/command.log` for command output
- Save tmux session: `capture-pane -S - -E -; save-buffer /workspace/session.log` after Ctrl-B
- SCP command: `scp -P <port> -i ~/.ssh/id_ed25519 root@<IP>:/workspace/session.log ~/Downloads/`

─── Experiment Workflow ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
- When running experiments (not sweeps): Init only first manually, then put requeue on preemptable with resumption-id and resume-suite-id in configs before submitting sbatch job

─── Evaluation Log Pattern ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
- Evaluation logs should be moved from `context/logs/` to `configs/experiment_suites/<method>/<dataset>.eval`
- Example: `context/logs/itransformer_time_mmd_ttc_evaluation.log` → `configs/experiment_suites/itransformer/time_mmd_ttc.eval`

─── Known Issues ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
- **TimeCMA on CAISO**: OOM issues, reduced chunk size to 50k from 500k (flag might not be working, may need to adjust actual config)
- **GPU Utilization**: Currently 3-4% on average (CPU bound, not GPU bound). Need to optimize for better utilization.

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STATUS LEGEND                                                                                                                              │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

- [✓] Complete
- [~] In Progress
- [>>] Running
- [ ] Pending
- [✗] Failed / Blocked
- [||] Paused
- [ERR] Error
