╔════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ EXPERIMENT TRACKER                                                                              ║
╚════════════════════════════════════════════════════════════════════════════════════════════════════╝

**Last Updated:** 2025-01-13

**Hetero stride note:** All multimodal experiments that use text need to be rerun without hetero striding (full-resolution text). TimeCMA LLM embeddings are not affected and remain valid.

┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ OVERVIEW: PLANNED RESULTS                                                                       │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘

- **[Point A]** FIATS outperforms unimodal on fidel-ts test sets (except CAISO). Show efficiency and performance on big datasets, compare to other multimodal methods, show loss curves.
- **[Point B]** Beat unimodal consistently on time mmd and ttc (where multimodal typically can't). Show we do better than time mmd method on big datasets.
- **[Point C]** Show that learning a residual can be damaging; learning the full signal is better.
- **[Point D]** Show prediction level ensembles are useful, and present other methods for ensuring text is useful but doesn't pollute time series signal when alternative modality isn't useful.

────────────────────────────────────────────────────────────────────────────────────────────────────

┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ PRIORITY SUMMARY                                                                                │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘

─── High Compute Priorities ──────────────────────────────────────────────────────────────────────
- (1) iTransformer pretraining on fidel-ts
- (2) TGTSF loss curves on fidel-ts
- (3) Other benchmarks on fidel-ts (leret, timecma, time-llm, zhanghanbest-patchtst, zhanghanbest-dlinear, mmtsflib-fedformer)
- (4) lynx_film_raw initial on fidel-ts
- (5) lynx_film on fidel-ts (after pretraining complete)
- (6) lynx_film_enhanced select configurations on fidel-ts (after testing, low compute ablations, and itransformer pretraining complete)
- (7) mmvision best configuration of lynx film enhanced or lynx film raw

─── Low Compute Priorities ───────────────────────────────────────────────────────────────────────
- (1) itransformer pretraining on time mmd and ttc [training complete, evaluation pending]
- (2) lynx_film on time mmd and ttc [complete]
- (3) lynx_film_enhanced ablations on time mmd and ttc [pending]

────────────────────────────────────────────────────────────────────────────────────────────────────

┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ CURRENT WORK                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘

> - HETERO STRIDE: Plan rerun of all text-using experiments without hetero striding (full-resolution text); only TimeCMA LLM embeddings are kept as-is.
> ────────────────────────────────────────────────────────────────────────────── rerunning time mmd ttc lynx_film_raw with full resolution
> - tensor_cache test (small dataset first, then large dataset, both on cpu interactive) -> cache on engaging cpu -> speed ups on high compute -> spin up high compute jobs with tensor cache scp to pods -- testing on jena_atmospheric... 
> - once it's tested and works, add a tcache column to all of the tracking in this document -> checkmark when tensor cache generated, which should always be done before running the experiment
> -- tensor cache works, but the cacheing takes up too much space (over 1TB just for bear room) due to lots of duplication. Looking into solutions. Once implemented, test again on Bear room (on a non-gpu node) then if it works properly, we proceed with high compute jobs.
> - lynx_film_enhanced working -> low compute -> analysis
> - eval and tables on low compute -> analysis

────────────────────────────────────────────────────────────────────────────────────────────────────

┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ TENSOR CACHE GENERATION                                                                         │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘

**Status:** [~]  
**Purpose:** Generate tensor caches for all fidel-ts datasets to speed up training runs

| Dataset | Status | Compute |
|--------|----------|--------|
| Canada Photovoltaics | [✓] | chimera_proj [need to port]-> (pod:canada) |
| Germany Renewable | [✓] | chimera_proj [need to port]-> (pod:germany) |
| NYC Traffic Speed | [✓] | chimera_proj [need to port]-> (pod:nyc_traffic) |
| California ISO | [✓] | chimera_proj [need to port]-> (pod:caiso) |
| Bear Room | [✓] | RunPod (chimera_proj [need to port]-> pod:bear_room) |
| Jena Atmospheric Physics | [✓] | chimera_proj [need to port]-> (pod:jena) |

note: I had generated nyc traffic speed on pod:nyc_traffic, but that was faulty before the splits fix (had the anomalous validation loss). Hence, I need to redo that.

────────────────────────────────────────────────────────────────────────────────────────────────────

┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ HIGH PRIORITY EXPERIMENTS                                                                       │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘

─── Priority 1: iTransformer Pretraining (Baselines) ──────────────────────────────────────────

**Status:** [~]  
**Purpose:** Unimodal baselines needed for all datasets before multimodal comparisons

#### Fidel-TS Datasets
| Dataset | Status | Suite ID | Experiment ID | Eval | Tables | Compute |
|--------|----------|---------|--------|--------|----------|--------|
| Canada Photovoltaics | [✓] | `itransformer_canada_photovoltaics_20260119_201134` | `20260119-201134_c94f3a672169` | - | - | RunPod (pod:canada) |
| Germany Renewable | [✓] | `itransformer_germany_renewable_20260119_211905` | `20260119-211905_5146f0e1f5d1` | - | - | RunPod (pod:germany) |
| NYC Traffic Speed | [✓] | `itransformer_nyc_traffic_speed_20260119_185413` | `20260119-185414_86a8624cf255` | - | - | RunPod (pod:nyc_traffic) |
| California ISO | [✓] | `itransformer_california_iso_20260119_205950` | `20260119-205950_1d71374b9677` | - | - | RunPod (pod:caiso) |
| Bear Room | [✓] | `itransformer_bear_room_20260120_013208` | `20260120-013208_17b7f42e444a` | - | - | RunPod (pod:bear_room) |
| Jena Atmospheric Physics | [✓] | `itransformer_jena_atmospheric_20260119_215144` | `20260119-215144_99e92264ac3f` | - | - | RunPod (pod:jena) |

note (10/18 3 am): canada and germany were completed on runpods.
note (10/18 3:45 am): caiso, nyc, bear room, jena submitted with train truncated for purge and using tensor cache, all on mit_preemptable 

#### Time MMD & TTC
| Dataset | Status | Suite ID | Experiment ID | Eval | Tables | Compute |
|--------|----------|---------|--------|--------|----------|--------|
| Time MMD & TTC | [✓] | `itransformer_time_mmd_ttc_20260113_111324` | - | [✓] | - | MIT (engaging) |

**Note:** Evaluation log at `context/logs/itransformer_time_mmd_ttc_evaluation.log` (should be moved to `configs/experiment_suites/itransformer/time_mmd_ttc.eval`)

────────────────────────────────────────────────────────────────────────────────────────────────────

─── Priority 2.1: TGTSF (FIATS) Loss Curves on Fidel-TS (Full Resolution) ──────────────────────────

**Status:** [ ]  
**Purpose:** Generate loss curves for comparison with other multimodal methods

| Dataset | Status | Suite ID | Experiment ID | Eval | Tables | Compute |
|--------|----------|---------|--------|--------|----------|--------|
| Canada Photovoltaics | [✓] | `tgtsf_canada_photovoltaics_20260120_080941` | `20260120-080941_916143b7bacd` | - | - | mit_preemptable |
| Germany Renewable | [>>] | - | - | - | - | mit_preemptable |
| NYC Traffic Speed | [✓] | `tgtsf_nyc_traffic_speed_20260119_192618` | `20260119-192618_2a4450cc3082` | - | - | mit_preemptable | 
| California ISO | [~] | - | - | - | - | mit_preemptable |
| Bear Room | [~] | - | - | - | - | mit_preemptable |
| Jena Atmospheric Physics | [~] | - | - | - | - | mit_preemptable |

note: memory hungry so run on preemptable rather than RTX Ada 4000/5000

────────────────────────────────────────────────────────────────────────────────────────────────────


─── Priority 2.2: TGTSF (FIATS) Loss Curves on Fidel-TS (Original Resolution) ──────────────────────

**Status:** [ ]  
**Purpose:** Generate loss curves for comparison with other multimodal methods

| Dataset | Status | Suite ID | Experiment ID | Eval | Tables | Compute |
|--------|----------|---------|--------|--------|----------|--------|
| Canada Photovoltaics | [~] | - | - | - | - | mit sloan |
| Germany Renewable | [~] | - | - | - | - | mit sloan |
| NYC Traffic Speed | [~] | - | - | - | - | mit sloan |
| California ISO | [~] | - | - | - | - | mit sloan |
| Bear Room | [~] | - | - | - | - | mit sloan |
| Jena Atmospheric Physics | [~] | - | - | - | - | mit sloan |

note: run after the full resolution is complete

────────────────────────────────────────────────────────────────────────────────────────────────────

note: need to generate original resolution tensor caches 

─── Priority 3: Benchmark Methods on Fidel-TS ─────────────────────────────────────────────────

**Status:** [~]  
**Purpose:** Compare against other multimodal methods

#### Time-LLM
| Dataset | Status | Suite ID | Experiment ID | Eval | Tables | Compute |
|--------|----------|---------|--------|--------|----------|--------|
| California ISO | [✓] | `timellm_california_iso_20260112_141144` | `20260112-141144_a05ebe165526` | - | - | MIT Sloan (A100) |
| NYC Traffic Speed | [✓] (1; nan loss; I terminated; resubmitted, nan loss again, made changes and resubmitted) | `timellm_nyc_traffic_speed_20260112_185045` | - | - | - | MIT Sloan (A100) |
| Canada Photovoltaics | [✓] (nan loss; resubmitted) | - | - | - | - | MIT Sloan (A100) |
| Germany Renewable | [✓] (nan loss; resubmitted) | - | - | - | - | MIT Sloan (A100) |
| Bear Room | [✓] (time limit; nan loss; resubmitted) | - | - | - | - | MIT Sloan (A100) |
| Jena Atmospheric Physics | [✓] (nan loss; resubmitted) | `timellm_jena_atmospheric_20260118_004853` | `20260118-004855_6864cff8fa25` | - | - | MIT Sloan (A100) |

note: all but nyc traffic speed and caiso are runs from later 01/16 or early 01/17. add suite and experiment later. evals not done yet. 
(1): Had resumption error. Added `mark_last_job_complete: true` to config and tried resubmitting. 
note (10/17 11 pm): bear room ran 30 epochs then died due to time limit. 
note (10/17 11 pm): bear room and jena have nan loss.
note (10/18 1:30 am): added a stopper and mark experiment failed for nan loss, changed some things that seemed to help with the nan loss (param initialization, learning rate, etc) and resubmitted
note (10/19 3 pm): jena, germany, canada completed with numeric loss. nyc traffic has nan loss again and is still running. bear room still running but numeric loss.
note (10/19 1 pm): all complete. evals in the sbatch logs

#### TimeCMA
| Dataset | Status | LLM Embed | Suite ID | Experiment ID | Eval | Tables | Compute |
|--------|----------|----------|---------|--------|--------|----------|--------|
| Time MMD & TTC | [ ] | [✓] | - | - | - | - | MIT Sloan (A100) |
| Bear Room | [✗] | [✓] | - | - | - | - | MIT Sloan (A100) |
| California ISO | [✗] | [✓] | - | - | - | - | MIT Sloan (A100) |
| Canada Photovoltaics | [✗] | [✓]  | - | - | - | - | MIT Sloan (A100) |
| Germany Renewable | [ ] | [✓] | - | - | - | - | MIT Sloan (A100) |
| NYC Traffic Speed | [ ] | [✓] | - | - | - | - | MIT Sloan (A100) |
| Jena Atmospheric Physics | [✗] | [✓] | - | - | - | - | MIT Sloan (A100) |

**Note:** TimeCMA has two steps: (1) LLM embedding generation, (2) model training. Suite ID and training status are all pending for Fidel-TS datasets.
note (10/18 3 am): timecma doesn't work with torch compile. that was the error before. timecma requires llm embeddings, not currently supported in tensor cache. will make tensor_cache_llm branch and support that (see temp_buffer for more info). cache will be distinct for the timecma prompt. once that's supported, we run tensor cache jobs for timecma, then the timecma jobs with use tensor cache enabled.
note (10/19 9 pm): timecma with tensor cache would need massive disk space since it's sample specific embeddings.
note (10/19 10 pm): timecma without tensor cache and without torch compile had some errors in sbatch_logs. going to spin up claude on the tensor_cache_llm branch then fix those errors then try again.

#### LeRet
| Dataset | Status | Suite ID | Experiment ID | Eval | Tables | Compute |
|--------|----------|---------|--------|--------|----------|--------|
| Canada Photovoltaics | [✓] | - | - | - | - | RunPod (pod:canada) |
| Germany Renewable | [>>] (timed out x2, submitted resume x2) | - | - | - | - | RunPod (pod:germany) |
| NYC Traffic Speed | [>>] (timed out, submitted resume) | - | - | - | - | RunPod (pod:nyc_traffic) |
| California ISO | [✓] | - | - | - | - | RunPod (pod:caiso) |
| Bear Room | [>>] (preempted, started over) | - | - | - | - | RunPod (pod:bear_room) |
| Jena Atmospheric Physics | [✓] | - | - | - | - | RunPod (pod:jena) |

note: (10/17 noon) running on preemptable; nyc traffic and bear room are waiting on a gpu (4 max on preemptable). pretraining at least was going well. 
note: (10/17 11 pm) bear room preempted epoch 1 pretrain, so starting over from scratch
note: (10/18 3 pm) going to rerun nyc traffic and germany on preemptable with requeue for 2 days each to get those done

#### MMTSFLib
| Dataset | Status | LLM Embed | Suite ID | Experiment ID | Eval | Tables | Compute |
|--------|----------|----------|---------|--------|--------|----------|--------|
| Time MMD & TTC | [ ] | [ ] | - | - | - | - | - |
| Canada Photovoltaics | [ ] | [✓] | - | - | - | - | - |
| Germany Renewable | [ ] | [✓] | - | - | - | - | - |
| NYC Traffic Speed | [ ] | [>>] | - | - | - | - | - |
| California ISO | [ ] | [✓] | - | - | - | - | - |
| Bear Room | [ ] | [✓] | - | - | - | - | - |
| Jena Atmospheric Physics | [ ] | [✓] | - | - | - | - | - |

note: all of the llm embeddings are runs from late 01/16 or early 01/17.
**Note:** MMTSFlib requires precomputed LLM embeddings. All experiments are pending setup and execution.

#### Other Benchmarks (Pending Experimentation)
- **ZhangHanBest-PatchTST**: [ ] (needs experimentation setup)
- **ZhangHanBest-DLinear**: [ ] (needs experimentation setup)
- **MMTSFLib-FedFormer**: [ ] (needs experimentation setup)

────────────────────────────────────────────────────────────────────────────────────────────────────

─── Priority 4: Lynx Film Raw on Fidel-TS ──────────────────────────────────────────────────────

#### Initial Runs (Other Datasets)
| Dataset | Status | Suite ID | Experiment ID | Eval | Tables | Compute |
|--------|----------|---------|--------|--------|----------|--------|
| NYC Traffic Speed | [>>] | `lynx_film_raw_nyc_traffic_speed_20260120_144758` | `20260120-144758_715c03fcb6a2` | - | - | RunPod (pod:nyc_traffic) |
| Canada Photovoltaics | [✓] | `lynx_film_raw_canada_photovoltaics_20260119_222019` | `20260119-222020_1a45a0aac8c7` | - | - | RunPod (pod:canada) | 
| Germany Renewable | [✓] | `lynx_film_raw_germany_renewable_20260120_051004` | `20260120-051004_19e0259912bb` | - | - | RunPod (pod:germany) | 
| California ISO | [✓] | `lynx_film_raw_california_iso_20260119_221759` | `20260119-221759_15466b14951a` | - | - | RunPod (pod:caiso) | 
| Bear Room | [>>] | `lynx_film_raw_bear_room_20260120_143656` | `20260120-143656_5eb93c6b25f7` | - | - | RunPod (pod:bear_room) | 
| Jena Atmospheric Physics | [✓] | `lynx_film_raw_jena_atmospheric_20260119_224922` | `20260119-224922_040d096de7db` | - | - | RunPod (pod:jena) | 

note: these are full hetero resolution 
note: we want to do this on original resolution for the ones that look good (make a new table for that)

────────────────────────────────────────────────────────────────────────────────────────────────────

─── Priority 5: Lynx Film on Fidel-TS ────────────────────────────────────────────────────────

**Status:** [~]  
**Purpose:** Compare with residual learning (Point C)

| Dataset | Status | Suite ID | Experiment ID | Eval | Tables | Compute |
|--------|----------|---------|--------|--------|----------|--------|
| Canada Photovoltaics | [✓] | `lynx_film_canada_photovoltaics_20260119_202157` | `20260119-202157_f5e855bd761c` | - | - | RunPod (pod:canada) |
| Germany Renewable | [✓] | `lynx_film_germany_renewable_20260119_213142` | `20260119-213142_5adc752cde19` | - | - | RunPod (pod:germany) |
| NYC Traffic Speed | [✓] | `lynx_film_nyc_traffic_speed_20260119_202319` | `20260119-202319_6c58cf9ff591` | - | - | RunPod (pod:nyc_traffic) |
| California ISO | [✓] | `lynx_film_california_iso_20260119_211605` | `20260119-211605_780ef802fcf8` | - | - | RunPod (pod:caiso) |
| Bear Room | [✓] | `lynx_film_bear_room_20260120_021406` | `20260120-021407_775b5d8b4d66` | - | - | RunPod (pod:bear_room) |
| Jena Atmospheric Physics | [✓] | `lynx_film_jena_atmospheric_20260119_215914` | `20260119-215914_27b837b9f927` | - | - | RunPod (pod:jena) |

**Note:** Requires iTransformer pretraining to complete first.
note: these are full hetero resolution

────────────────────────────────────────────────────────────────────────────────────────────────────

─── Priority 6: Lynx Film Enhanced (Select Configurations) ────────────────────────────────────

**Status:** [ ]  
**Purpose:** Ablations on various ways to mix with gating (Point D)

**Prerequisites:**
- [✓] Testing complete
- [✓] Low compute ablations complete
- [✓] iTransformer pretraining complete

| Configuration | Status | Suite ID | Experiment ID | Eval | Tables | Compute |
|--------|----------|---------|--------|--------|----------|--------|
| TBD | [ ] | - | - | - | - | TBD (x6 x nconfigurations) |

note: we want to do this on original resolution for the ones that look good

────────────────────────────────────────────────────────────────────────────────────────────────────

─── Priority 7: MMVision Best Configuration ────────────────────────────────────────────────────

**Status:** [ ]  
**Purpose:** Final best configuration selection

**Prerequisites:**
- [✓] Lynx Film Enhanced configurations tested
- [✓] Best configuration identified

| Status | Suite ID | Experiment ID | Eval | Tables | Compute |
|----------|---------|--------|--------|----------|--------|
| [ ] | - | - | - | - | TBD (x1) |

────────────────────────────────────────────────────────────────────────────────────────────────────

┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ LOW COMPUTE EXPERIMENTS                                                                         │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘

─── Quick Runs on Time MMD & TTC ─────────────────────────────────────────────────────────────────

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

────────────────────────────────────────────────────────────────────────────────────────────────────

┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ EXPERIMENTATION & DEVELOPMENT                                                                   │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘

**Status:** [~]

| Task | Status | Notes |
|--------|----------|---------|
| GPU Utilization Optimization | [~] | Currently 3-4% GPU utilization, need to increase |
| Lynx Film Enhanced Configurations | [~] | Testing all configurations |
| MMTSFLib-FedFormer Setup | [ ] | Make performant and able to run on fidel-ts |
| ZhangHanBest-PatchTST Setup | [ ] | Make performant and able to run on fidel-ts |
| ZhangHanBest-DLinear Setup | [ ] | Make performant and able to run on fidel-ts |
| MMVision Best Configuration | [ ] | Select best from lynx_film_enhanced or lynx_film_raw |

────────────────────────────────────────────────────────────────────────────────────────────────────

┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ LEGACY COMPLETED TEST SUITES                                                                    │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘

| Suite ID | Method | Description | Results |
|---------|----------|---------|--------|
| `lynx_test_20251208_134724` | Lynx | Initial test on single dataset/horizon | Underperformed; stopped early at 15 epochs |
| `tgtsf_test_20251207_145025` | TGTSF (FIATS) | Initial test on single dataset/horizon | Performed well; 21 epochs |
| `itransformer_pretraining_test_20251205_163130` | iTransformer | Initial test; used as frozen base for lynx_test | Baseline established |
| `film_test_20251209_143628` | Lynx Film | Initial test on single dataset/horizon | Better than TGTSF and Lynx, but not as good as lynx_film_raw |
| `lynx_film_raw_test_20251209_174125` | Lynx Film Raw | Initial test on single dataset/horizon | Excellent performance; all gains from first epoch |

────────────────────────────────────────────────────────────────────────────────────────────────────

┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ COMPUTE RESOURCE ALLOCATION                                                                     │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘

─── High Compute Resources ──────────────────────────────────────────────────────────────────────

| Resource | Allocation | Status |
|---------|--------|----------|
| MIT Preemptable (L40s x 4) | Lynx Film Raw sweep (NYC Traffic) | [-] Paused (new job submission) |
| MIT Sloan (A100) | Time-LLM on Fidel-TS | [>>] |
| MIT Sloan (A100) | TimeCMA on Fidel-TS | [✗] Crashed |
| RunPod-Canada | iTransformer pretraining, TGTSF, LeRet, Lynx Film Raw, Lynx Film | [ ] |
| RunPod-Germany | iTransformer pretraining, TGTSF, LeRet, Lynx Film Raw, Lynx Film | [ ] |
| RunPod-NYC Traffic | iTransformer pretraining, TGTSF, LeRet, Lynx Film Raw, Lynx Film | [ ] |
| RunPod-CAISO | iTransformer pretraining, TGTSF, LeRet, Lynx Film Raw, Lynx Film | [ ] |
| RunPod-Bear Room | iTransformer pretraining, TGTSF, LeRet, Lynx Film Raw, Lynx Film | [ ] |
| RunPod-Jena | iTransformer pretraining, TGTSF, LeRet, Lynx Film Raw, Lynx Film | [ ] |

Note: MIT Sloan gpu partition (`sched_mit_sloan_gpu_r8`) doesn't appear to impose a max number of GPUs I can use, which makes it valuable.

─── Low Compute Resources ───────────────────────────────────────────────────────────────────────

| Resource | Allocation | Status |
|---------|--------|----------|
| MIT Normal GPU (L40s) | Low compute quick runs | [>>] |
| MIT Normal GPU (L40s) | Experimentation | [~] |
| RunPod (1x) | GPU utilization testing | [ ] |

────────────────────────────────────────────────────────────────────────────────────────────────────

┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ NOTES & WORKFLOWS                                                                               │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘

─── RunPod → Engaging Workflow ──────────────────────────────────────────────────────────────────
- RunPod jobs complete on pod, then results must be SCP-ed to engaging (output subfolder recursive)
- Storage volume contains persistent storage for all RunPod jobs (attached to all pods)
- Spin pods on community cloud for pricing and availability
- One pod per Fidel-TS dataset (run everything on that dataset on that pod)
- Logging: Use `| tee /workspace/command.log` for command output
- Save tmux session: `capture-pane -S - -E -; save-buffer /workspace/session.log` after Ctrl-B and colon, or more easily, do `tmux capture-pane -pS - > /workspace/runpod-<dataset>-session-<session>-<time>.log`. which works wonders and does not require the control b colon. just works in bash.
- SCP command: `scp -P <port> -i ~/.ssh/id_ed25519 root@<IP>:/workspace/session.log ~/Downloads/`

─── Experiment Workflow ─────────────────────────────────────────────────────────────────────────
- When running experiments (not sweeps): Init only first manually, then put requeue on preemptable with resumption-id and resume-suite-id in configs before submitting sbatch job

─── Evaluation Log Pattern ──────────────────────────────────────────────────────────────────────
- Evaluation logs should be moved from `context/logs/` to `configs/experiment_suites/<method>/<dataset>.eval`
- Example: `context/logs/itransformer_time_mmd_ttc_evaluation.log` → `configs/experiment_suites/itransformer/time_mmd_ttc.eval`

─── Known Issues ────────────────────────────────────────────────────────────────────────────────
- **TimeCMA on CAISO**: OOM issues, reduced chunk size to 50k from 500k (flag might not be working, may need to adjust actual config)
- **GPU Utilization**: Currently 3-4% on average (CPU bound, not GPU bound). Need to optimize for better utilization.

────────────────────────────────────────────────────────────────────────────────────────────────────

┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STATUS LEGEND                                                                                   │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘

- [✓] Complete
- [~] In Progress
- [>>] Running
- [ ] Pending
- [✗] Failed / Blocked
- [-] Paused
- [ERR] Error
