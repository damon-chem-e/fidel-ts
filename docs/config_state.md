# Config state and mapping

This document maps **all configuration files** in this repository, explains the **nested structure / composition flow**, and calls out **known issues** (including the `model_config` naming collision risk with Pydantic v2).

## What “config” means in this repo

There are **four config domains**:

- **`configs/`**: “orchestration configs” (templates, single-run experiment configs, and experiment suites).
- **`model_configs/`**: model-specific hyperparameter YAMLs loaded at runtime via `model.config_path`.
- **`data_configs/`**: dataset-specific YAMLs loaded at runtime via `data.config_path`.
- **JSON “config-like” files** used for sampling / metadata:
  - `sample_indexes/*.json`
  - `data/data_structure.json`

There is **no Hydra/OmegaConf composition** here (the code path is YAML + Pydantic + custom deep-merge).

## High-level nesting / composition flow

### Direct experiment run (`cli.train ...`)

1. `cli.train` calls `cli.config.loader.load_config()`.
2. `ExperimentConfig.from_yaml()` parses an experiment YAML (under `configs/experiments/`) into a **Pydantic** model (`cli/config/models.py`).
3. Runtime then loads:
   - the **model YAML** at `config.model.config_path` (e.g. `model_configs/general/TGTSF.yaml`)
   - the **data YAML** at `config.data.config_path` (e.g. `data_configs/California_ISO/fullCAISO_hetero_TGTSF_H.yaml`)
4. Runtime optionally applies “overrides” that are **embedded inside the ExperimentConfig** (details below).

### Suite run (`cli.suite run ...`)

Suites add another layer:

1. `runs/suite_executor.py` loads:
   - a suite YAML (`configs/experiment_suites/*.yaml`)
   - a template YAML (`configs/templates/*.yaml`)
2. For each experiment entry in the suite:
   - template YAML is loaded
   - suite experiment’s `overrides:` are **deep-merged** into the template via `utils.config_utils.merge_configs()`
   - placeholders (e.g. `${experiment_name}`) are substituted recursively
   - the merged dict is parsed into `ExperimentConfig(**config)` (Pydantic)
3. Runtime then loads model/data YAMLs from `model.config_path` and `data.config_path` and may merge additional overrides.

### Deep merge semantics (important)

`utils/config_utils.merge_configs()` performs a **recursive dict merge**, with “right-hand side wins”.

- **dict + dict**: merged recursively
- **list + list**: the override list fully replaces the template list (no item-wise merge)
- **scalar**: overridden

This matters when overrides touch nested sections like `data_config.hetero_info` (works) vs lists (replaced).

## Canonical schemas (what keys belong where)

### `ExperimentConfig` (Pydantic) schema

Defined in `cli/config/models.py`. The top-level typed fields are:

- **`model`**: `{ name, config_path }` (extra forbidden)
- **`data`**: `{ name, config_path }` (extra forbidden)
- **`training`**: many fields (extra allowed)
- **`device`**: `{ use_gpu, gpu, use_multi_gpu, devices }` (extra forbidden)
- **`wandb`**: `{ project, entity, run_name, run_id, tags, notes, enabled, mode }` (extra forbidden)
- **top-level optional**:
  - `hf_mirror`, `hf_offline`, `base_data_path`
  - `plotting`, `evaluation` (paths to nested YAMLs; see loader)
  - `model_config_overrides` (dict of model-yaml key overrides)
  - `experiment_name`, `job_id`, `job_name`, `random_seed`
  - `resume_experiment_id`, `resume_suite_id`, `mark_last_job_complete`

Also: `ExperimentConfig` sets **`extra="allow"`**, so suite templates can include **non-modeled** sections like `experiment:` (details below).

### Suite schema (`configs/experiment_suites/*.yaml`)

Top-level:

- **`suite`**:
  - `name`, `description`, `tags`
  - `execution`: `parallel`, `continue_on_error`, `log_dir`, and optional suite-level resume fields
  - `experiments`: list of experiments

Each experiment entry:

- `name`, `description`, `enabled`
- `template`: path into `configs/templates/*.yaml`
- `overrides`: a dict that is deep-merged into the template. Common keys:
  - `model`, `data`, `training`, `device`, `wandb`
  - `model_config_overrides` (top-level)
  - `data_config` (top-level) — see “issues” for framework differences
  - `resume_experiment_id` (when resuming suite runs)

Special behavior:

- If `overrides.training.output_lens` exists, `SuiteExecutor` fans out the experiment into multiple runs (and edits `training.output_len` + the experiment name accordingly).

### Template schema (`configs/templates/*.yaml`)

Templates are “base experiment dicts” used by suites. They typically contain:

- **`experiment`**: `{ name, type, ... }` (this is *not* modeled by Pydantic; it’s accepted via `extra="allow"`)
  - `type` selects runner: `pytorch`, `lightning`, `llm`, `fm`, `evaluation`
- **`model`**, **`data`**, **`training`**, **`device`**, **`wandb`**
- placeholder strings like `${experiment_name}`, `${model_config_path}`, etc.

### Model config YAML schema (`model_configs/**.yaml`)

These YAMLs are **not** validated by Pydantic today; they’re loaded as raw dicts and wrapped into `dotdict` at runtime (see `runs/pytorch.py`, `runs/lightning.py`).

Examples:

- `model_configs/general/TGTSF.yaml` defines keys like `enc_in`, `d_model`, `text_dim`, `patch_len`, `stride`, `task`, etc.
- `model_configs/general/lynx.yaml` extends TGTSF-like keys plus pretrained checkpoint fields:
  - `pretrained_model_path`, `pretrained_model_type`, `pretrained_model_config_path`, `pretrained_checkpoint_base`, etc.
- `model_configs/LLM/**.yaml` holds LLM inference parameters and prompt template paths.

### Data config YAML schema (`data_configs/**.yaml`)

These YAMLs are also loaded as raw dicts and wrapped into `dotdict`.

Common fields:

- `root_path`, `data_path` (for CSV) or `formatter` (for parquet patterns)
- `spliter` and `split_info` (ratio/timestamp split)
- `timestamp_col`, `target`, `id`, `id_info`
- `downsample`, `sampling_rate`, `base_T`

Heterogeneous / text augmentation fields appear in some datasets:

- `hetero_info`: includes a nested structure with keys like:
  - `hetero_type`, `root_path`, `formatter`, `matching`
  - `input_format` (e.g. `embedding`)
  - `static_path`
- `timemmd_*` fields for Time-MMD (e.g. `timemmd_text_output`, embedding model/dim, caching)

## Full file mapping (by directory)

### `configs/`

#### `configs/templates/` (5 files)

- **`evaluation.yaml`**: base template for evaluation runs (note: evaluation runner uses a separate schema at runtime).
- **`fm_testing.yaml`**: base template for foundation model tests.
- **`lightning_training.yaml`**: base template for Lightning training runs.
- **`llm_testing.yaml`**: base template for LLM inference/testing.
- **`pytorch_training.yaml`**: base template for standard PyTorch training runs.

#### `configs/experiments/` (3 files)

These are “single-run” experiment YAMLs meant to be invoked directly via `cli.train`.

- **`tgtsf_test_1epoch.yaml`**
- **`time_mmd_test.yaml`**
- **`time_mmd_test_tgtsf.yaml`**

#### `configs/experiment_suites/` (23 files)

Suite YAMLs that enumerate many experiments and deep-merge overrides into templates.

- **`ablation_studies.yaml`**
- **`chattime.yaml`**
- **`embeddings_test.yaml`**
- **`filtered_samples.yaml`**
- **`foundation_models.yaml`**
- **`gpt4mts_rpllm.yaml`**
- **`gpt4ts.yaml`**
- **`itransformer_pretraining.yaml`**
- **`itransformer_pretraining_test.yaml`**
- **`linear_models.yaml`**
- **`llm_qwen3_14b.yaml`**
- **`lynx_film_raw_test.yaml`**
- **`lynx_film_test.yaml`**
- **`lynx_test.yaml`**
- **`lynx_training.yaml`**
- **`test_evaluation.yaml`**
- **`tgtsf_iatsf.yaml`**
- **`tgtsf_test.yaml`**
- **`time_mmd_test.yaml`**
- **`transformer_models.yaml`**
- **`ttc_test.yaml`**
- **`zero_shot_evaluation.yaml`**
- **`zhanghanbest_test.yaml`**

### `model_configs/` (29 files)

#### `model_configs/general/` (15 files + 2 nested subfolders)

- **`DLinear.yaml`**
- **`FITS.yaml`**
- **`GPT4MTS.yaml`**
- **`GPT4TS.yaml`**
- **`PatchTST.yaml`**
- **`PatchTST/PatchTST-Bear.yaml`**
- **`PatchTST/PatchTST-CAISO.yaml`**
- **`TGTSF.yaml`**
- **`TGTSF/TGTSF-Bear.yaml`**
- **`TGTSF/TGTSF-CAISO.yaml`**
- **`ZhangHanBest.yaml`**
- **`iTransformer.yaml`**
- **`lynx.yaml`**
- **`lynx_film.yaml`**
- **`lynx_film_raw.yaml`**

#### `model_configs/FM/` (4 files)

- **`ChatTime.yaml`**
- **`Chronos.yaml`**
- **`Sundial.yaml`**
- **`TimeMoE.yaml`**

#### `model_configs/LLM/` (10 files)

- **`LLM/MultiModal/DeepSeek-R1.yaml`**
- **`LLM/MultiModal/Qwen2.5-14B-Instruct-1m.yaml`**
- **`LLM/MultiModal/Qwen2.5-14B-Instruct.yaml`**
- **`LLM/MultiModal/Qwen3-14B.yaml`**
- **`LLM/MultiModal/Time-R1.yaml`**
- **`LLM/UniModal/DeepSeek-R1.yaml`**
- **`LLM/UniModal/Qwen2.5-14B-Instruct-1m.yaml`**
- **`LLM/UniModal/Qwen2.5-14B-Instruct.yaml`**
- **`LLM/UniModal/Qwen3-14B.yaml`**
- **`LLM/UniModal/Time-R1.yaml`**

### `data_configs/` (57 files)

Organized by dataset name. Notable subfamilies:

- Many datasets have a base `full*.yaml` plus variants:
  - `*_H.yaml` (often indicating a sampling rate / horizon variant)
  - `*_hetero_*.yaml` (heterogeneous data augmentation enabled)
  - `*_zero_shot_*.yaml` or `*_ablation_*.yaml` (evaluation regimes)

#### `data_configs/time_mmd/*/config.yaml` (9 files)

- **`time_mmd/Algriculture/config.yaml`**
- **`time_mmd/Climate/config.yaml`**
- **`time_mmd/Economy/config.yaml`**
- **`time_mmd/Energy/config.yaml`**
- **`time_mmd/Environment/config.yaml`**
- **`time_mmd/Public_Health/config.yaml`**
- **`time_mmd/Security/config.yaml`**
- **`time_mmd/SocialGood/config.yaml`**
- **`time_mmd/Traffic/config.yaml`**

#### `data_configs/ttc/*/config.yaml` (2 files)

- **`ttc/climate/config.yaml`**
- **`ttc/medical/config.yaml`**

#### Other dataset folders (selected; full list is in the repo tree)

- **`Bear_room/`**: 9 files (base + hetero + ablation + zero-shot variants)
- **`California_ISO/`**: 6 files (base + hetero variants)
- **`Canada_photovoltaics_plants/`**: 4 files
- **`ETT/`**: 2 files
- **`Germany_Renewable_Power_Grid/`**: 6 files
- **`Jena_Atmospheric_Physics/`**: 6 files
- **`NYC_traffic_speed/`**: 9 files
- **`electricity/`**: 1 file
- **`traffic/`**: 1 file
- **`weather/`**: 2 files

## Known issues / sharp edges in config structure and flow

### `model_config` naming collision (Pydantic v2 reserved name)

**Why this is dangerous**

In Pydantic v2, `model_config` is a **reserved class attribute** used to configure model behavior (e.g. `ConfigDict(extra=...)`). This repo correctly uses it in its Pydantic models:

- `cli/config/models.py` sets `model_config = ConfigDict(...)` on several BaseModels.

Historically, there was legacy runtime support for reading a user-provided YAML key named `model_config` as “model YAML overrides”.
This repo now **hard-forbids** `model_config` in `ExperimentConfig` input and runners only accept `model_config_overrides`.

Even if current YAML files do not contain a `model_config:` key, keeping this compatibility hook is risky because it encourages a config key name that conflicts with Pydantic internals and can lead to:

- confusing behavior (is `model_config` a Pydantic config or user overrides?)
- serialization / `model_dump()` surprises
- accidental shadowing / hard-to-debug attribute access

**What to use instead**

Use **`model_config_overrides`** (a first-class `ExperimentConfig` field) for model-yaml overrides.

### `data_config` overrides are inconsistent across runners

- `runs/pytorch.py` merges top-level `data_config` overrides (from `config.model_dump()`) into the loaded data YAML.
- `runs/lightning.py` **does not merge** `data_config` overrides at all (it only loads the data YAML and applies `base_data_path` replacement).

This means suite entries that rely on `overrides.data_config` may behave differently depending on whether `experiment.type` is `pytorch` or `lightning`.

### `training.random_seed` in templates likely does not affect seeding

Templates set `training.random_seed`, but runtime seeds from **`config.random_seed`** (top-level field of `ExperimentConfig`), not `config.training.random_seed`.

Because `TrainingConfig` has `extra="allow"`, `training.random_seed` will be silently accepted but not necessarily used, which is a classic “looks right, does nothing” config pitfall.

### Relative path resolution depends on current working directory

- `cli.config.loader.resolve_config_path()` resolves relative paths against `Path.cwd()`.
- `runs.suite_executor.load_template()` calls `resolve_config_path(template_path)` without passing `base_dir`.

If commands are run from a directory other than repo root, many relative config paths can break (or worse, resolve to unintended locations).

### Two different “schemas” exist (Pydantic vs dotdict evaluation path)

Suite execution has a special case for `experiment.type == 'evaluation'` where it bypasses `ExperimentConfig` and uses a dotdict-based schema.

That increases the chance of config drift and inconsistent validation across experiment types.

### Typos can silently pass due to `extra="allow"` in key models

Both `TrainingConfig` and `ExperimentConfig` allow extra fields. That is useful for experimentation, but it also means:

- typos like `trainig:` or `batc_size:` won’t be caught early
- suite templates can accumulate unused keys

## Migration status: ported away from the `model_config` issue

The migration is now complete in core code paths:

- `ExperimentConfig` rejects any input containing top-level `model_config`.
- Runners merge only `model_config_overrides` into the loaded model YAML.
- A regression test enforces that repo YAML configs do not contain `model_config:`.


