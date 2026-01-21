# TGTSF Experiment Suites

This directory contains TGTSF experiment configurations.

## Full Hetero Resolution Configs

**All configs in this directory use full hetero resolution** (`text_embedding_stride: "full"`), which results in `hetero_stride=1` (full resolution text embeddings at every timestep).

This setting:
- Matches the tensor cache format generated with full resolution embeddings
- Uses more memory and computation than the original paper's strided approach
- Provides maximum temporal resolution for hetero data

## Original Resolution Configs

For configs matching the original TGTSF paper's hetero stride behavior (aligned with model's patch stride, typically `hetero_stride=3`), see the [`original_resolution/`](./original_resolution/) subdirectory.

The original resolution configs:
- Omit `text_embedding_stride`, defaulting to model-aligned stride
- Use less memory and computation
- Match the original TGTSF paper's implementation

## Usage

```bash
# Run full resolution configs (this directory)
python -m cli.suite run configs/experiment_suites/tgtsf/california_iso.yaml

# Run original resolution configs
python -m cli.suite run configs/experiment_suites/tgtsf/original_resolution/california_iso.yaml
```
