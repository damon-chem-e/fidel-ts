# PyTorch Lightning for Time Series Forecasting

This extension provides a PyTorch Lightning implementation of the time series forecasting pipeline, making it easier to leverage multi-GPU training, mixed precision, and other advanced training features.

## Benefits of the PyTorch Lightning Implementation

- **Multi-GPU Training**: Easily distribute training across multiple GPUs with minimal code changes.
- **Mixed Precision Training**: Reduce memory usage and speed up training with 16-bit precision.
- **Better Code Organization**: Clear separation between model logic, data processing, and training loop.
- **Built-in Features**: Access to features like gradient clipping, early stopping, and model checkpointing.
- **Experiment Tracking**: Native TensorBoard integration for experiment monitoring.

## How to Use

### Running a Training Job

The PyTorch Lightning pipeline uses the new CLI structure with YAML configuration files:

```bash
python -m cli.train lightning configs/experiments/dlinear_solar.yaml
```

### Configuration File

Create a YAML configuration file (e.g., `configs/experiments/dlinear_solar.yaml`):

```yaml
model:
  name: DLinear
  config_path: model_configs/general/DLinear.yaml

data:
  name: solar
  config_path: data_configs/fullsolar.yaml

training:
  epochs: 20
  batch_size: 96
  learning_rate: 5e-4
  patience: 3
  input_len: 96
  output_len: 96
  precision: 16  # Lightning-specific: 32, 16, or bf16
  gradient_clip_val: 0.0  # Lightning-specific: gradient clipping

device:
  use_gpu: true
  use_multi_gpu: true
  devices: "0,1,2,3"
```

### Key Configuration Parameters

- **Model Configuration**: `model.name`, `model.config_path`
- **Data Configuration**: `data.name`, `data.config_path`, `training.input_len`, `training.output_len`
- **Training Parameters**: `training.epochs`, `training.batch_size`, `training.learning_rate`, `training.patience`
- **GPU Options**: `device.use_gpu`, `device.gpu`, `device.use_multi_gpu`, `device.devices`
- **Lightning-Specific**: `training.precision`, `training.gradient_clip_val`

### Sample Multi-GPU Command

```bash
python -m cli.train lightning configs/experiments/dlinear_etth1.yaml
```

With config file containing:
```yaml
model:
  name: DLinear
  config_path: model_configs/general/DLinear.yaml

data:
  name: ETTh1
  config_path: data_configs/ETT/fullETT_H.yaml

training:
  epochs: 20
  batch_size: 256
  input_len: 96
  output_len: 96
  precision: 16

device:
  use_gpu: true
  use_multi_gpu: true
  devices: "0,1,2,3"
```

## Directory Structure

```
├── cli/
│   ├── train.py               # Training CLI commands
│   └── config/                # Configuration loading
├── runs/
│   └── lightning.py           # Lightning training execution
├── data_provider/
│   ├── data_factory.py        # Original data provider
│   ├── data_loader.py         # Original dataset loader
│   └── lightning_data_module.py # Lightning data module wrapper
├── exp/
│   ├── exp_universal.py       # Original training pipeline
│   └── exp_lightning.py       # Lightning model and training logic
└── configs/
    └── experiments/           # Experiment configuration files
```

## Implementation Details

The PyTorch Lightning implementation wraps the existing models and datasets with Lightning components:

1. **TimeSeriesDataModule**: Wraps the existing `Data_Provider` to create PyTorch Lightning-compatible data loaders
2. **TimeSeriesLightningModel**: Wraps the model implementation and training logic in a Lightning module
3. **train_lightning_model**: Orchestrates the training process with Lightning Trainer

The implementation is designed to be compatible with the existing codebase, so you can use the same models and configurations as before.

## Advanced Usage

### Mixed Precision Training

Configure in your YAML file:

```yaml
training:
  precision: 16  # Options: 32, 16, bf16
```

Then run:
```bash
python -m cli.train lightning configs/experiments/your_config.yaml
```

### Gradient Clipping

Configure in your YAML file:

```yaml
training:
  gradient_clip_val: 0.5
```

Then run:
```bash
python -m cli.train lightning configs/experiments/your_config.yaml
```

### Changing Early Stopping Criteria

Modify the `EarlyStopping` callback in `exp_lightning.py` to change the early stopping criteria.

### Custom Learning Rate Schedules

The implementation already supports the existing learning rate adjustment strategies. For custom schedules, modify the `configure_optimizers` method in the `TimeSeriesLightningModel` class. 