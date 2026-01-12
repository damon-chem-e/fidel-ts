"""
Lightning evaluation module for PyTorch Lightning-trained models.

This module provides evaluation functionality for models trained with PyTorch Lightning,
migrated from criterias_lightning.py.
"""

import torch
import os
import json
import glob
import yaml
from tqdm import tqdm
from models import model_init
from data_provider.data_factory import Data_Provider
from utils.tools import dotdict


def run_test(loader, model, config, device, indexes, channel_wise):
    """
    Run evaluation test on a data loader.
    
    Args:
        loader: DataLoader for test dataset
        model: Trained model
        config: Configuration object
        device: PyTorch device
        indexes: List of sample indexes to evaluate (None for all)
        channel_wise: Whether to compute channel-wise metrics
    
    Returns:
        If channel_wise: (channel_mse, channel_mae, channel_counts)
        Otherwise: (total_mse, total_mae, num_samples)
    """
    total_mse, total_mae = 0.0, 0.0
    num_samples = 0

    channel_mse = None
    channel_mae = None
    channel_counts = None

    for i, iter_data in tqdm(enumerate(loader), total=len(loader), desc="Running tests"):
        if indexes is not None and i not in indexes:
            continue
        with torch.no_grad():
            batch_x, batch_y, _, _, _, y_hetero, _, _, _, hetero_channel = iter_data

            batch_x = torch.tensor(batch_x).to(device)
            batch_y = torch.tensor(batch_y).to(device)
            y_hetero = torch.tensor(y_hetero).to(device)
            hetero_channel = torch.tensor(hetero_channel).to(device)

            prediction = model(x=batch_x) if config.task == 'TSF' else model(x=batch_x, news=y_hetero, channel_description=hetero_channel)
            prediction = prediction[:, -config.output_len:, :]  # [B, L, C]

            if channel_wise:
                if channel_mse is None:
                    C = prediction.shape[2]
                    channel_mse = [0.0] * C
                    channel_mae = [0.0] * C
                    channel_counts = [0] * C
                for k in range(prediction.shape[2]):
                    mse_loss = torch.nn.MSELoss()(prediction[:, :, k], batch_y[:, :, k])
                    mae_loss = torch.nn.L1Loss()(prediction[:, :, k], batch_y[:, :, k])
                    channel_mse[k] += mse_loss.item() * batch_y.size(0)
                    channel_mae[k] += mae_loss.item() * batch_y.size(0)
                    channel_counts[k] += batch_y.size(0)

            else:
                mse_loss = torch.nn.MSELoss()(prediction, batch_y)
                mae_loss = torch.nn.L1Loss()(prediction, batch_y)
                total_mae += mae_loss.item() * batch_y.size(0)
                total_mse += mse_loss.item() * batch_y.size(0)
                num_samples += batch_y.size(0)

    if channel_wise:
        return channel_mse, channel_mae, channel_counts
    
    else:
        return total_mse, total_mae, num_samples


def find_checkpoint(eval_config):
    """
    Find Lightning checkpoint path based on evaluation configuration.
    
    This function supports both new format (experiment directory structure) and
    legacy format (pattern matching) for backward compatibility.
    
    Args:
        eval_config: Evaluation configuration with checkpoint_base (experiment dir) or checkpoint_base (parent dir)
    
    Returns:
        Path to checkpoint directory (for backward compatibility) or checkpoint file (for new format)
    """
    from pathlib import Path
    from evaluation.checkpoint_finder import find_checkpoint_from_experiment_dir, find_checkpoint_legacy
    
    checkpoint_base = Path(eval_config.checkpoint_base)
    
    # Check if checkpoint_base points directly to an experiment directory (new format)
    # Experiment directories have a 'checkpoints' subdirectory
    if (checkpoint_base / "checkpoints").exists():
        # New format: use experiment directory structure
        if hasattr(eval_config, 'experiment_dir'):
            experiment_dir = Path(eval_config.experiment_dir)
        else:
            experiment_dir = checkpoint_base
        
        # Find checkpoint file (not directory)
        ckpt_file = find_checkpoint_from_experiment_dir(experiment_dir, version=eval_config.version)
        # Return the parent directory (checkpoints dir) for compatibility with existing code
        return ckpt_file.parent
    else:
        # Legacy format: use pattern matching (backward compatibility)
        import warnings
        warnings.warn(
            "Using legacy checkpoint finding with pattern matching. "
            "Consider migrating to experiment directory structure with resume_experiment_id.",
            DeprecationWarning,
            stacklevel=2
        )
        # Legacy Lightning pattern matching
        ckpt_id = f'_{eval_config.model}_{eval_config.data}_{eval_config.output_len}_{eval_config.input_len}_pl'
        
        if eval_config.version == 'latest':
            ckpt_paths = [
                os.path.join(eval_config.checkpoint_base, i) 
                for i in os.listdir(eval_config.checkpoint_base) 
                if ckpt_id in i and os.path.isdir(os.path.join(eval_config.checkpoint_base, i))
            ]
            if not ckpt_paths:
                raise FileNotFoundError(f"No checkpoint found for pattern: *{ckpt_id}")
            ckpt_paths.sort()
            ckpt_path = ckpt_paths[-1]
        elif eval_config.version == 'oldest':
            ckpt_paths = [
                os.path.join(eval_config.checkpoint_base, i) 
                for i in os.listdir(eval_config.checkpoint_base) 
                if ckpt_id in i and os.path.isdir(os.path.join(eval_config.checkpoint_base, i))
            ]
            if not ckpt_paths:
                raise FileNotFoundError(f"No checkpoint found for pattern: *{ckpt_id}")
            ckpt_paths.sort()
            ckpt_path = ckpt_paths[0]
        else:
            pattern = os.path.join(eval_config.checkpoint_base, eval_config.version + ckpt_id)
            ckpt_paths = [p for p in glob.glob(pattern) if os.path.isdir(p)]
            if not ckpt_paths:
                raise FileNotFoundError(f"No checkpoint found for pattern: {pattern}")
            ckpt_paths.sort()
            ckpt_path = ckpt_paths[-1]
        
        return ckpt_path


def evaluate(config):
    """
    Evaluate PyTorch Lightning-trained models.
    
    This function loads a Lightning checkpoint, runs inference on test datasets,
    and calculates evaluation metrics. Handles Lightning checkpoint format.
    
    Args:
        config: dotdict containing evaluation configuration with structure:
            - evaluation: {model, data, version, input_len, output_len, checkpoint_base,
                          batch_size, task, filtered_samples, device, channel_wise}
            - data_config: Optional path to override data config
    
    Example:
        >>> from cli.config.loader import load_config_with_nested
        >>> config_hierarchy = load_config_with_nested("configs/experiments/dlinear_solar.yaml")
        >>> evaluate(config_hierarchy['primary'])
    """
    eval_config = config.evaluation
    
    # Determine experiment directory and load checkpoint config
    from pathlib import Path
    from evaluation.checkpoint_finder import find_checkpoint_from_experiment_dir
    from evaluation.config_builder import load_checkpoint_config
    
    # Check if using new format (experiment directory structure)
    if hasattr(eval_config, 'experiment_dir'):
        experiment_dir = Path(eval_config.experiment_dir)
    elif (Path(eval_config.checkpoint_base) / "checkpoints").exists():
        experiment_dir = Path(eval_config.checkpoint_base)
    else:
        # Legacy format: use old find_checkpoint logic
        ckpt_path = find_checkpoint(eval_config)
        experiment_dir = Path(ckpt_path)
        print(f'[Info] Using checkpoint path (legacy): {ckpt_path}')
    
    # Load configuration from checkpoint directory
    print(f"[Info] Loading config from experiment directory: {experiment_dir}")
    checkpoint_config_dict = load_checkpoint_config(experiment_dir)
    
    # Convert to dotdict and handle both new and legacy formats
    if 'training' in checkpoint_config_dict:
        # New format: experiment_config.yaml
        # Load saved model and data configs from experiment directory
        model_config_path = experiment_dir / "configs" / "model_config.yaml"
        data_config_path = experiment_dir / "configs" / "data_config.yaml"
        
        checkpoint_config = dotdict()
        checkpoint_config.model = checkpoint_config_dict.get('model', {}).get('name', 'unknown')
        checkpoint_config.data = checkpoint_config_dict.get('data', {}).get('name', 'unknown')
        training_config = checkpoint_config_dict.get('training', {})
        checkpoint_config.input_len = training_config.get('input_len')
        checkpoint_config.output_len = training_config.get('output_len')
        checkpoint_config.batch_size = training_config.get('batch_size', 128)
        
        # Load model config
        if model_config_path.exists():
            with open(model_config_path, 'r', encoding='utf-8') as f:
                checkpoint_config.model_config = dotdict(yaml.safe_load(f))
        else:
            # Fallback: try to load from original path
            model_cfg_path = checkpoint_config_dict.get('model', {}).get('config_path')
            if model_cfg_path:
                with open(model_cfg_path, 'r', encoding='utf-8') as f:
                    checkpoint_config.model_config = dotdict(yaml.safe_load(f))
            else:
                raise FileNotFoundError(f"Model config not found: {model_config_path}")
        
        # CRITICAL: Apply model_config_overrides from experiment config
        # These overrides (e.g., d_model, e_layers, n_heads) were used during training
        # and must be applied to ensure the model architecture matches the checkpoint
        model_config_overrides = checkpoint_config_dict.get('model_config_overrides', {})
        if model_config_overrides:
            print(f"[Info] Applying model_config_overrides: {model_config_overrides}")
            for key, value in model_config_overrides.items():
                checkpoint_config.model_config[key] = value
        
        # Load data config
        if data_config_path.exists():
            with open(data_config_path, 'r', encoding='utf-8') as f:
                checkpoint_config.data_config = dotdict(yaml.safe_load(f))
        else:
            # Fallback: try to load from original path
            data_cfg_path = checkpoint_config_dict.get('data', {}).get('config_path')
            if data_cfg_path:
                with open(data_cfg_path, 'r', encoding='utf-8') as f:
                    checkpoint_config.data_config = dotdict(yaml.safe_load(f))
            else:
                raise FileNotFoundError(f"Data config not found: {data_config_path}")
    else:
        # Legacy format: args.json (flat structure)
        # In legacy format, model_config and data_config are already dictionaries
        checkpoint_config = dotdict(checkpoint_config_dict)
        if isinstance(checkpoint_config.model_config, dict):
            checkpoint_config.model_config = dotdict(checkpoint_config.model_config)
        elif isinstance(checkpoint_config.model_config, str):
            # If it's a path string, load it
            with open(checkpoint_config.model_config, 'r', encoding='utf-8') as f:
                checkpoint_config.model_config = dotdict(yaml.safe_load(f))
        
        if isinstance(checkpoint_config.data_config, dict):
            checkpoint_config.data_config = dotdict(checkpoint_config.data_config)
        elif isinstance(checkpoint_config.data_config, str):
            # If it's a path string, load it
            with open(checkpoint_config.data_config, 'r', encoding='utf-8') as f:
                checkpoint_config.data_config = dotdict(yaml.safe_load(f))
        else:
            checkpoint_config.data_config = dotdict({})
    
    # Use provided data_config override if available
    if hasattr(config, 'data_config') and config.data_config:
        checkpoint_config.data_config = dotdict(yaml.safe_load(open(config.data_config, 'r')))
    elif not hasattr(checkpoint_config, 'data_config'):
        checkpoint_config.data_config = dotdict({})
    
    checkpoint_config.devices = eval_config.device
    checkpoint_config.num_workers = 0
    checkpoint_config.batch_size = 1 if eval_config.filtered_samples is not None else eval_config.batch_size
    checkpoint_config.task = eval_config.task

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{eval_config.device}")
    else:
        device = torch.device("cpu")
        print("[Warning] CUDA is not available, use CPU instead.")
    print(f"[Info] Running on device: {device}")

    model = model_init(checkpoint_config.model, checkpoint_config.model_config, checkpoint_config).to(device)
    
    # Find checkpoint file
    if hasattr(eval_config, 'experiment_dir') or (Path(eval_config.checkpoint_base) / "checkpoints").exists():
        # New format: use experiment directory structure
        ckpt_file_path = find_checkpoint_from_experiment_dir(experiment_dir, version=eval_config.version)
    else:
        # Legacy format: look in checkpoint directory
        ckpt_file = glob.glob(os.path.join(str(experiment_dir), 'checkpoint*'))
        if not ckpt_file:
            raise FileNotFoundError(f"Checkpoint file not found in {experiment_dir}")
        ckpt_file_path = Path(ckpt_file[0])
    
    print(f"[Info] Loading model from: {ckpt_file_path}")
    checkpoint = torch.load(str(ckpt_file_path), map_location=device)

    if str(ckpt_file_path).endswith('.ckpt'):
        if eval_config.model == "PatchTST":
            state_dict = {key.replace("model.model.", "model."): value for key, value in checkpoint['state_dict'].items()} 
        else:
            state_dict = {key.replace("model.", ""): value for key, value in checkpoint['state_dict'].items()} 
    else:
        state_dict = checkpoint

    # Handle torch.compile checkpoints (state dict keys have "_orig_mod." prefix)
    # Check if this is a compiled model checkpoint
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        print("[Info] Detected torch.compile checkpoint - stripping '_orig_mod.' prefix from state dict keys")
        state_dict = {key.replace('_orig_mod.', ''): value for key in state_dict.keys() for value in [state_dict[key]]}

    model.load_state_dict(state_dict)
    model.eval()

    print(f'[Info] Successfully loaded model: {checkpoint_config.model}')
    
    # Apply torch.compile if enabled in evaluation config (PyTorch 2.0+)
    # Check for torch_compile flag in eval_config (backward compatible - defaults to False)
    if getattr(eval_config, 'torch_compile', False):
        if hasattr(torch, 'compile'):
            compile_mode = getattr(eval_config, 'compile_mode', 'reduce-overhead')
            print(f"[Info] Compiling model for evaluation with torch.compile (mode={compile_mode})...")
            model = torch.compile(
                model,
                mode=compile_mode,
                fullgraph=False  # More compatible with dynamic models
            )
            # Warm-up compilation with a dummy forward pass
            print("[Info] Warming up torch.compile...")
            try:
                with torch.no_grad():
                    dummy_x = torch.randn(1, checkpoint_config.input_len, checkpoint_config.model_config.enc_in).to(device)
                    _ = model(x=dummy_x)
                print("[Info] Compilation warm-up complete")
            except Exception as e:
                print(f"[Warning] Warm-up forward pass failed (model will compile on first real batch): {e}")
        else:
            print("[Warning] torch.compile requested but not available (requires PyTorch 2.0+)")

    # Prepare datasets
    id_data = Data_Provider(checkpoint_config)
    fullloader = id_data.get_test('loader')
    print(f'[Info] Found {len(fullloader)} datasets to test: {list(fullloader.keys())}')

    # Handle filtered samples if provided
    if eval_config.filtered_samples is not None:
        filtered_samples = json.load(open(eval_config.filtered_samples))
        print(f"[Info] Using filtered samples from: {eval_config.filtered_samples}")

    all_mae = 0.0 if not eval_config.channel_wise else {}
    all_mse = 0.0 if not eval_config.channel_wise else {}
    all_sample_num = 0 if not eval_config.channel_wise else {}

    for name, loader in fullloader.items():
        print(f"\n[Info] Testing on dataset: {name}")

        if eval_config.filtered_samples is not None:
            indexes = filtered_samples[name]
            print(f"[Info] Using {len(indexes)} filtered samples for testing.")
            print(f"[Info] Sample indexes: {indexes}")
        else:
            indexes = None
            print("[Info] Using all samples for testing.")
        
        result = run_test(loader, model, checkpoint_config, device, indexes, eval_config.channel_wise)
        if eval_config.channel_wise:
            channel_mse, channel_mae, channel_counts = result
            if sum(channel_counts) == 0:
                print(f"-> No index found in '{name}'")
            else:
                all_mse[name] = channel_mse
                all_mae[name] = channel_mae
                all_sample_num[name] = channel_counts
                avg_ch_mse = [m / count if count > 0 else 0 for m, count in zip(channel_mse, channel_counts)]
                avg_ch_mae = [m / count if count > 0 else 0 for m, count in zip(channel_mae, channel_counts)]
                print(f"-> Results for '{name}': Channel-wise MSE = {avg_ch_mse}, MAE = {avg_ch_mae}")
                print(f"-> Results for '{name}': All channel MSE = {sum(avg_ch_mse) / len(avg_ch_mse):.7f}, MAE = {sum(avg_ch_mae) / len(avg_ch_mae):.7f}")
        
        else:
            total_mse, total_mae, num_samples = result
            if num_samples > 0:
                all_mse += total_mse
                all_mae += total_mae
                all_sample_num += num_samples
                avg_mse = total_mse / num_samples
                avg_mae = total_mae / num_samples
                print(f"-> Results for '{name}': MSE = {avg_mse:.7f}, MAE = {avg_mae:.7f}")
            else:
                print(f"-> No index found in '{name}'")
            
    print("\n" + "="*50)
    print(" " * 15 + "Overall Test Summary")
    if eval_config.channel_wise:
        sum_mse = [sum(m) for m in zip(*all_mse.values())]
        sum_mae = [sum(m) for m in zip(*all_mae.values())]
        sum_counts = [sum(c) for c in zip(*all_sample_num.values())]
        overall_mse = [m / c if c > 0 else 0 for m, c in zip(sum_mse, sum_counts)]
        overall_mae = [m / c if c > 0 else 0 for m, c in zip(sum_mae, sum_counts)]
        print(f"-> Results for all subsets: channel-wise MSE = {overall_mse}, channel-wise MAE = {overall_mae}")
        print(f"-> Results for all subsets: All channel MSE = {sum(overall_mse) / len(overall_mse):.7f}, MAE = {sum(overall_mae) / len(overall_mae):.7f}")
    else:
        print(f"-> Results for all subsets: MSE = {all_mse / all_sample_num:.7f}, MAE = {all_mae / all_sample_num:.7f}")
    print("="*50)

