import numpy as np
import torch
import matplotlib.pyplot as plt
import time

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args, return_rate=False):
    """
    Adjust the learning rate during training based on the specified strategy.
    
    This function implements various learning rate adjustment strategies to improve
    training convergence and model performance. It supports multiple predefined
    strategies and can either modify the optimizer directly or return a rate
    multiplier for use with PyTorch Lightning.
    
    Args:
        optimizer (torch.optim.Optimizer): The optimizer whose learning rate will be adjusted.
        epoch (int): Current training epoch number.
        args (object): Arguments object containing learning rate configuration with attributes:
            - learning_rate (float): Initial learning rate
            - lradj (str): Learning rate adjustment strategy ('type1', 'type2', 'type3', 'type4', 'constant')
        return_rate (bool, optional): If True, returns the rate multiplier instead of
                                    modifying the optimizer directly. Useful for PyTorch Lightning.
                                    Defaults to False.
    
    Returns:
        float or None: If return_rate is True, returns the learning rate multiplier.
                      Otherwise, returns None and modifies the optimizer directly.
    
    Raises:
        ValueError: If the specified learning rate adjustment strategy is not supported.
    
    Strategies:
        - 'type1': Halve the learning rate every epoch
        - 'type2': Predefined schedule with specific rates at certain epochs
        - 'type3': Constant for first 3 epochs, then decay by 0.9 each epoch
        - 'type4': Decay by 0.9 every epoch from the start
        - 'constant': Keep learning rate constant throughout training
    """
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.50 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'type4':
        lr_adjust = {epoch: args.learning_rate * (0.9 ** ((epoch - 1) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    else:
        raise ValueError('Learning rate adjustment strategy not supported')
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        # If a multiplier rate is needed for PyTorch Lightning's LambdaLR
        if return_rate:
            return lr / args.learning_rate
        # Otherwise, adjust optimizer directly
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))
    else:
        # If we're just returning the rate for PyTorch Lightning
        if return_rate:
            return None if epoch not in lr_adjust.keys() else lr_adjust[epoch] / args.learning_rate


class EarlyStopping:
    """
    Early stopping utility to prevent overfitting during model training.

    This class monitors validation loss and stops training when the loss stops
    improving for a specified number of epochs (patience). It also handles
    model checkpointing by saving the best model encountered during training.

    Attributes:
        patience (int): Number of epochs to wait without improvement before stopping
        verbose (bool): Whether to print messages about validation loss improvements
        counter (int): Current count of epochs without improvement
        best_score (float): Best validation score encountered so far
        early_stop (bool): Flag indicating whether to stop training
        val_loss_min (float): Minimum validation loss encountered
        delta (float): Minimum change required to qualify as an improvement
        best_epoch (int): Epoch number when the best checkpoint was saved (1-indexed)
    """
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Initialize the EarlyStopping monitor.

        Args:
            patience (int, optional): Number of epochs to wait for improvement before
                                    stopping training. Defaults to 7.
            verbose (bool, optional): If True, prints messages when validation loss
                                    improves and model is saved. Defaults to False.
            delta (float, optional): Minimum change in validation loss to qualify as
                                   an improvement. Must be positive. Defaults to 0.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.best_epoch = None
        self.best_train_loss = None

    def __call__(self, val_loss, model, path, epoch=None, train_loss=None):
        """
        Check if training should stop based on validation loss and save model if improved.

        This method is called after each epoch to evaluate whether training should
        continue. It compares the current validation loss with the best seen so far
        and updates counters accordingly.

        Args:
            val_loss (float): Current epoch's validation loss.
            model (torch.nn.Module): The model to be saved if improvement is detected.
            path (str): Directory path where the model checkpoint should be saved.
            epoch (int, optional): Current epoch number (1-indexed). If provided,
                                   best_epoch will be tracked when checkpoint is saved.
            train_loss (float, optional): Current epoch's training loss. If provided,
                                         will be saved in checkpoint for consistency.

        Side Effects:
            - Updates self.early_stop flag if patience is exceeded
            - Saves model checkpoint if validation loss improves
            - Updates internal counters and best score tracking
            - Updates self.best_epoch and self.best_train_loss when checkpoint is saved
        """
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, epoch, train_loss)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, epoch, train_loss)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, epoch=None, train_loss=None):
        """
        Save the model checkpoint when validation loss improves.

        This method saves a checkpoint dictionary containing the model's state
        dictionary along with training metrics (train_loss, val_loss, epoch).
        Handles torch.compile by saving the underlying model's state_dict
        (without _orig_mod prefix).

        Args:
            val_loss (float): Current validation loss that represents an improvement.
            model (torch.nn.Module): The model whose state dict will be saved.
            path (str): Directory path where the checkpoint file will be saved.
                       The file will be saved as 'checkpoint.pth' in this directory.
            epoch (int, optional): Current epoch number (1-indexed). If provided,
                                   updates self.best_epoch to track when the best
                                   checkpoint was saved.
            train_loss (float, optional): Current training loss. If provided,
                                         will be saved in checkpoint and tracked.

        Side Effects:
            - Saves checkpoint dict to '{path}/checkpoint.pth' containing:
              model_state_dict, train_loss, val_loss, epoch
            - Updates self.val_loss_min with the new minimum validation loss
            - Updates self.best_epoch if epoch is provided
            - Updates self.best_train_loss if train_loss is provided
            - Prints improvement message if verbose mode is enabled
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        # Track best epoch and train loss if provided
        if epoch is not None:
            self.best_epoch = epoch
        if train_loss is not None:
            self.best_train_loss = train_loss

        # Handle torch.compile: access underlying model to save state_dict without _orig_mod prefix
        # This ensures checkpoints are consistent regardless of compilation status
        # First check if model is wrapped in DataParallel
        if hasattr(model, 'module'):
            # Model wrapped in DataParallel
            if hasattr(model.module, '_orig_mod'):
                # DataParallel + compiled - access underlying model
                state_dict = model.module._orig_mod.state_dict()
            else:
                # DataParallel but not compiled - save underlying module
                state_dict = model.module.state_dict()
        elif hasattr(model, '_orig_mod'):
            # Model is compiled (not DataParallel) - save underlying model's state_dict
            state_dict = model._orig_mod.state_dict()
        else:
            # Model not compiled and not DataParallel - save normally
            state_dict = model.state_dict()

        # Save checkpoint as dict with model state and metrics from best epoch
        # This is THE BEST checkpoint - saved when validation loss improves
        # Contains:
        #   - model_state_dict: Model parameters at best epoch
        #   - epoch: Best epoch number (1-indexed)
        #   - train_loss: Training loss at best epoch
        #   - val_loss: Validation loss at best epoch (the metric being optimized)
        # Note: Test metrics are added later by _finalize_training after final test evaluation
        checkpoint = {
            'model_state_dict': state_dict,
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
        }

        torch.save(checkpoint, path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss



class dotdict(dict):
    """
    A dictionary subclass that allows attribute-style access to dictionary keys.
    
    This class extends the built-in dict class to provide convenient dot notation
    access to dictionary items, making configuration objects more intuitive to use.
    It supports getting, setting, and deleting attributes as if they were regular
    object attributes while maintaining full dictionary functionality.
    
    Example:
        >>> config = dotdict({'learning_rate': 0.001, 'epochs': 100})
        >>> config.learning_rate  # Returns 0.001
        >>> config.batch_size = 32  # Sets config['batch_size'] = 32
        >>> del config.epochs  # Deletes config['epochs']
    """
    def __getattr__(self, name):
        """
        Get dictionary value using attribute notation.
        
        Args:
            name (str): The key name to retrieve from the dictionary.
        
        Returns:
            object: The value associated with the key, or None if key doesn't exist.
        """
        if name in self.keys():

            return self[name]
        else:
            return None
    def __setattr__(self, name, value):
        """
        Set dictionary value using attribute notation.
        
        Args:
            name (str): The key name to set in the dictionary.
            value (object): The value to associate with the key.
        """
        self[name] = value
    def __delattr__(self, name):
        """
        Delete dictionary key using attribute notation.
        
        Args:
            name (str): The key name to delete from the dictionary.
        
        Raises:
            KeyError: If the key doesn't exist in the dictionary.
        """
        del self[name]


class StandardScaler():
    """
    A standard scaler for normalizing data using pre-computed mean and standard deviation.
    
    This class implements z-score normalization (standardization) using provided
    mean and standard deviation values. Unlike sklearn's StandardScaler, this
    implementation uses pre-computed statistics rather than computing them from data.
    
    Attributes:
        mean (float or array-like): The mean value(s) used for normalization
        std (float or array-like): The standard deviation value(s) used for normalization
    """
    def __init__(self, mean, std):
        """
        Initialize the StandardScaler with pre-computed statistics.
        
        Args:
            mean (float or array-like): Mean value(s) to subtract during transformation.
                                      Should match the dimensionality of the data to be transformed.
            std (float or array-like): Standard deviation value(s) to divide by during transformation.
                                     Should match the dimensionality of the data to be transformed.
                                     Must be non-zero to avoid division by zero.
        """
        self.mean = mean
        self.std = std

    def transform(self, data):
        """
        Apply standardization to the input data.
        
        Transforms the data by subtracting the mean and dividing by the standard deviation:
        transformed_data = (data - mean) / std
        
        Args:
            data (array-like): Input data to be standardized. Should be compatible
                             with the mean and std provided during initialization.
        
        Returns:
            array-like: Standardized data with zero mean and unit variance (approximately).
        """
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """
        Reverse the standardization transformation.
        
        Converts standardized data back to the original scale by multiplying by
        the standard deviation and adding the mean:
        original_data = (data * std) + mean
        
        Args:
            data (array-like): Standardized data to be converted back to original scale.
        
        Returns:
            array-like: Data in the original scale before standardization.
        """
        return (data * self.std) + self.mean

import psutil, os
from datetime import datetime


def format_test_results(per_entity_metrics, best_epoch, checkpoint_path):
    """
    Format test evaluation results into a standardized structure.

    Creates a comprehensive results dictionary containing overall metrics (weighted
    average across entities), per-entity metrics, and metadata for traceability.

    Args:
        per_entity_metrics: dict mapping entity_id to metrics dict with keys:
            - 'mse_normalized': float
            - 'mae_normalized': float
            - 'mse_denormalized': float (optional)
            - 'mae_denormalized': float (optional)
            - 'num_samples': int
        best_epoch: int, the epoch number of the best checkpoint
        checkpoint_path: str, path to the checkpoint file used for evaluation

    Returns:
        dict: Standardized results with structure:
            {
                "overall": {
                    "mse_normalized": float,
                    "mae_normalized": float,
                    "mse_denormalized": float (if available),
                    "mae_denormalized": float (if available),
                    "num_samples": int
                },
                "per_entity": {
                    "entity_id": {...metrics...},
                    ...
                },
                "metadata": {
                    "timestamp": str (ISO format),
                    "best_epoch": int,
                    "best_checkpoint": str
                }
            }

    Example:
        >>> metrics = {
        ...     'entity_1': {'mse_normalized': 0.01, 'mae_normalized': 0.05, 'num_samples': 100},
        ...     'entity_2': {'mse_normalized': 0.02, 'mae_normalized': 0.08, 'num_samples': 200}
        ... }
        >>> results = format_test_results(metrics, best_epoch=15, checkpoint_path='checkpoint.pth')
        >>> results['overall']['mse_normalized']  # Weighted average
        0.0166...
    """
    if not per_entity_metrics:
        return None

    # Compute weighted averages across entities
    total_samples = 0
    total_mse_norm = 0.0
    total_mae_norm = 0.0
    total_mse_denorm = 0.0
    total_mae_denorm = 0.0
    has_denorm_metrics = False
    denorm_samples = 0

    for entity_id, metrics in per_entity_metrics.items():
        num_samples = metrics.get('num_samples', 0)
        if num_samples == 0:
            continue

        total_samples += num_samples
        total_mse_norm += metrics.get('mse_normalized', 0.0) * num_samples
        total_mae_norm += metrics.get('mae_normalized', 0.0) * num_samples

        # Track denormalized metrics separately (may not be available for all entities)
        if 'mse_denormalized' in metrics and metrics['mse_denormalized'] is not None:
            has_denorm_metrics = True
            total_mse_denorm += metrics['mse_denormalized'] * num_samples
            total_mae_denorm += metrics.get('mae_denormalized', 0.0) * num_samples
            denorm_samples += num_samples

    if total_samples == 0:
        return None

    # Build overall metrics
    overall = {
        'mse_normalized': total_mse_norm / total_samples,
        'mae_normalized': total_mae_norm / total_samples,
        'num_samples': total_samples
    }

    # Only include denormalized metrics if available
    if has_denorm_metrics and denorm_samples > 0:
        overall['mse_denormalized'] = total_mse_denorm / denorm_samples
        overall['mae_denormalized'] = total_mae_denorm / denorm_samples

    # Build result structure
    result = {
        'overall': overall,
        'per_entity': per_entity_metrics,
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'best_epoch': best_epoch,
            'best_checkpoint': str(checkpoint_path) if checkpoint_path else None
        }
    }

    return result


def log_memory_usage():
    """
    Log the current memory usage of the running process.
    
    This utility function prints the memory consumption of the current Python process
    in megabytes. It's useful for monitoring memory usage during training or debugging
    memory-related issues.
    
    Side Effects:
        Prints a message showing the process ID and memory usage in MB to the console.
    
    Example:
        >>> log_memory_usage()
        Memory usage 12345: 1024.50 MB
    """
    process = psutil.Process(os.getpid())
    print(f"Memory usage {os.getpid()}: {process.memory_info().rss / 1024 ** 2:.2f} MB")


def general_move_to_device(batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel, device):
    """
    Move time series data tensors to the specified computing device.
    
    This utility function handles the transfer of multiple data tensors from CPU to GPU
    or between different devices. It's specifically designed for time series forecasting
    workflows that involve both homogeneous time series data and heterogeneous auxiliary data.
    
    Args:
        batch_x (torch.Tensor): Input time series data tensor.
        batch_y (torch.Tensor): Target time series data tensor.
        timestamp_x (object): Timestamps for input data (not moved to device).
        timestamp_y (object): Timestamps for target data (not moved to device).
        batch_x_hetero (object): Heterogeneous input data (not moved to device).
        batch_y_hetero (object): Heterogeneous target data (not moved to device).
        hetero_x_time (object): Heterogeneous input time data (not moved to device).
        hetero_y_time (object): Heterogeneous target time data (not moved to device).
        hetero_general (object): General heterogeneous data (not moved to device).
        hetero_channel (object): Channel-specific heterogeneous data (not moved to device).
        device (str or torch.device): Target device for tensor placement (e.g., 'cuda:0', 'cpu').
    
    Returns:
        tuple: A tuple containing all input arguments with batch_x and batch_y moved to
               the specified device as float tensors, while other arguments remain unchanged.
    
    Note:
        Only batch_x and batch_y tensors are moved to the device and converted to float.
        All other arguments (timestamps, heterogeneous data) are returned as-is.
    """
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float().to(device)

    return batch_x, batch_y, timestamp_x, timestamp_y, batch_x_hetero, batch_y_hetero, hetero_x_time, hetero_y_time, hetero_general, hetero_channel