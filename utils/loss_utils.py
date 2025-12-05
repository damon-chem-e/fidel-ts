
import torch
import torch.nn as nn

def compute_per_sample_loss(output, gt, criterion):
    """
    Compute loss for each sample in batch.
    
    Args:
        output: Model predictions [batch_size, seq_len, num_vars]
        gt: Ground truth [batch_size, seq_len, num_vars]
        criterion: Loss function (e.g., nn.MSELoss)
        
    Returns:
        losses: Per-sample losses [batch_size]
    """
    if isinstance(criterion, nn.MSELoss):
        # For MSE, compute per-sample manually: mean over time and vars
        # output shape: [B, L, D]
        # (output - gt)**2 -> [B, L, D]
        # mean(dim=(1,2)) -> [B]
        mse_per_sample = torch.mean((output - gt) ** 2, dim=(1, 2))
        return mse_per_sample
    elif isinstance(criterion, nn.L1Loss):
        mae_per_sample = torch.mean(torch.abs(output - gt), dim=(1, 2))
        return mae_per_sample
    else:
        # Fallback for other losses: compute iteratively (slower)
        batch_size = output.shape[0]
        losses = []
        for i in range(batch_size):
            # Keep dimensions to satisfy criterion expectations
            sample_loss = criterion(output[i:i+1], gt[i:i+1])
            losses.append(sample_loss.item())
        return torch.tensor(losses, device=output.device)


def compute_per_sample_per_channel_loss(output, gt, criterion):
    """
    Compute loss for each sample and each channel in batch.
    
    Averages loss over time dimension only, preserving channel dimension.
    This enables per-channel analysis of model performance.
    
    Args:
        output: Model predictions [batch_size, seq_len, num_vars]
        gt: Ground truth [batch_size, seq_len, num_vars]
        criterion: Loss function (e.g., nn.MSELoss)
        
    Returns:
        losses: Per-sample per-channel losses [batch_size, num_vars]
                Each element is the average loss over time for that channel
    """
    if isinstance(criterion, nn.MSELoss):
        # For MSE, compute per-sample per-channel: mean over time only
        # output shape: [B, L, D]
        # (output - gt)**2 -> [B, L, D]
        # mean(dim=1) -> [B, D] (mean over time, keep channels)
        mse_per_sample_per_channel = torch.mean((output - gt) ** 2, dim=1)
        return mse_per_sample_per_channel
    elif isinstance(criterion, nn.L1Loss):
        # For MAE, compute per-sample per-channel: mean over time only
        mae_per_sample_per_channel = torch.mean(torch.abs(output - gt), dim=1)
        return mae_per_sample_per_channel
    else:
        # Fallback for other losses: compute iteratively (slower)
        batch_size, seq_len, num_vars = output.shape
        losses = []
        for i in range(batch_size):
            sample_losses = []
            for j in range(num_vars):
                # Loss for sample i, channel j (averaged over time)
                # Keep dimensions to satisfy criterion expectations
                channel_loss = criterion(
                    output[i:i+1, :, j:j+1], 
                    gt[i:i+1, :, j:j+1]
                )
                sample_losses.append(channel_loss.item())
            losses.append(sample_losses)
        return torch.tensor(losses, device=output.device)
