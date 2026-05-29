import torch


def masked_mae(pred, true, null_val=0.0):
    """Mean Absolute Error, ignoring positions where true == null_val (missing data)."""
    mask = (true != null_val).float()
    mask /= mask.mean().clamp(min=1e-5)
    loss = torch.abs(pred - true) * mask
    return loss.mean()


def masked_mse(pred, true, null_val=0.0):
    mask = (true != null_val).float()
    mask /= mask.mean().clamp(min=1e-5)
    loss = ((pred - true) ** 2) * mask
    return loss.mean()


def masked_rmse(pred, true, null_val=0.0):
    return torch.sqrt(masked_mse(pred, true, null_val))


def masked_mape(pred, true, null_val=0.0):
    mask = (true != null_val).float()
    mask /= mask.mean().clamp(min=1e-5)
    loss = torch.abs((pred - true) / true.clamp(min=1e-5)) * mask
    return loss.mean() * 100
