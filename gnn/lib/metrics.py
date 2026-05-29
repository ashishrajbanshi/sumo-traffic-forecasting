"""metrics.py — MAE, RMSE, MAPE with zero-masking (same as DCRNN)."""

import torch


def _mask(true):
    return (true.abs() > 1e-4).float()


def masked_mae(pred, true):
    m = _mask(true)
    return (m * (pred - true).abs()).sum() / m.sum().clamp(min=1)


def masked_rmse(pred, true):
    m = _mask(true)
    return ((m * (pred - true) ** 2).sum() / m.sum().clamp(min=1)).sqrt()


def masked_mape(pred, true):
    m = _mask(true)
    return (m * ((pred - true).abs() / true.abs().clamp(min=1e-4))).sum() / m.sum().clamp(min=1) * 100
