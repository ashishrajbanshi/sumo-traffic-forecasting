"""
dataset.py — Speed2Vec-style sliding window dataset (DeepSUMO style).

Each sample: (x, y) where
    x : (N_hist, N, F)  — past F-feature observations
    y : (N_pred, N, 1)  — future speed values to predict
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TrafficDataset(Dataset):
    def __init__(self, data: np.ndarray, n_hist: int, n_pred: int,
                 scaler_mean: np.ndarray, scaler_std: np.ndarray):
        """
        Args:
            data        : (T, N, F)  raw feature values
            n_hist      : number of input timesteps
            n_pred      : number of output timesteps
            scaler_mean : (N, F)  fitted on training set
            scaler_std  : (N, F)  fitted on training set
        """
        self.n_hist = n_hist
        self.n_pred = n_pred
        self.mean   = scaler_mean.astype(np.float32)
        self.std    = scaler_std.astype(np.float32)

        # Normalise all features
        norm = (data.astype(np.float32) - self.mean[None]) / self.std[None]
        self.data = norm                               # (T, N, F)

        # Speed is feature index 0 — target is speed only
        self.n_samples = len(data) - n_hist - n_pred + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.data[idx            : idx + self.n_hist]           # (n_hist, N, F)
        y = self.data[idx + self.n_hist : idx + self.n_hist + self.n_pred, :, 0:1]  # (n_pred, N, 1)
        return torch.FloatTensor(x), torch.FloatTensor(y)


def make_loaders(npz_path: str, n_hist: int, n_pred: int,
                 batch_size: int, val_batch_size: int = 64, test_batch_size: int = 64):
    """Load dataset.npz and return (train_loader, val_loader, test_loader, meta)."""
    d = np.load(npz_path, allow_pickle=True)

    scaler_mean = d['scaler_mean']   # (N, F)
    scaler_std  = d['scaler_std']    # (N, F)
    node_ids    = d['node_ids']

    train_ds = TrafficDataset(d['x_train'], n_hist, n_pred, scaler_mean, scaler_std)
    val_ds   = TrafficDataset(d['x_val'],   n_hist, n_pred, scaler_mean, scaler_std)
    test_ds  = TrafficDataset(d['x_test'],  n_hist, n_pred, scaler_mean, scaler_std)

    train_loader = DataLoader(train_ds, batch_size=batch_size,     shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=val_batch_size,  shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=test_batch_size, shuffle=False)

    meta = {
        'scaler_mean': scaler_mean,
        'scaler_std':  scaler_std,
        'node_ids':    node_ids,
        'n_train':     len(train_ds),
        'n_val':       len(val_ds),
        'n_test':      len(test_ds),
    }
    return train_loader, val_loader, test_loader, meta
