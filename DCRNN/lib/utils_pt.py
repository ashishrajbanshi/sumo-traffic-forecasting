import logging
import numpy as np
import os
import pickle
import scipy.sparse as sp
import torch


# ── Scaler ─────────────────────────────────────────────────────────────────────

class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


# ── Data loader ────────────────────────────────────────────────────────────────

class DataLoader:
    def __init__(self, x, y, batch_size, shuffle=False):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.size = len(x)

    def __len__(self):
        return (self.size + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        idx = np.arange(self.size)
        if self.shuffle:
            np.random.shuffle(idx)
        for start in range(0, self.size, self.batch_size):
            end = min(start + self.batch_size, self.size)
            b = idx[start:end]
            yield torch.FloatTensor(self.x[b]), torch.FloatTensor(self.y[b])


def load_dataset(dataset_dir, batch_size, val_batch_size, test_batch_size, **_):
    data = {}
    for cat in ('train', 'val', 'test'):
        d = np.load(os.path.join(dataset_dir, f'{cat}.npz'))
        data[f'x_{cat}'] = d['x'].astype(np.float32)
        data[f'y_{cat}'] = d['y'].astype(np.float32)

    # Fit scaler on channel 0 (speed) of training set only
    scaler = StandardScaler(
        mean=data['x_train'][..., 0].mean(),
        std=data['x_train'][..., 0].std(),
    )

    for cat in ('train', 'val', 'test'):
        data[f'x_{cat}'][..., 0] = scaler.transform(data[f'x_{cat}'][..., 0])
        data[f'y_{cat}'][..., 0] = scaler.transform(data[f'y_{cat}'][..., 0])

    data['train_loader'] = DataLoader(data['x_train'], data['y_train'],
                                      batch_size, shuffle=True)
    data['val_loader']   = DataLoader(data['x_val'],   data['y_val'],   val_batch_size)
    data['test_loader']  = DataLoader(data['x_test'],  data['y_test'],  test_batch_size)
    data['scaler'] = scaler
    return data


# ── Graph helpers ──────────────────────────────────────────────────────────────

def load_graph_data(pkl_filename):
    with open(pkl_filename, 'rb') as f:
        sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f)
    return sensor_ids, sensor_id_to_ind, adj_mx


def _row_normalize(adj):
    """Return D^{-1} A as CSR sparse matrix."""
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1)).flatten()
    d_inv = np.zeros_like(d)
    nonzero = d != 0
    d_inv[nonzero] = 1.0 / d[nonzero]
    return (sp.diags(d_inv) @ adj).tocsr()


def build_sparse_supports(adj_mx, filter_type='dual_random_walk'):
    """Return list of scipy sparse support matrices."""
    if filter_type == 'dual_random_walk':
        return [_row_normalize(adj_mx), _row_normalize(adj_mx.T)]
    if filter_type == 'random_walk':
        return [_row_normalize(adj_mx)]
    raise ValueError(f'Unknown filter_type: {filter_type}')


def sparse_to_torch(sparse_mx):
    """Convert scipy COO sparse matrix to a coalesced torch sparse_coo_tensor."""
    coo = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack([coo.row, coo.col])).long()
    values  = torch.from_numpy(coo.data).float()
    return torch.sparse_coo_tensor(indices, values, coo.shape).coalesce()


# ── Logging ────────────────────────────────────────────────────────────────────

def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(os.path.join(log_dir, log_filename))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger
