#!/usr/bin/env python3
"""
prepare_data.py — Prepare multi-feature sliding-window dataset (DeepSUMO style).

Reads traffic_jan2026.h5 (speed, occupancy, volume), aligns with the graph
node list, filters low-variance detectors, and saves train/val/test arrays.

Run from gnn/:
    python scripts/prepare_data.py [--config config.yaml]

Outputs (saved to data/):
    dataset.npz — arrays:
        x_train  (T_train, N, F)  node features
        x_val    (T_val,   N, F)
        x_test   (T_test,  N, F)
        node_ids (N,)   string array of kept node IDs
        scaler_mean (N, F)
        scaler_std  (N, F)
"""

import argparse
import os

import numpy as np
import pandas as pd
import yaml


FEATURES = ['speed', 'occupancy', 'volume']


def load_feature(h5_path: str, key: str, node_ids: list) -> np.ndarray:
    """Load one feature from HDF5, aligned to node_ids order. Returns (T, N)."""
    df = pd.read_hdf(h5_path, key=key)
    # Keep only nodes that exist in the graph
    available = [nid for nid in node_ids if nid in df.columns]
    missing   = [nid for nid in node_ids if nid not in df.columns]
    if missing:
        print(f'  [{key}] {len(missing)} nodes missing from HDF5 — will be zero-filled')
    arr = np.zeros((len(df), len(node_ids)), dtype=np.float32)
    for i, nid in enumerate(node_ids):
        if nid in df.columns:
            arr[:, i] = df[nid].values.astype(np.float32)
    return arr


def prepare_data(cfg: dict):
    data_cfg  = cfg['data']
    model_cfg = cfg['model']

    # ── Load graph node list ───────────────────────────────────────────────────
    import pickle
    graph_path = os.path.join(data_cfg['output_dir'], 'graph.pkl')
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f'Graph not found at {graph_path}. Run build_graph.py first.')
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)

    node_ids = graph['node_ids']
    N_all    = len(node_ids)
    print(f'Graph nodes: {N_all}')

    # ── Load all features (T, N_all) ──────────────────────────────────────────
    h5_path  = data_cfg['traffic_h5']
    feat_names = data_cfg.get('features', FEATURES)
    arrays = []
    for feat in feat_names:
        print(f'Loading {feat} ...')
        arr = load_feature(h5_path, feat, node_ids)
        arrays.append(arr)
        print(f'  shape={arr.shape}  min={arr.min():.2f}  max={arr.max():.2f}')

    X = np.stack(arrays, axis=-1)   # (T, N, F)
    T, N, F = X.shape
    print(f'\nFull dataset shape: {X.shape}  (timesteps × nodes × features)')

    # ── Variance filter: drop low-variance speed nodes ────────────────────────
    speed_idx = feat_names.index('speed') if 'speed' in feat_names else 0
    speed_var = X[:, :, speed_idx].var(axis=0)   # (N,)
    var_thresh = data_cfg.get('variance_threshold', 1.0)
    keep_mask  = speed_var >= var_thresh
    n_removed  = (~keep_mask).sum()
    print(f'\nVariance filter (threshold={var_thresh}): removing {n_removed} low-variance nodes')
    print(f'Keeping {keep_mask.sum()} / {N} nodes')

    X        = X[:, keep_mask, :]
    node_ids = [nid for nid, k in zip(node_ids, keep_mask) if k]
    N        = X.shape[1]

    # ── Update adjacency to match filtered nodes ───────────────────────────────
    adj_full = graph['adj_matrix']                             # (N_all, N_all)
    adj      = adj_full[np.ix_(keep_mask, keep_mask)]         # (N, N)
    n_edges  = int(adj.sum())
    isolated = int((adj.sum(axis=1) == 0).sum())
    print(f'Graph after filter: {N} nodes, {n_edges} edges, {isolated} isolated')

    # Re-save filtered graph
    graph_filtered = {
        'adj_matrix':  adj.astype(np.float32),
        'node_ids':    node_ids,
        'num_nodes':   N,
        'num_edges':   n_edges,
        'threshold_m': graph['threshold_m'],
    }
    import pickle
    filtered_path = os.path.join(data_cfg['output_dir'], 'graph_filtered.pkl')
    with open(filtered_path, 'wb') as f:
        pickle.dump(graph_filtered, f)
    print(f'Filtered graph saved → {filtered_path}')

    # ── Train / val / test split (chronological) ──────────────────────────────
    train_r = data_cfg.get('train_ratio', 0.70)
    val_r   = data_cfg.get('val_ratio',   0.10)
    t_train = int(T * train_r)
    t_val   = int(T * (train_r + val_r))

    x_train = X[:t_train]
    x_val   = X[t_train:t_val]
    x_test  = X[t_val:]
    print(f'\nSplit: train={x_train.shape} val={x_val.shape} test={x_test.shape}')

    # ── Fit StandardScaler on train only ──────────────────────────────────────
    scaler_mean = x_train.mean(axis=0)   # (N, F)
    scaler_std  = x_train.std(axis=0)    # (N, F)
    scaler_std[scaler_std < 1e-6] = 1.0  # avoid div-by-zero

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(data_cfg['output_dir'], 'dataset.npz')
    np.savez(
        out_path,
        x_train      = x_train,
        x_val        = x_val,
        x_test       = x_test,
        node_ids     = np.array(node_ids),
        scaler_mean  = scaler_mean,
        scaler_std   = scaler_std,
        feat_names   = np.array(feat_names),
    )
    print(f'Dataset saved → {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    prepare_data(cfg)


if __name__ == '__main__':
    main()
