#!/usr/bin/env python3
"""
predict.py — Run ST-GAT predictions on a specific date/time window.

Run from gnn/:
    python predict.py \\
        --checkpoint data/runs/stgat_.../best_model.pt \\
        --start      "2026-01-25 08:00" \\
        --hours      2
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch

from model.stgat import STGAT


def load_model(ckpt_path, device):
    ckpt      = torch.load(ckpt_path, map_location=device)
    cfg       = ckpt['model_cfg']
    data_cfg  = ckpt['data_cfg']

    with open(ckpt['graph_path'], 'rb') as f:
        graph = pickle.load(f)

    adj = torch.FloatTensor(graph['adj_matrix']).to(device)
    N   = graph['num_nodes']

    model = STGAT(
        n_nodes     = N,
        n_features  = cfg['n_features'],
        n_hist      = cfg['n_hist'],
        n_pred      = cfg['n_pred'],
        gat_heads   = cfg['gat_heads'],
        gat_hidden  = cfg['gat_hidden'],
        gat_layers  = cfg['gat_layers'],
        lstm_hidden = cfg['lstm_hidden'],
        dropout     = 0.0,
    ).to(device)

    model.load_state_dict(ckpt['model_state'])
    model.eval()

    mean = ckpt['scaler_mean']   # (N, F)
    std  = ckpt['scaler_std']    # (N, F)

    print(f'Model loaded from epoch {ckpt["epoch"]}  val_mae={ckpt["val_mae"]:.4f} m/s')
    return model, adj, mean, std, cfg, graph


@torch.no_grad()
def predict_window(model, adj, mean, std, node_ids, start_ts, hours, device,
                   n_hist=12, n_pred=12):
    steps_needed = hours * 12
    num_runs     = (steps_needed + n_pred - 1) // n_pred

    # Load traffic data
    import yaml
    # Try to find traffic h5
    h5_options = [
        '../data/traffic/traffic_jan2026.h5',
        'data/traffic/traffic_jan2026.h5',
    ]
    h5_path = next((p for p in h5_options if os.path.exists(p)), None)
    if h5_path is None:
        raise FileNotFoundError('Could not find traffic_jan2026.h5')

    feat_names = ['speed', 'occupancy', 'volume']
    N = len(node_ids)
    F = len(feat_names)

    arrays = {}
    for feat in feat_names:
        df = pd.read_hdf(h5_path, key=feat)
        arrays[feat] = df

    # Get common index
    idx_ref = arrays['speed']
    ts_index = idx_ref.index.get_loc(pd.Timestamp(start_ts))
    if ts_index < n_hist:
        raise ValueError(f'Not enough history before {start_ts}')

    past_slice = slice(ts_index - n_hist + 1, ts_index + 1)
    past_times = idx_ref.index[past_slice]

    # Build input (n_hist, N, F)
    raw = np.zeros((n_hist, N, F), dtype=np.float32)
    for fi, feat in enumerate(feat_names):
        for ni, nid in enumerate(node_ids):
            if nid in arrays[feat].columns:
                raw[:, ni, fi] = arrays[feat].loc[past_times, nid].values.astype(np.float32)

    # Normalise
    x_norm = (raw - mean[None]) / std[None]   # (n_hist, N, F)

    all_preds = []
    all_times = []
    current_x    = torch.FloatTensor(x_norm).unsqueeze(0).to(device)  # (1, n_hist, N, F)
    current_time = pd.Timestamp(start_ts)

    for run in range(num_runs):
        pred = model(current_x, adj)     # (1, n_pred, N, 1) normalised
        pred_speed = pred[0, :, :, 0].cpu().numpy() * std[:, 0] + mean[:, 0]  # (n_pred, N)

        run_times = pd.date_range(
            start=current_time + pd.Timedelta(minutes=5),
            periods=n_pred,
            freq='5min'
        )
        all_preds.append(pred_speed)
        all_times.extend(run_times)

        # Slide window with predicted speed (only speed feature kept, others from data if avail)
        next_norm = pred[0, :, :, 0].cpu().numpy()   # (n_pred, N)  normalised speed
        # For other features, use zeros (unavailable future)
        next_x = np.zeros((n_pred, N, F), dtype=np.float32)
        next_x[:, :, 0] = next_norm
        combined = np.concatenate([current_x[0].cpu().numpy(), next_x], axis=0)  # (n_hist+n_pred, N, F)
        next_window = combined[-n_hist:]
        current_x    = torch.FloatTensor(next_window).unsqueeze(0).to(device)
        current_time = run_times[-1]

    all_preds = np.concatenate(all_preds, axis=0)[:steps_needed]
    all_times = all_times[:steps_needed]

    pred_df = pd.DataFrame(all_preds, index=all_times, columns=node_ids)

    # Ground truth if available
    true_df = None
    try:
        true_df = arrays['speed'][node_ids].loc[all_times]
    except Exception:
        pass

    return pred_df, true_df


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, adj, mean, std, cfg, graph = load_model(args.checkpoint, device)
    node_ids = graph['node_ids']

    print(f'\nPredicting {args.hours}h ahead from {args.start} ...')
    pred_df, true_df = predict_window(
        model, adj, mean, std, node_ids,
        start_ts=args.start, hours=args.hours, device=device,
        n_hist=cfg['n_hist'], n_pred=cfg['n_pred'],
    )

    print(f'\nPrediction window: {pred_df.index[0]} → {pred_df.index[-1]}')
    print(f'Shape: {pred_df.shape}')
    print(f'\nSample (first 3 nodes, first 6 steps):')
    print(pred_df.iloc[:6, :3].round(3).to_string())

    out_stem = os.path.join(
        os.path.dirname(args.checkpoint),
        f'pred_{pd.Timestamp(args.start).strftime("%Y-%m-%d_%H-%M")}_{args.hours}h'
    )
    pred_df.to_csv(out_stem + '.csv')
    print(f'\nCSV saved → {out_stem}.csv')

    if true_df is not None:
        from lib.metrics import masked_mae, masked_rmse, masked_mape
        p = torch.FloatTensor(pred_df.values)
        t = torch.FloatTensor(true_df.values)
        print(f'\nMetrics vs ground truth:')
        print(f'  MAE  = {masked_mae(p, t).item():.4f} m/s')
        print(f'  RMSE = {masked_rmse(p, t).item():.4f} m/s')
        print(f'  MAPE = {masked_mape(p, t).item():.2f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--start', default='2026-01-25 08:00')
    parser.add_argument('--hours', type=int, default=2)
    args = parser.parse_args()
    main(args)
