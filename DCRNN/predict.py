#!/usr/bin/env python3
"""
predict.py — Run DCRNN predictions on a specific date/time window.

Loads the best trained model and predicts speed for the next N hours
starting from a given date and time that exists in the dataset.

Usage (from sumo_3/DCRNN/):

    # Predict next 2 hours from 2026-01-25 08:00 using best run
    python predict.py \\
        --checkpoint data/model/dcrnn_pt_DR_2_.../best_model.pt \\
        --start      "2026-01-25 08:00" \\
        --hours      2

    # List available dates in the test set
    python predict.py --list_dates

Output (saved next to the checkpoint):
    predictions_YYYY-MM-DD_HH-MM_Nh.csv   speed predictions (timesteps × nodes)
    predictions_YYYY-MM-DD_HH-MM_Nh.png   plot of sample nodes
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch

from lib.utils_pt       import load_graph_data, build_sparse_supports, sparse_to_torch
from lib.metrics_pt     import masked_mae, masked_rmse, masked_mape
from model.dcrnn_model_pt import DCRNNModel


# ── Helpers ────────────────────────────────────────────────────────────────────

class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std  = std

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse_transform(self, x):
        return x * self.std + self.mean


def load_speed_df():
    """Load the full speed HDF5 as a DataFrame (timesteps × nodes)."""
    return pd.read_hdf('data/traffic_chattanooga.h5')


def time_of_day(timestamps):
    """Return fraction-of-day values (0.0 = midnight, 0.5 = noon) for each timestamp."""
    return np.array([
        (ts.hour * 3600 + ts.minute * 60 + ts.second) / 86400.0
        for ts in timestamps
    ], dtype=np.float32)


def load_model(checkpoint_path, device):
    """Load model and scaler from a best_model.pt checkpoint."""
    ckpt      = torch.load(checkpoint_path, map_location=device)
    model_cfg = ckpt['model_cfg']
    data_cfg  = ckpt['data_cfg']

    _, _, adj_mx   = load_graph_data(data_cfg['graph_pkl_filename'])
    scipy_supports = build_sparse_supports(adj_mx, model_cfg['filter_type'])
    supports       = [sparse_to_torch(s).to_dense().to(device) for s in scipy_supports]

    model = DCRNNModel(
        input_dim          = model_cfg['input_dim'],
        output_dim         = model_cfg['output_dim'],
        num_units          = model_cfg['rnn_units'],
        num_nodes          = model_cfg['num_nodes'],
        supports           = supports,
        max_diffusion_step = model_cfg['max_diffusion_step'],
        num_layers         = model_cfg['num_rnn_layers'],
        seq_len            = model_cfg['seq_len'],
        horizon            = model_cfg['horizon'],
    ).to(device)

    model.load_state_dict(ckpt['model_state'])
    model.eval()

    scaler = StandardScaler(mean=ckpt['scaler_mean'], std=ckpt['scaler_std'])

    print(f'Model loaded from epoch {ckpt["epoch"]}  val_mae={ckpt["val_mae"]:.4f} mph')
    return model, scaler, model_cfg


@torch.no_grad()
def predict_window(model, scaler, speed_df, start_ts, hours, device, seq_len=12):
    """
    Predict speed for `hours` hours starting from start_ts.

    The model natively predicts `horizon` steps (1 hour = 12 × 5min).
    For longer horizons, the model is run auto-regressively:
      - Run 1 → predict steps  1–12  (t+5min  to t+60min)
      - Run 2 → predict steps 13–24  (t+65min to t+120min)
      ...

    Args:
        start_ts : pandas Timestamp — the last observed timestamp
                   (model looks back seq_len steps from here)
        hours    : int — how many hours ahead to predict

    Returns:
        pred_df  : DataFrame  (prediction_timestamps × nodes)  speed in mph
        true_df  : DataFrame  (prediction_timestamps × nodes)  true speed if available
    """
    horizon      = model.decoder._horizon
    steps_needed = hours * 12          # 1 hour = 12 × 5-min steps
    num_runs     = (steps_needed + horizon - 1) // horizon   # ceil division

    # ── Build encoder input ────────────────────────────────────────────────────
    # The model needs seq_len past timesteps ending at start_ts
    ts_index = speed_df.index.get_loc(start_ts)
    if ts_index < seq_len:
        raise ValueError(
            f'Not enough history before {start_ts}. '
            f'Need {seq_len} steps, only {ts_index} available.'
        )

    past_slice     = speed_df.iloc[ts_index - seq_len + 1 : ts_index + 1]  # (12, N)
    past_speed     = past_slice.values.astype(np.float32)                   # (12, N)
    past_tod       = time_of_day(past_slice.index)                          # (12,)
    past_tod_nodes = np.tile(past_tod[:, None], (1, past_speed.shape[1]))   # (12, N)

    # Normalise speed channel
    past_speed_norm = scaler.transform(past_speed)

    # Build input tensor (1, seq_len, N, 2)
    x = np.stack([past_speed_norm, past_tod_nodes], axis=-1)  # (12, N, 2)
    x = torch.FloatTensor(x).unsqueeze(0).to(device)          # (1, 12, N, 2)

    # ── Auto-regressive prediction ─────────────────────────────────────────────
    all_preds = []
    all_times = []

    current_x    = x
    current_time = start_ts

    for run in range(num_runs):
        pred = model(current_x)              # (1, horizon, N, 1)  normalised
        pred_speed = scaler.inverse_transform(pred[0, :, :, 0].cpu().numpy())  # (horizon, N)

        # Timestamps for this run's predictions
        run_times = pd.date_range(
            start=current_time + pd.Timedelta(minutes=5),
            periods=horizon,
            freq='5min'
        )

        all_preds.append(pred_speed)
        all_times.extend(run_times)

        # Build input for next run from predicted speeds
        next_speed_norm = scaler.transform(pred_speed)        # (horizon, N)
        next_tod        = time_of_day(run_times)              # (horizon,)
        next_tod_nodes  = np.tile(next_tod[:, None], (1, pred_speed.shape[1]))
        next_x = np.stack([next_speed_norm, next_tod_nodes], axis=-1)  # (horizon, N, 2)

        # Slide window: take last seq_len steps from [current window + predictions]
        combined_norm  = np.concatenate([
            current_x[0, :, :, 0].cpu().numpy(),  # (12, N) normalised speed
            next_speed_norm                         # (horizon, N) normalised speed
        ], axis=0)
        combined_tod = np.concatenate([
            current_x[0, :, :, 1].cpu().numpy(),   # (12, N) time-of-day
            next_tod_nodes                           # (horizon, N)
        ], axis=0)
        combined = np.stack([combined_norm, combined_tod], axis=-1)  # (12+horizon, N, 2)
        next_window = combined[-seq_len:]           # (12, N, 2) — last seq_len steps

        current_x    = torch.FloatTensor(next_window).unsqueeze(0).to(device)
        current_time = run_times[-1]

    # ── Assemble results ───────────────────────────────────────────────────────
    all_preds = np.concatenate(all_preds, axis=0)[:steps_needed]  # (steps_needed, N)
    all_times = all_times[:steps_needed]

    pred_df = pd.DataFrame(all_preds, index=all_times, columns=speed_df.columns)

    # True values if available in the dataset
    true_df = None
    try:
        true_df = speed_df.loc[all_times]
    except KeyError:
        pass

    return pred_df, true_df


def plot_predictions(pred_df, true_df, start_ts, hours, out_path, n_nodes=6):
    """Plot predicted vs true speed for a sample of nodes."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        nodes = pred_df.columns[:n_nodes].tolist()
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(
            f'DCRNN Speed Predictions — {hours}h ahead from {start_ts}',
            fontsize=13, fontweight='bold'
        )
        axes = axes.flatten()

        for i, node in enumerate(nodes):
            ax = axes[i]
            ax.plot(pred_df.index, pred_df[node], 'b-o', markersize=3,
                    label='Predicted', linewidth=1.5)
            if true_df is not None and node in true_df.columns:
                ax.plot(true_df.index, true_df[node], 'r--o', markersize=3,
                        label='True', linewidth=1.5)
            ax.set_title(f'Node: {node[:20]}', fontsize=9)
            ax.set_ylabel('Speed (mph)')
            ax.tick_params(axis='x', rotation=30, labelsize=7)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Plot saved → {out_path}')

    except ImportError:
        print('matplotlib not available — skipping plot')


def print_metrics(pred_df, true_df):
    """Print MAE/RMSE/MAPE comparing predictions to true values."""
    if true_df is None:
        print('No ground truth available for this time window.')
        return
    pred = torch.FloatTensor(pred_df.values)
    true = torch.FloatTensor(true_df.values)
    print(f'\nMetrics vs ground truth:')
    print(f'  MAE  = {masked_mae(pred, true).item():.4f} mph')
    print(f'  RMSE = {masked_rmse(pred, true).item():.4f} mph')
    print(f'  MAPE = {masked_mape(pred, true).item():.2f}%')


# ── Entry point ────────────────────────────────────────────────────────────────

def main(args):
    # List available dates and exit
    if args.list_dates:
        df = load_speed_df()
        print('\nAvailable date range in dataset:')
        print(f'  Start : {df.index[0]}')
        print(f'  End   : {df.index[-1]}')
        print(f'  Steps : {len(df)} × 5min')
        print('\nExample --start values:')
        for ts in df.index[::288]:    # one per day
            print(f'  "{ts.strftime("%Y-%m-%d %H:%M")}"')
        return

    if not args.checkpoint:
        print('ERROR: --checkpoint is required. Use --list_dates to see available times.')
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load model
    model, scaler, model_cfg = load_model(args.checkpoint, device)

    # Load speed data
    speed_df = load_speed_df()
    start_ts = pd.Timestamp(args.start)
    if start_ts not in speed_df.index:
        # Snap to nearest 5-min slot
        start_ts = speed_df.index[speed_df.index.get_indexer([start_ts], method='nearest')[0]]
        print(f'Snapped to nearest timestamp: {start_ts}')

    print(f'\nPredicting {args.hours}h ahead from {start_ts} ...')

    # Run prediction
    pred_df, true_df = predict_window(
        model, scaler, speed_df, start_ts, args.hours, device,
        seq_len=model_cfg['seq_len']
    )

    print(f'\nPrediction window:')
    print(f'  Input  : {start_ts - pd.Timedelta(minutes=5*model_cfg["seq_len"])}  →  {start_ts}')
    print(f'  Output : {pred_df.index[0]}  →  {pred_df.index[-1]}')
    print(f'  Shape  : {pred_df.shape}  (timesteps × nodes)')
    print(f'\nSample predictions (first 3 nodes, first 6 steps):')
    print(pred_df.iloc[:6, :3].round(2).to_string())

    # Save CSV
    out_stem = os.path.join(
        os.path.dirname(args.checkpoint),
        f'predictions_{start_ts.strftime("%Y-%m-%d_%H-%M")}_{args.hours}h'
    )
    pred_df.to_csv(out_stem + '.csv')
    print(f'\nCSV saved → {out_stem}.csv')

    # Metrics vs ground truth
    print_metrics(pred_df, true_df)

    # Plot
    plot_predictions(pred_df, true_df, start_ts, args.hours, out_stem + '.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DCRNN inference script')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='Path to best_model.pt')
    parser.add_argument('--start', type=str, default='2026-01-25 08:00',
                        help='Start timestamp for prediction (last observed time)')
    parser.add_argument('--hours', type=int, default=2,
                        help='How many hours ahead to predict (default: 2)')
    parser.add_argument('--list_dates', action='store_true',
                        help='Print available timestamps and exit')
    args = parser.parse_args()
    main(args)
