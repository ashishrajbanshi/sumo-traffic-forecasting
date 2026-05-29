#!/usr/bin/env python3
"""
train.py — Train ST-GAT on Chattanooga SUMO data (DeepSUMO style).

Run from gnn/:
    python train.py [--config config.yaml]
"""

import argparse
import os
import pickle
import time
from datetime import datetime

import numpy as np
import torch
import yaml

from lib.dataset  import make_loaders
from lib.metrics  import masked_mae, masked_rmse, masked_mape
from model.stgat  import STGAT


# ── Helpers ────────────────────────────────────────────────────────────────────

def inverse_transform(x, mean, std):
    """Un-normalise speed channel (index 0) only. mean/std: (N, F)."""
    # x: (B, T, N, 1)  — normalised speed
    return x * std[None, None, :, 0:1] + mean[None, None, :, 0:1]


@torch.no_grad()
def evaluate(model, adj, loader, mean, std, device):
    model.eval()
    mae_sum = rmse_sum = mape_sum = n = 0.0
    mean_t = torch.FloatTensor(mean).to(device)
    std_t  = torch.FloatTensor(std).to(device)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x, adj)

        pred_speed = pred * std_t[None, None, :, 0:1] + mean_t[None, None, :, 0:1]
        true_speed = y    * std_t[None, None, :, 0:1] + mean_t[None, None, :, 0:1]

        mae_sum  += masked_mae(pred_speed,  true_speed).item()
        rmse_sum += masked_rmse(pred_speed, true_speed).item()
        mape_sum += masked_mape(pred_speed, true_speed).item()
        n += 1

    return mae_sum / n, rmse_sum / n, mape_sum / n


def plot_metrics(history, log_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('ST-GAT Training — Chattanooga', fontsize=14, fontweight='bold')
        epochs = history['epochs']

        ax = axes[0]
        ax.plot(epochs, history['train_mae'], 'b-o', label='Train', markersize=4)
        ax.plot(epochs, history['val_mae'],   'r-o', label='Val',   markersize=4)
        if history['test_mae_epochs']:
            ax.plot(history['test_mae_epochs'], history['test_mae'], 'g-^', label='Test', markersize=6)
        ax.set_title('MAE (m/s)')
        ax.set_xlabel('Epoch')
        ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(epochs, history['val_rmse'], 'r-o', markersize=4, label='Val RMSE')
        if history['test_rmse_epochs']:
            ax.plot(history['test_rmse_epochs'], history['test_rmse'], 'g-^', markersize=6, label='Test RMSE')
        ax.set_title('RMSE (m/s)')
        ax.set_xlabel('Epoch')
        ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[2]
        ax.plot(epochs, history['val_mape'], 'r-o', markersize=4, label='Val MAPE')
        if history['test_mape_epochs']:
            ax.plot(history['test_mape_epochs'], history['test_mape'], 'g-^', markersize=6, label='Test MAPE')
        ax.set_title('MAPE (%)')
        ax.set_xlabel('Epoch')
        ax.legend(); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out = os.path.join(log_dir, 'training_curves.png')
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Plot saved → {out}')
    except ImportError:
        print('matplotlib not available — skipping plot')


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg  = cfg['data']
    model_cfg = cfg['model']
    train_cfg = cfg['train']

    # ── Log dir ────────────────────────────────────────────────────────────────
    ts       = datetime.now().strftime('%m%d%H%M%S')
    run_name = (f"stgat_h{model_cfg['gat_heads']}"
                f"_d{model_cfg['gat_hidden']}"
                f"_l{model_cfg['gat_layers']}"
                f"_lstm{model_cfg['lstm_hidden']}"
                f"_bs{train_cfg['batch_size']}"
                f"_{ts}")
    log_dir = os.path.join(cfg.get('base_dir', 'data'), 'runs', run_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f'Run: {run_name}')

    # ── Device ─────────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    # ── Graph ──────────────────────────────────────────────────────────────────
    graph_path = os.path.join(data_cfg['output_dir'], 'graph_filtered.pkl')
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)

    adj = torch.FloatTensor(graph['adj_matrix']).to(device)   # (N, N)
    N   = graph['num_nodes']
    print(f'Graph: {N} nodes, {graph["num_edges"]} edges')

    # ── Data ───────────────────────────────────────────────────────────────────
    npz_path = os.path.join(data_cfg['output_dir'], 'dataset.npz')
    train_loader, val_loader, test_loader, meta = make_loaders(
        npz_path        = npz_path,
        n_hist          = model_cfg['n_hist'],
        n_pred          = model_cfg['n_pred'],
        batch_size      = train_cfg['batch_size'],
        val_batch_size  = 64,
        test_batch_size = 64,
    )
    mean = meta['scaler_mean']   # (N, F)
    std  = meta['scaler_std']    # (N, F)
    print(f'Samples — train:{meta["n_train"]}  val:{meta["n_val"]}  test:{meta["n_test"]}')

    # ── Model ──────────────────────────────────────────────────────────────────
    model = STGAT(
        n_nodes     = N,
        n_features  = model_cfg['n_features'],
        n_hist      = model_cfg['n_hist'],
        n_pred      = model_cfg['n_pred'],
        gat_heads   = model_cfg['gat_heads'],
        gat_hidden  = model_cfg['gat_hidden'],
        gat_layers  = model_cfg['gat_layers'],
        lstm_hidden = model_cfg['lstm_hidden'],
        dropout     = model_cfg['dropout'],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {n_params:,}')

    # ── Optimizer ──────────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = train_cfg['lr'],
        weight_decay = train_cfg['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # ── Training config ────────────────────────────────────────────────────────
    epochs      = train_cfg['epochs']
    patience    = train_cfg['patience']
    test_every  = train_cfg['test_every_n_epochs']
    num_batches = len(train_loader)

    best_val_mae = float('inf')
    wait         = 0

    mean_t = torch.FloatTensor(mean).to(device)
    std_t  = torch.FloatTensor(std).to(device)

    history = {
        'epochs':           [],
        'train_mae':        [], 'val_mae':  [], 'val_rmse':  [], 'val_mape':  [],
        'test_mae_epochs':  [], 'test_mae': [],
        'test_rmse_epochs': [], 'test_rmse': [],
        'test_mape_epochs': [], 'test_mape': [],
    }

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        train_mae_sum = train_n = 0

        print(f'\nEpoch {epoch+1}/{epochs}', flush=True)

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            pred = model(x, adj)                           # (B, n_pred, N, 1)
            pred_speed = pred * std_t[None, None, :, 0:1] + mean_t[None, None, :, 0:1]
            true_speed = y    * std_t[None, None, :, 0:1] + mean_t[None, None, :, 0:1]

            loss = masked_mae(pred_speed, true_speed)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_mae_sum += loss.item()
            train_n       += 1

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                elapsed = time.time() - t0
                eta     = elapsed / (batch_idx + 1) * (num_batches - batch_idx - 1)
                print(
                    f'  batch {batch_idx+1:3d}/{num_batches}  '
                    f'loss={train_mae_sum/train_n:.4f}  '
                    f'elapsed={elapsed:.0f}s  eta={eta:.0f}s',
                    flush=True
                )

        # ── Validation ─────────────────────────────────────────────────────────
        val_mae, val_rmse, val_mape = evaluate(model, adj, val_loader, mean, std, device)
        train_mae = train_mae_sum / train_n
        elapsed   = time.time() - t0
        lr_now    = optimizer.param_groups[0]['lr']

        print(
            f'  → val_mae={val_mae:.4f}  val_rmse={val_rmse:.4f}  '
            f'val_mape={val_mape:.2f}%  lr={lr_now:.2e}  {elapsed:.0f}s',
            flush=True
        )

        scheduler.step(val_mae)

        history['epochs'].append(epoch + 1)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        history['val_rmse'].append(val_rmse)
        history['val_mape'].append(val_mape)

        # ── Checkpoint ─────────────────────────────────────────────────────────
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            wait = 0
            ckpt = os.path.join(log_dir, 'best_model.pt')
            torch.save({
                'epoch':        epoch + 1,
                'model_state':  model.state_dict(),
                'optim_state':  optimizer.state_dict(),
                'val_mae':      val_mae,
                'scaler_mean':  mean,
                'scaler_std':   std,
                'model_cfg':    model_cfg,
                'data_cfg':     data_cfg,
                'graph_path':   graph_path,
            }, ckpt)
            print(f'  ✓ best model saved (val_mae={val_mae:.4f})', flush=True)
        else:
            wait += 1
            if wait >= patience:
                print(f'Early stopping at epoch {epoch+1}', flush=True)
                break

        # ── Periodic test ──────────────────────────────────────────────────────
        if (epoch + 1) % test_every == 0:
            test_mae, test_rmse, test_mape = evaluate(model, adj, test_loader, mean, std, device)
            print(f'  [Test] MAE={test_mae:.4f}  RMSE={test_rmse:.4f}  MAPE={test_mape:.2f}%', flush=True)
            history['test_mae_epochs'].append(epoch + 1)
            history['test_mae'].append(test_mae)
            history['test_rmse_epochs'].append(epoch + 1)
            history['test_rmse'].append(test_rmse)
            history['test_mape_epochs'].append(epoch + 1)
            history['test_mape'].append(test_mape)

    # ── Final test ─────────────────────────────────────────────────────────────
    print('\nLoading best model for final test ...', flush=True)
    ckpt_data = torch.load(os.path.join(log_dir, 'best_model.pt'), map_location=device)
    model.load_state_dict(ckpt_data['model_state'])

    test_mae, test_rmse, test_mape = evaluate(model, adj, test_loader, mean, std, device)
    print(f'\nFinal test (best from epoch {ckpt_data["epoch"]}):')
    print(f'  MAE  = {test_mae:.4f} m/s')
    print(f'  RMSE = {test_rmse:.4f} m/s')
    print(f'  MAPE = {test_mape:.2f}%')

    # Record final test
    fe = ckpt_data['epoch']
    if fe not in history['test_mae_epochs']:
        history['test_mae_epochs'].append(fe); history['test_mae'].append(test_mae)
        history['test_rmse_epochs'].append(fe); history['test_rmse'].append(test_rmse)
        history['test_mape_epochs'].append(fe); history['test_mape'].append(test_mape)

    plot_metrics(history, log_dir)
    np.save(os.path.join(log_dir, 'history.npy'), history)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()
    main(args)
