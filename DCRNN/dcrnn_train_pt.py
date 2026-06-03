#!/usr/bin/env python3
"""
dcrnn_train_pt.py — PyTorch training entry point for DCRNN (Chattanooga).

Run from sumo_3/DCRNN/:
    python dcrnn_train_pt.py --config_filename data/model/dcrnn_chattanooga.yaml
"""

import argparse
import math
import os
import time

import numpy as np
import torch
import yaml

from lib.utils_pt   import load_dataset, load_graph_data, build_sparse_supports, \
                           sparse_to_torch, get_logger
from lib.metrics_pt import masked_mae, masked_rmse, masked_mape
from model.dcrnn_model_pt import DCRNNModel


# ── Curriculum learning schedule ───────────────────────────────────────────────

def _cl_ratio(global_step, cl_decay_steps):
    """Probability of using ground truth at decoder step t (inverse exponential)."""
    return max(cl_decay_steps / (cl_decay_steps + math.exp(global_step / cl_decay_steps)), 0.0)


# ── Evaluation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, scaler, device, output_dim):
    model.eval()
    mae_sum = rmse_sum = mape_sum = n = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        pred_speed = scaler.inverse_transform(pred[..., 0])
        true_speed = scaler.inverse_transform(y[..., 0])
        mae_sum  += masked_mae(pred_speed,  true_speed).item()
        rmse_sum += masked_rmse(pred_speed, true_speed).item()
        mape_sum += masked_mape(pred_speed, true_speed).item()
        n += 1
    return mae_sum / n, rmse_sum / n, mape_sum / n


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_metrics(history, log_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')          # non-interactive backend — works without a display
        import matplotlib.pyplot as plt

        epochs = history['epochs']

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('DCRNN Training — Chattanooga', fontsize=14, fontweight='bold')

        # ── MAE ──
        ax = axes[0]
        ax.plot(epochs, history['train_mae'], 'b-o', label='Train MAE', markersize=4)
        ax.plot(epochs, history['val_mae'],   'r-o', label='Val MAE',   markersize=4)
        if history['test_mae_epochs']:
            ax.plot(history['test_mae_epochs'], history['test_mae'],
                    'g-^', label='Test MAE', markersize=6, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MAE (mph)')
        ax.set_title('Mean Absolute Error')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ── RMSE ──
        ax = axes[1]
        ax.plot(epochs, history['val_rmse'], 'r-o', label='Val RMSE', markersize=4)
        if history['test_rmse_epochs']:
            ax.plot(history['test_rmse_epochs'], history['test_rmse'],
                    'g-^', label='Test RMSE', markersize=6, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE (mph)')
        ax.set_title('Root Mean Squared Error')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ── MAPE ──
        ax = axes[2]
        ax.plot(epochs, history['val_mape'], 'r-o', label='Val MAPE', markersize=4)
        if history['test_mape_epochs']:
            ax.plot(history['test_mape_epochs'], history['test_mape'],
                    'g-^', label='Test MAPE', markersize=6, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MAPE (%)')
        ax.set_title('Mean Absolute Percentage Error')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(log_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'\nPlot saved → {plot_path}')

    except ImportError:
        print('\nmatplotlib not installed — skipping plot. Install with: pip install matplotlib')


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    with open(args.config_filename) as f:
        config = yaml.safe_load(f)

    data_cfg  = config['data']
    model_cfg = config['model']
    train_cfg = config['train']

    # ── Log directory ──────────────────────────────────────────────────────────
    from datetime import datetime
    ts = datetime.now().strftime('%m%d%H%M%S')
    run_name = (f"dcrnn_pt_DR_{model_cfg['max_diffusion_step']}"
                f"_h_{model_cfg['horizon']}"
                f"_{model_cfg['rnn_units']}-{model_cfg['rnn_units']}"
                f"_lr_{train_cfg['base_lr']}"
                f"_bs_{data_cfg['batch_size']}_{ts}")
    log_dir = os.path.join(config.get('base_dir', 'data/model'), run_name)
    os.makedirs(log_dir, exist_ok=True)

    logger = get_logger(log_dir, __name__, 'info.log')
    logger.info(config)

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        logger.info(f'GPU: {torch.cuda.get_device_name(0)}')

    use_amp = (device.type == 'cuda')
    amp_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── Data ──────────────────────────────────────────────────────────────────
    data = load_dataset(
        dataset_dir    = data_cfg['dataset_dir'],
        batch_size     = data_cfg['batch_size'],
        val_batch_size = data_cfg['val_batch_size'],
        test_batch_size= data_cfg['test_batch_size'],
        num_workers    = args.num_workers,
    )
    scaler = data['scaler']
    logger.info(f"x_train: {data['x_train'].shape}  y_train: {data['y_train'].shape}")
    logger.info(f"x_val:   {data['x_val'].shape}    y_val:   {data['y_val'].shape}")
    logger.info(f"x_test:  {data['x_test'].shape}   y_test:  {data['y_test'].shape}")

    # ── Graph ─────────────────────────────────────────────────────────────────
    _, _, adj_mx = load_graph_data(data_cfg['graph_pkl_filename'])
    scipy_supports = build_sparse_supports(adj_mx, model_cfg['filter_type'])
    supports = [sparse_to_torch(s).to_dense().to(device) for s in scipy_supports]
    logger.info(f'Support matrices: {len(supports)} dense tensors, '
                f'shape={list(supports[0].shape)}, device={supports[0].device}')

    # ── Model ─────────────────────────────────────────────────────────────────
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

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Trainable parameters: {n_params:,}')

    # ── Optimizer & LR schedule ───────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr  = train_cfg['base_lr'],
        eps = float(train_cfg['epsilon']),
    )
    lr_steps  = train_cfg.get('steps', [20, 30, 40, 50])
    lr_decay  = train_cfg.get('lr_decay_ratio', 0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=lr_steps, gamma=lr_decay
    )

    # ── Training config ───────────────────────────────────────────────────────
    epochs         = train_cfg['epochs']
    patience       = train_cfg.get('patience', 50)
    test_every     = train_cfg.get('test_every_n_epochs', 10)
    cl_decay_steps = model_cfg.get('cl_decay_steps', 2000)
    use_curriculum = model_cfg.get('use_curriculum_learning', True)
    output_dim     = model_cfg['output_dim']
    max_grad_norm  = train_cfg.get('max_grad_norm', 5.0)

    best_val_mae = float('inf')
    wait         = 0
    global_step  = 0
    num_batches  = len(data['train_loader'])

    # ── Metric history for plotting ───────────────────────────────────────────
    history = {
        'epochs':           [],
        'train_mae':        [],
        'val_mae':          [],
        'val_rmse':         [],
        'val_mape':         [],
        'test_mae_epochs':  [],
        'test_mae':         [],
        'test_rmse_epochs': [],
        'test_rmse':        [],
        'test_mape_epochs': [],
        'test_mape':        [],
    }

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        train_mae_sum = train_n = 0

        print(f"\nEpoch {epoch+1}/{epochs}", flush=True)

        for batch_idx, (x_batch, y_batch) in enumerate(data['train_loader']):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            cl_ratio = _cl_ratio(global_step, cl_decay_steps) if use_curriculum else 0.0

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(x_batch, y_batch, teacher_forcing_ratio=cl_ratio)
                loss = masked_mae(pred[..., 0], y_batch[..., 0])

            optimizer.zero_grad()
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            amp_scaler.step(optimizer)
            amp_scaler.update()

            train_mae_sum += loss.item()
            train_n       += 1
            global_step   += 1

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                elapsed_so_far = time.time() - t0
                secs_per_batch = elapsed_so_far / (batch_idx + 1)
                remaining      = secs_per_batch * (num_batches - batch_idx - 1)
                print(
                    f"  batch {batch_idx+1:3d}/{num_batches}  "
                    f"loss={train_mae_sum/train_n:.4f}  "
                    f"elapsed={elapsed_so_far:.0f}s  "
                    f"eta={remaining:.0f}s",
                    flush=True
                )

        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────────
        val_mae, val_rmse, val_mape = evaluate(
            model, data['val_loader'], scaler, device, output_dim
        )
        elapsed    = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']
        train_mae  = train_mae_sum / train_n

        logger.info(
            f"Epoch [{epoch+1}/{epochs}] ({global_step})  "
            f"train_mae: {train_mae:.4f}  "
            f"val_mae: {val_mae:.4f}  val_rmse: {val_rmse:.4f}  val_mape: {val_mape:.2f}%  "
            f"lr: {current_lr:.6f}  {elapsed:.1f}s"
        )
        print(
            f"  → val_mae={val_mae:.4f} mph  val_rmse={val_rmse:.4f}  "
            f"val_mape={val_mape:.2f}%  lr={current_lr}",
            flush=True
        )

        # Record for plot
        history['epochs'].append(epoch + 1)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        history['val_rmse'].append(val_rmse)
        history['val_mape'].append(val_mape)

        # ── Checkpoint ────────────────────────────────────────────────────────
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            wait = 0
            ckpt = os.path.join(log_dir, 'best_model.pt')
            torch.save({
                'epoch':        epoch + 1,
                'global_step':  global_step,
                'model_state':  model.state_dict(),
                'optim_state':  optimizer.state_dict(),
                'val_mae':      val_mae,
                'scaler_mean':  scaler.mean,
                'scaler_std':   scaler.std,
                'model_cfg':    model_cfg,
                'data_cfg':     data_cfg,
            }, ckpt)
            logger.info(f'  Val MAE improved → {val_mae:.4f}  saved to {ckpt}')
            print(f'  ✓ best model saved (val_mae={val_mae:.4f})', flush=True)
        else:
            wait += 1
            if wait >= patience:
                logger.info(f'Early stopping at epoch {epoch+1}')
                print(f'Early stopping at epoch {epoch+1}', flush=True)
                break

        # ── Test evaluation every N epochs ────────────────────────────────────
        if (epoch + 1) % test_every == 0:
            test_mae, test_rmse, test_mape = evaluate(
                model, data['test_loader'], scaler, device, output_dim
            )
            logger.info(
                f'  [Test @ epoch {epoch+1}]  '
                f'MAE: {test_mae:.4f}  RMSE: {test_rmse:.4f}  MAPE: {test_mape:.2f}%'
            )
            print(
                f'  [Test] MAE={test_mae:.4f} mph  RMSE={test_rmse:.4f}  MAPE={test_mape:.2f}%',
                flush=True
            )
            history['test_mae_epochs'].append(epoch + 1)
            history['test_mae'].append(test_mae)
            history['test_rmse_epochs'].append(epoch + 1)
            history['test_rmse'].append(test_rmse)
            history['test_mape_epochs'].append(epoch + 1)
            history['test_mape'].append(test_mape)

    # ── Final test ────────────────────────────────────────────────────────────
    print('\nLoading best model for final test ...', flush=True)
    ckpt_data = torch.load(os.path.join(log_dir, 'best_model.pt'), map_location=device)
    model.load_state_dict(ckpt_data['model_state'])

    test_mae, test_rmse, test_mape = evaluate(
        model, data['test_loader'], scaler, device, output_dim
    )
    logger.info(
        f'Final test — MAE: {test_mae:.4f}  RMSE: {test_rmse:.4f}  MAPE: {test_mape:.2f}%'
    )
    print(
        f'\nFinal test results (best model from epoch {ckpt_data["epoch"]}):',
        f'\n  MAE  = {test_mae:.4f} mph',
        f'\n  RMSE = {test_rmse:.4f} mph',
        f'\n  MAPE = {test_mape:.2f}%',
        flush=True
    )

    # ── Save plot ─────────────────────────────────────────────────────────────
    # Add final test point to history if not already recorded
    final_epoch = ckpt_data['epoch']
    if final_epoch not in history['test_mae_epochs']:
        history['test_mae_epochs'].append(final_epoch)
        history['test_mae'].append(test_mae)
        history['test_rmse_epochs'].append(final_epoch)
        history['test_rmse'].append(test_rmse)
        history['test_mape_epochs'].append(final_epoch)
        history['test_mape'].append(test_mape)

    plot_metrics(history, log_dir)

    # Save raw history as numpy for later re-plotting
    np.save(os.path.join(log_dir, 'history.npy'), history)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', type=str,
                        default='data/model/dcrnn_chattanooga.yaml')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader worker processes (default: 4)')
    args = parser.parse_args()
    main(args)
