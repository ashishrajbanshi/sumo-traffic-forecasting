#!/usr/bin/env python3
"""
evaluate.py — Classification evaluation of DCRNN predictions.

Traffic states are defined relative to each node's speed limit:
    Free flow  : speed > 75% of speed limit
    Moderate   : 50–75% of speed limit
    Congested  : < 50% of speed limit

Compares DCRNN against a persistence baseline (last known speed repeated).

Run from DCRNN/:
    python evaluate.py --checkpoint data/model/<run>/best_model.pt
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

# ── Constants ──────────────────────────────────────────────────────────────────

CLASSES      = ['Congested', 'Moderate', 'Free Flow']
THRESHOLDS   = [0.60, 0.85]        # fraction of speed limit
NODE_IDS_TXT = 'data/sensor_graph/node_ids.txt'
NODES_CSV    = '../data/graph/nodes.csv'
TEST_NPZ     = 'data/chattanooga/test.npz'


# ── Data helpers ───────────────────────────────────────────────────────────────

def load_node_speed_limits() -> np.ndarray:
    """Return speed limits (mph) aligned to node_ids order."""
    with open(NODE_IDS_TXT) as f:
        node_ids = [x for x in f.read().strip().split(',') if x]

    nodes = pd.read_csv(NODES_CSV)
    nodes = nodes.set_index('node_id')

    limits = np.array([
        nodes.loc[nid, 'speed_limit_ms'] * 2.23694   # m/s → mph
        if nid in nodes.index else 30.0               # fallback: 30 mph
        for nid in node_ids
    ], dtype=np.float32)

    return limits   # (N,)


def load_test_data():
    """Return (x, y) arrays from test.npz, speed channel only."""
    d    = np.load(TEST_NPZ)
    x    = d['x'][:, :, :, 0]   # (S, 12, N) — input speed, normalised
    y    = d['y'][:, :, :, 0]   # (S, 12, N) — target speed, normalised
    return x, y


def denorm(arr: np.ndarray, mean: float, std: float) -> np.ndarray:
    return arr * std + mean


# ── Classification ─────────────────────────────────────────────────────────────

def classify(speed_mph: np.ndarray, limits_mph: np.ndarray) -> np.ndarray:
    """
    Map speed values to traffic state labels.

    Args:
        speed_mph  : any shape (..., N)
        limits_mph : (N,)

    Returns:
        int array same shape, values 0=Congested 1=Moderate 2=Free Flow
    """
    ratio  = speed_mph / limits_mph                    # (..., N)
    labels = np.zeros(speed_mph.shape, dtype=np.int8)
    labels[ratio >= THRESHOLDS[0]] = 1                 # Moderate or better
    labels[ratio >= THRESHOLDS[1]] = 2                 # Free flow
    return labels


# ── Metrics ────────────────────────────────────────────────────────────────────

def confusion_matrix(true: np.ndarray, pred: np.ndarray,
                     n_classes: int = 3) -> np.ndarray:
    """Return (n_classes, n_classes) confusion matrix. Rows=true, cols=pred."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(true.ravel(), pred.ravel()):
        cm[t, p] += 1
    return cm


def class_metrics(cm: np.ndarray):
    """
    Per-class precision, recall, F1 from a confusion matrix.
    Returns dict with arrays of length n_classes.
    """
    n   = cm.shape[0]
    tp  = np.diag(cm).astype(float)
    fp  = cm.sum(axis=0) - tp
    fn  = cm.sum(axis=1) - tp

    precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    recall    = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
    f1        = np.where(precision + recall > 0,
                         2 * precision * recall / (precision + recall), 0.0)
    support   = cm.sum(axis=1)

    # Macro averages (unweighted)
    macro_p  = precision.mean()
    macro_r  = recall.mean()
    macro_f1 = f1.mean()

    # Weighted averages
    total    = support.sum()
    w        = support / total
    w_p      = (precision * w).sum()
    w_r      = (recall    * w).sum()
    w_f1     = (f1        * w).sum()

    accuracy = tp.sum() / total

    return {
        'precision': precision,
        'recall':    recall,
        'f1':        f1,
        'support':   support,
        'macro':     dict(precision=macro_p, recall=macro_r, f1=macro_f1),
        'weighted':  dict(precision=w_p,     recall=w_r,     f1=w_f1),
        'accuracy':  accuracy,
    }


# ── Printing ───────────────────────────────────────────────────────────────────

def print_confusion_matrix(cm: np.ndarray, title: str):
    print(f'\n{title}')
    print(f'  {"":12s}  ' + '  '.join(f'{c:>10s}' for c in CLASSES) + '  (predicted)')
    print('  ' + '-' * (14 + 12 * len(CLASSES)))
    for i, row_label in enumerate(CLASSES):
        row = '  '.join(f'{cm[i, j]:>10,}' for j in range(len(CLASSES)))
        print(f'  {row_label:12s}  {row}')
    print('  (actual)')


def print_report(metrics: dict, title: str):
    print(f'\n{title}')
    print(f'  {"Class":12s}  {"Precision":>10s}  {"Recall":>10s}  {"F1":>10s}  {"Support":>10s}')
    print('  ' + '-' * 58)
    for i, cls in enumerate(CLASSES):
        print(f'  {cls:12s}  '
              f'{metrics["precision"][i]:>10.3f}  '
              f'{metrics["recall"][i]:>10.3f}  '
              f'{metrics["f1"][i]:>10.3f}  '
              f'{metrics["support"][i]:>10,}')
    print('  ' + '-' * 58)
    print(f'  {"Macro avg":12s}  '
          f'{metrics["macro"]["precision"]:>10.3f}  '
          f'{metrics["macro"]["recall"]:>10.3f}  '
          f'{metrics["macro"]["f1"]:>10.3f}')
    print(f'  {"Weighted avg":12s}  '
          f'{metrics["weighted"]["precision"]:>10.3f}  '
          f'{metrics["weighted"]["recall"]:>10.3f}  '
          f'{metrics["weighted"]["f1"]:>10.3f}')
    print(f'\n  Accuracy: {metrics["accuracy"]:.3f}')


def print_comparison(dcrnn_m: dict, persist_m: dict):
    print('\n' + '=' * 60)
    print('COMPARISON — DCRNN vs Persistence Baseline')
    print('=' * 60)
    print(f'  {"Metric":20s}  {"DCRNN":>10s}  {"Persistence":>12s}  {"Winner":>8s}')
    print('  ' + '-' * 56)

    rows = [
        ('Accuracy',       dcrnn_m['accuracy'],                  persist_m['accuracy'],                  True),
        ('Macro F1',        dcrnn_m['macro']['f1'],               persist_m['macro']['f1'],               True),
        ('Macro Precision', dcrnn_m['macro']['precision'],        persist_m['macro']['precision'],        True),
        ('Macro Recall',    dcrnn_m['macro']['recall'],           persist_m['macro']['recall'],           True),
        ('F1 Congested',    dcrnn_m['f1'][0],                     persist_m['f1'][0],                     True),
        ('F1 Moderate',     dcrnn_m['f1'][1],                     persist_m['f1'][1],                     True),
        ('F1 Free Flow',    dcrnn_m['f1'][2],                     persist_m['f1'][2],                     True),
    ]
    for label, d_val, p_val, higher_better in rows:
        winner = 'DCRNN' if (d_val > p_val) == higher_better else 'Baseline'
        print(f'  {label:20s}  {d_val:>10.3f}  {p_val:>12.3f}  {winner:>8s}')


# ── Plot ───────────────────────────────────────────────────────────────────────

def plot_confusion_matrices(cm_dcrnn: np.ndarray, cm_persist: np.ndarray,
                             out_path: str):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Confusion Matrices — Traffic State Classification', fontsize=13)

        for ax, cm, title in zip(
                axes,
                [cm_dcrnn, cm_persist],
                ['DCRNN', 'Persistence Baseline']):
            # Normalise rows for display
            row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
            cm_norm  = cm / row_sums

            im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
            ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES, rotation=15)
            ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES)
            ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
            ax.set_title(title)

            for i in range(len(CLASSES)):
                for j in range(len(CLASSES)):
                    ax.text(j, i,
                            f'{cm[i,j]:,}\n({cm_norm[i,j]:.1%})',
                            ha='center', va='center',
                            color='white' if cm_norm[i, j] > 0.5 else 'black',
                            fontsize=8)
            plt.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'\nConfusion matrix plot saved → {out_path}')
    except ImportError:
        print('matplotlib not available — skipping plot')


def plot_class_distribution(true_labels: np.ndarray, out_path: str):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        counts  = [(true_labels == i).sum() for i in range(len(CLASSES))]
        total   = sum(counts)
        colours = ['#d62728', '#ff7f0e', '#2ca02c']

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(CLASSES, counts, color=colours, edgecolor='white')
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + total * 0.005,
                    f'{count:,}\n({count/total:.1%})',
                    ha='center', fontsize=9)
        ax.set_title('Ground Truth — Traffic State Distribution (Test Set)')
        ax.set_ylabel('Number of (timestep, node) observations')
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Class distribution plot saved → {out_path}')
    except ImportError:
        pass


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Load model & scaler ───────────────────────────────────────────────────
    ckpt      = torch.load(args.checkpoint, map_location=device, weights_only=False)
    mean_mph  = float(ckpt['scaler_mean'])
    std_mph   = float(ckpt['scaler_std'])
    print(f'Model: epoch {ckpt["epoch"]}  val_mae={ckpt["val_mae"]:.4f} mph')
    print(f'Scaler: mean={mean_mph:.2f} mph  std={std_mph:.2f} mph')

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f'\nLoading test set from {TEST_NPZ} ...')
    x_norm, y_norm = load_test_data()          # (S, 12, N) normalised
    S, T_in, N     = x_norm.shape
    T_out          = y_norm.shape[1]
    print(f'  Samples: {S}  |  Nodes: {N}  |  Pred steps: {T_out}')

    print(f'Loading speed limits from {NODES_CSV} ...')
    limits = load_node_speed_limits()          # (N,) mph
    print(f'  Speed limit range: {limits.min():.1f} – {limits.max():.1f} mph')

    # ── Run DCRNN inference ───────────────────────────────────────────────────
    from model.dcrnn_model_pt import DCRNNModel
    from lib.utils_pt import load_graph_data, build_sparse_supports, sparse_to_torch

    graph_pkl              = ckpt['data_cfg']['graph_pkl_filename']
    _, _, adj_mx           = load_graph_data(graph_pkl)
    supports               = [sparse_to_torch(s).to_dense().to(device)
                               for s in build_sparse_supports(
                                   adj_mx, ckpt['model_cfg']['filter_type'])]

    cfg   = ckpt['model_cfg']
    model = DCRNNModel(
        input_dim          = cfg['input_dim'],
        output_dim         = cfg['output_dim'],
        num_units          = cfg['rnn_units'],
        num_nodes          = cfg['num_nodes'],
        supports           = supports,
        max_diffusion_step = cfg['max_diffusion_step'],
        num_layers         = cfg['num_rnn_layers'],
        seq_len            = cfg['seq_len'],
        horizon            = cfg['horizon'],
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    print(f'\nRunning inference on {S} test samples ...')
    batch_size = 64
    all_preds  = []

    with torch.no_grad():
        for start in range(0, S, batch_size):
            end   = min(start + batch_size, S)
            x_b   = torch.FloatTensor(x_norm[start:end]).to(device)  # (B, 12, N)
            x_b   = x_b.unsqueeze(-1)                                 # (B, 12, N, 1)
            # Append zero time-of-day for inference (model expects 2 features)
            tod   = torch.zeros_like(x_b)
            x_b   = torch.cat([x_b, tod], dim=-1)                    # (B, 12, N, 2)

            pred  = model(x_b)                                         # (B, 12, N, 1)
            all_preds.append(pred[:, :, :, 0].cpu().numpy())

            if (start // batch_size + 1) % 5 == 0:
                print(f'  batch {start//batch_size+1}/{(S+batch_size-1)//batch_size}')

    pred_norm = np.concatenate(all_preds, axis=0)   # (S, 12, N) normalised

    # ── Denormalise ───────────────────────────────────────────────────────────
    pred_mph  = denorm(pred_norm,          mean_mph, std_mph)   # (S, T_out, N)
    true_mph  = denorm(y_norm,             mean_mph, std_mph)   # (S, T_out, N)
    last_mph  = denorm(x_norm[:, -1, :],   mean_mph, std_mph)   # (S, N) last input step

    # ── Persistence baseline: repeat last observation ─────────────────────────
    persist_mph = np.tile(last_mph[:, np.newaxis, :],
                          (1, T_out, 1))                         # (S, T_out, N)

    # ── Classify ──────────────────────────────────────────────────────────────
    # limits: (N,) → broadcast over (S, T_out, N)
    true_labels    = classify(true_mph,    limits)   # (S, T_out, N)
    pred_labels    = classify(pred_mph,    limits)
    persist_labels = classify(persist_mph, limits)

    print(f'\nClass distribution in ground truth:')
    total = true_labels.size
    for i, cls in enumerate(CLASSES):
        cnt = (true_labels == i).sum()
        print(f'  {cls:12s}: {cnt:>10,}  ({cnt/total:.1%})')

    # ── Confusion matrices ────────────────────────────────────────────────────
    cm_dcrnn   = confusion_matrix(true_labels, pred_labels)
    cm_persist = confusion_matrix(true_labels, persist_labels)

    print_confusion_matrix(cm_dcrnn,   'DCRNN — Confusion Matrix')
    print_confusion_matrix(cm_persist, 'Persistence — Confusion Matrix')

    # ── Classification reports ────────────────────────────────────────────────
    m_dcrnn   = class_metrics(cm_dcrnn)
    m_persist = class_metrics(cm_persist)

    print_report(m_dcrnn,   'DCRNN — Classification Report')
    print_report(m_persist, 'Persistence — Classification Report')

    # ── Side-by-side comparison ───────────────────────────────────────────────
    print_comparison(m_dcrnn, m_persist)

    # ── Save results ──────────────────────────────────────────────────────────
    out_dir = os.path.dirname(args.checkpoint)

    rows = []
    for i, cls in enumerate(CLASSES):
        rows.append({
            'class':                cls,
            'dcrnn_precision':      round(m_dcrnn['precision'][i], 4),
            'dcrnn_recall':         round(m_dcrnn['recall'][i], 4),
            'dcrnn_f1':             round(m_dcrnn['f1'][i], 4),
            'persist_precision':    round(m_persist['precision'][i], 4),
            'persist_recall':       round(m_persist['recall'][i], 4),
            'persist_f1':           round(m_persist['f1'][i], 4),
            'support':              int(m_dcrnn['support'][i]),
        })
    for avg in ('macro', 'weighted'):
        rows.append({
            'class':                avg + '_avg',
            'dcrnn_precision':      round(m_dcrnn[avg]['precision'], 4),
            'dcrnn_recall':         round(m_dcrnn[avg]['recall'], 4),
            'dcrnn_f1':             round(m_dcrnn[avg]['f1'], 4),
            'persist_precision':    round(m_persist[avg]['precision'], 4),
            'persist_recall':       round(m_persist[avg]['recall'], 4),
            'persist_f1':           round(m_persist[avg]['f1'], 4),
            'support':              '',
        })

    csv_path = os.path.join(out_dir, 'classification_report.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f'\nReport saved → {csv_path}')

    plot_confusion_matrices(
        cm_dcrnn, cm_persist,
        os.path.join(out_dir, 'confusion_matrices.png')
    )
    plot_class_distribution(
        true_labels,
        os.path.join(out_dir, 'class_distribution.png')
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DCRNN classification evaluation')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to best_model.pt')
    args = parser.parse_args()
    main(args)
