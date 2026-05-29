# DCRNN — Chattanooga Traffic Forecasting

Diffusion Convolutional Recurrent Neural Network trained on 30-day SUMO simulation data
for Chattanooga, TN (January 2026). Predicts speed 1 hour ahead (12 × 5-min steps) across
1174 road segments.

---

## Directory structure

```
DCRNN/
├── data/
│   ├── sensor_graph/
│   │   ├── node_ids.txt                  # 1174 clean node IDs
│   │   ├── distances_chattanooga.csv     # 297,588 directed road-network distances
│   │   └── adj_mx_chattanooga.pkl        # 1174×1174 weighted adjacency matrix
│   ├── chattanooga/
│   │   ├── train.npz                     # 6032 sliding-window samples (70%)
│   │   ├── val.npz                       # 862 samples (10%)
│   │   └── test.npz                      # 1723 samples (20%)
│   ├── model/
│   │   └── dcrnn_chattanooga.yaml        # model + training hyperparameters
│   └── traffic_chattanooga.h5            # 8640 × 1174 speed matrix (mph, 5-min)
├── scripts/
│   ├── prepare_data.py                   # Step 1 — filter nodes, extract speed HDF5
│   ├── gen_adj_mx.py                     # Step 2 — build adjacency matrix
│   └── generate_training_data.py         # Step 3 — create train/val/test splits
├── model/
│   ├── dcrnn_cell_pt.py                  # DCGRUCell (PyTorch)
│   └── dcrnn_model_pt.py                 # Encoder-decoder model (PyTorch)
├── lib/
│   ├── utils_pt.py                       # Data loading, graph math, scaler
│   └── metrics_pt.py                     # Masked MAE / RMSE / MAPE
├── dcrnn_train_pt.py                     # Step 4 — PyTorch training entry point
└── README.md
```

---

## One-time environment setup

Requires Python 3.12 (already available at `/usr/bin/python3.12`).

```bash
# Create virtual environment
python3.12 -m venv venv_dcrnn_pt

# Activate it
source venv_dcrnn_pt/bin/activate

# Install PyTorch (nightly, CUDA 12.8 — matches RTX PRO 500 Blackwell GPU)
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

# Install remaining dependencies
pip install scipy pyyaml

# Verify GPU is visible
python -c "import torch; print(torch.__version__, '| CUDA:', torch.cuda.is_available())"
```

> Steps 1–3 (data preparation) do **not** need the venv — they use the system `python3.12`
> which already has pandas, numpy, and tables installed.

---

## Files to delete before a fresh run

Run from `sumo_3/DCRNN/`:

```bash
rm -f data/sensor_graph/node_ids.txt
rm -f data/sensor_graph/distances_chattanooga.csv
rm -f data/sensor_graph/adj_mx_chattanooga.pkl
rm -f data/chattanooga/train.npz data/chattanooga/val.npz data/chattanooga/test.npz
rm -f data/traffic_chattanooga.h5
rm -rf data/model/dcrnn_pt_*/
```

---

## Run sequence

All commands run from `sumo_3/DCRNN/`.

### Step 1 — Prepare data

Reads from `sumo_3/data/`. Filters 9 bad nodes (2 isolated + 7 zero-volume),
extracts speed signal, writes Chattanooga-specific files.

```bash
python3.12 scripts/prepare_data.py
```

Generates:
- `data/sensor_graph/node_ids.txt`
- `data/sensor_graph/distances_chattanooga.csv`
- `data/traffic_chattanooga.h5`

---

### Step 2 — Build adjacency matrix

Applies a Gaussian kernel to road-network distances (σ = std of all distances ≈ 490 m,
threshold 0.1) to produce a sparse weighted graph.

```bash
python3.12 scripts/gen_adj_mx.py \
    --sensor_ids_filename data/sensor_graph/node_ids.txt \
    --distances_filename  data/sensor_graph/distances_chattanooga.csv \
    --output_pkl_filename data/sensor_graph/adj_mx_chattanooga.pkl
```

Generates:
- `data/sensor_graph/adj_mx_chattanooga.pkl`

---

### Step 3 — Generate train / val / test splits

Slides a 12-step input → 12-step output window over all 8640 timesteps.
Adds time-of-day as a second input feature.

```bash
python3.12 scripts/generate_training_data.py \
    --traffic_df_filename data/traffic_chattanooga.h5 \
    --output_dir          data/chattanooga/
```

Generates:
- `data/chattanooga/train.npz` — shape (6032, 12, 1174, 2)
- `data/chattanooga/val.npz`   — shape (862,  12, 1174, 2)
- `data/chattanooga/test.npz`  — shape (1723, 12, 1174, 2)

---

### Step 4 — Train DCRNN

Activate the venv first, then run training.

```bash
source venv_dcrnn_pt/bin/activate

python dcrnn_train_pt.py \
    --config_filename data/model/dcrnn_chattanooga.yaml
```

Generates a timestamped run folder under `data/model/`, for example:
```
data/model/dcrnn_pt_DR_2_h_12_64-64_lr_0.01_bs_64_MMDDHHMMSS/
    info.log          # epoch-by-epoch MAE / RMSE / MAPE log
    best_model.pt     # best checkpoint (lowest val MAE)
```

---

## Changing hyperparameters

Edit `data/model/dcrnn_chattanooga.yaml` before running Step 4.

| Setting | Location in YAML | Default |
|---|---|---|
| Number of epochs | `train.epochs` | `100` |
| Early stopping patience | `train.patience` | `50` |
| Test evaluation frequency | `train.test_every_n_epochs` | `10` |
| Batch size | `data.batch_size` | `64` |
| Hidden units per node | `model.rnn_units` | `64` |
| Diffusion hops | `model.max_diffusion_step` | `2` |
| Learning rate | `train.base_lr` | `0.01` |

Each training run creates a **new** timestamped folder — previous runs are never overwritten.

---

## Input data source

The `traffic_chattanooga.h5` is derived from a 30-day SUMO mesoscopic simulation:

```
sumo_3/data/traffic/traffic_jan2026.h5   (8640 × 1183, /speed /volume /occupancy /flow)
    ↓  prepare_data.py  (removes 9 bad nodes, extracts /speed)
data/traffic_chattanooga.h5              (8640 × 1174, speed in mph)
```

The adjacency matrix is derived from road-network distances computed by Dijkstra
through the SUMO network (`sumo_3/data/graph/edges.csv`, max radius 2000 m).
