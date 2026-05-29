# Traffic Forecasting on Chattanooga SUMO Network

Deep learning traffic prediction using DCRNN and ST-GAT (DeepSUMO) on a
SUMO-simulated road network of Chattanooga, TN.

---

## Requirements

- Python 3.10+
- [SUMO](https://sumo.dlr.de/) installed and on PATH (`sumo`, `duarouter`)
- NVIDIA GPU (6 GB+ VRAM recommended)

```bash
pip install -r requirements.txt
```

> **Blackwell GPU (RTX 5000 series):** PyTorch nightly is required.
> ```bash
> pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
> ```

---

## Project Structure

```
sumo_3/
├── config/          # SUMO config files (.sumocfg)
├── network/         # Road network (osm.net.xml)
├── detectors/       # Loop detector definitions (.add.xml)
├── routes/          # Vehicle route files + generation script
├── output/          # Simulation output (detector XML, ignored by git)
├── script/          # Data processing pipeline scripts
├── data/            # Processed graph and traffic data
├── DCRNN/           # DCRNN model implementation
└── gnn/             # ST-GAT (DeepSUMO) model implementation
```

---

## Step 1 — Generate Routes

Generate daily route files for January 2026 using DUARouter:

```bash
cd routes
python generate_january_2026.py
```

Output: `routes/january_2026/routes_2026-01-XX.rou.xml` (one per day)

---

## Step 2 — Run SUMO Simulation

Run the full month simulation (January 2026, 30 days):

```bash
python script/run_simulation.py
```

This runs SUMO headlessly for each day using
`config/sumo_chattanooga.sumocfg` and saves detector output to
`output/jan2026/YYYY-MM-DD/`.

> Each day takes ~5–15 minutes. Full month ~4–6 hours.

---

## Step 3 — Build Sensor Graph

### 3a. Aggregate detectors into nodes (lane merging)

```bash
python script/build_sensor_locations.py \
    --net    network/osm.net.xml \
    --dets   detectors/detectors_weekday.add.xml \
    --jan2026-dir output/jan2026 \
    --out    data/graph/nodes.csv
```

Merges per-lane detectors into one node per road segment
(speed averaged, volume summed across lanes).
Produces `data/graph/nodes.csv` with 1,183 nodes.

### 3b. Compute road-network distances (Dijkstra)

```bash
python script/build_distances.py \
    --net  network/osm.net.xml \
    --nodes data/graph/nodes.csv \
    --out   data/graph/edges.csv \
    --max-dist 2000
```

Computes shortest-path distances between all detector pairs within 2,000 m.
Produces `data/graph/edges.csv`.

---

## Step 4 — Parse Detector Output → HDF5

```bash
python script/parse_detector_output.py \
    --input-dir output/jan2026 \
    --nodes     data/graph/nodes.csv \
    --out       data/traffic/traffic_jan2026.h5
```

Reads raw SUMO XML detector files and assembles a single HDF5 file with
keys `speed`, `occupancy`, `volume` — shape `(8640 timesteps × N nodes)`.

---

## Step 5 — DCRNN

### 5a. Prepare graph adjacency matrix

```bash
cd DCRNN
python scripts/prepare_data.py \
    --nodes_csv  ../data/graph/nodes.csv \
    --edges_csv  ../data/graph/edges.csv \
    --output_pkl data/sensor_graph/adj_mx_chattanooga.pkl
```

### 5b. Generate training arrays

```bash
python scripts/generate_training_data.py \
    --traffic_h5    ../data/traffic/traffic_jan2026.h5 \
    --graph_pkl     data/sensor_graph/adj_mx_chattanooga.pkl \
    --output_dir    data/chattanooga
```

### 5c. Train

```bash
python dcrnn_train_pt.py --config_filename data/model/dcrnn_chattanooga.yaml
```

Checkpoints saved to `data/model/dcrnn_pt_<run>/best_model.pt`.

### 5d. Predict

```bash
python predict.py \
    --checkpoint data/model/dcrnn_pt_<run>/best_model.pt \
    --start "2026-01-25 08:00" \
    --hours 2
```

---

## Step 6 — ST-GAT (DeepSUMO)

### 6a. Build graph

```bash
cd gnn
python scripts/build_graph.py --threshold 500
```

Connects nodes within 500 m road-network distance.
Output: `gnn/data/graph.pkl`

### 6b. Prepare dataset

```bash
python scripts/prepare_data.py
```

Applies variance filtering (removes constant-speed nodes), normalises
features, creates sliding-window train/val/test arrays.
Output: `gnn/data/graph_filtered.pkl`, `gnn/data/dataset.npz`

### 6c. Train

```bash
python train.py
```

Checkpoints saved to `gnn/data/runs/<run>/best_model.pt`.
Training curves saved as `training_curves.png`.

### 6d. Predict

```bash
python predict.py \
    --checkpoint data/runs/<run>/best_model.pt \
    --start "2026-01-25 08:00" \
    --hours 2
```

---

## Configuration

| File | Purpose |
|------|---------|
| `config/sumo_chattanooga.sumocfg` | SUMO simulation config |
| `DCRNN/data/model/dcrnn_chattanooga.yaml` | DCRNN hyperparameters |
| `gnn/config.yaml` | ST-GAT hyperparameters |
