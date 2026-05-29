# SUMO → Graph Dataset Pipeline Architecture

## Overview

Full pipeline for transforming a SUMO traffic simulation of Chattanooga, TN
into a model-agnostic graph dataset for traffic speed forecasting.

The pipeline produces three self-contained artefacts:

| File | Description |
|---|---|
| `data/graph/nodes.csv` | Node positions (lat/lon) — one row per edge |
| `data/graph/edges.csv` | Directed road-network distances — all reachable pairs |
| `data/traffic/speed_jan2026.h5` | (time × nodes) speed matrix — January 2026 |

These three files are sufficient to build the adjacency matrix and training
windows for any GNN or STGNN model (DCRNN, GraphWaveNet, STGCN, etc.).

---

## Directory layout

```
sumo_3/
├── config/
│   └── sumo_chattanooga*.sumocfg        # SUMO simulation configs
├── network/
│   ├── osm.net.xml[.gz]                 # Road network
│   └── osm.poly.xml.gz                  # Polygons (optional)
├── routes/
│   └── january_2026/                    # Per-day route files ✓ COMPLETE
│       ├── trips_2026-01-DD.xml         # randomTrips OD demand
│       └── routes_2026-01-DD.rou.xml    # duarouter output
├── detectors/
│   ├── detectors_weekday.add.xml        # E1 midpoint detectors — weekday ✓
│   ├── detectors_weekend.add.xml        # E1 midpoint detectors — weekend ✓
│   ├── detectors_filtered_weekday.add.xml   # post-variance-filter
│   └── detectors_filtered_weekend.add.xml   # post-variance-filter
├── output/
│   └── jan2026/
│       └── YYYY-MM-DD/                  # one dir per day ← run_simulation.py
│           ├── detector_output.xml      # E1 speed/flow (5-min intervals)
│           ├── queue_output.xml         # per-edge queue lengths
│           ├── tripinfos.xml            # per-vehicle stats
│           └── stats.xml               # simulation summary
├── script/
│   ├── generate_detectors.py            # Phase 0b — midpoint placement ✓
│   ├── build_routes.py                  # Phase 0a — January 2026 demand ✓
│   ├── run_simulation.py                # Phase 0d — 30-day SUMO runs
│   ├── filter_detectors.py             # Phase 0c — variance filter
│   ├── parse_detector_output.py         # Phase 1  — XML → HDF5
│   ├── build_sensor_locations.py        # Phase 2  — nodes.csv
│   └── build_distances.py              # Phase 3  — edges.csv
└── data/
    ├── graph/
    │   ├── nodes.csv                    # Phase 2 output
    │   ├── node_ids.txt                 # Phase 2 output
    │   └── edges.csv                   # Phase 3 output
    └── traffic/
        └── speed_jan2026.h5             # Phase 1 output
```

---

## Phase 0 — Simulation Setup

### 0a — January 2026 Route Files ✓ COMPLETE

**Script:** `script/build_routes.py`

Generates 30 days of demand and routes covering January 1–30, 2026 using
SUMO's `randomTrips.py` + `duarouter`.

**Demand parameters:**

| Day type | Period range | Vehicles/day |
|---|---|---|
| Weekday (Mon–Fri) | 1.60 – 1.65 s (random per day, seed=42) | ~52,500 – 54,000 |
| Weekend (Sat–Sun) | 1.83 – 1.88 s (random per day, seed=42) | ~46,000 – 47,200 |

**January 2026 — all 30 days:**

| Date | Day | Period (s) | ~Vehicles |
|---|---|---|---|
| 2026-01-01 | Thursday  | 1.6320 | 52,941 |
| 2026-01-02 | Friday    | 1.6013 | 53,956 |
| 2026-01-03 | Saturday  | 1.8438 | 46,859 |
| 2026-01-04 | Sunday    | 1.8412 | 46,925 |
| 2026-01-05 | Monday    | 1.6368 | 52,785 |
| 2026-01-06 | Tuesday   | 1.6338 | 52,882 |
| 2026-01-07 | Wednesday | 1.6446 | 52,535 |
| 2026-01-08 | Thursday  | 1.6043 | 53,855 |
| 2026-01-09 | Friday    | 1.6211 | 53,297 |
| 2026-01-10 | Saturday  | 1.8315 | 47,174 |
| 2026-01-11 | Sunday    | 1.8409 | 46,933 |
| 2026-01-12 | Monday    | 1.6253 | 53,159 |
| 2026-01-13 | Tuesday   | 1.6013 | 53,956 |
| 2026-01-14 | Wednesday | 1.6099 | 53,667 |
| 2026-01-15 | Thursday  | 1.6325 | 52,924 |
| 2026-01-16 | Friday    | 1.6272 | 53,097 |
| 2026-01-17 | Saturday  | 1.8410 | 46,931 |
| 2026-01-18 | Sunday    | 1.8595 | 46,464 |
| 2026-01-19 | Monday    | 1.6405 | 52,666 |
| 2026-01-20 | Tuesday   | 1.6003 | 53,989 |
| 2026-01-21 | Wednesday | 1.6403 | 52,673 |
| 2026-01-22 | Thursday  | 1.6349 | 52,847 |
| 2026-01-23 | Friday    | 1.6170 | 53,432 |
| 2026-01-24 | Saturday  | 1.8378 | 47,012 |
| 2026-01-25 | Sunday    | 1.8779 | 46,008 |
| 2026-01-26 | Monday    | 1.6168 | 53,438 |
| 2026-01-27 | Tuesday   | 1.6046 | 53,845 |
| 2026-01-28 | Wednesday | 1.6048 | 53,838 |
| 2026-01-29 | Thursday  | 1.6424 | 52,605 |
| 2026-01-30 | Friday    | 1.6302 | 52,999 |

```bash
python3 script/build_routes.py
```

### 0b — Detector Placement ✓ COMPLETE

**Script:** `script/generate_detectors.py`

**Strategy: one detector per lane at the lane midpoint.**

```
Edge with 3 lanes, 300 m:
  Lane 0  ──────────●──────────  pos = 150.0 m
  Lane 1  ──────────●──────────  pos = 150.0 m
  Lane 2  ──────────●──────────  pos = 150.0 m
```

All lanes on the same edge have identical length → `pos = lane_len / 2` is
the same across all lanes. The three sensors sit at the same GPS coordinate
and collapse to a single graph node (averaged speed) in Phase 1.

**Result:** 2,123 detectors on motorway, trunk, primary, and secondary roads.
Both `detectors_weekday.add.xml` and `detectors_weekend.add.xml` contain the
same detector set; they differ only in the `file=` output attribute.

```bash
python3 script/generate_detectors.py
```

### 0c — Variance Filter

Run a short trial simulation per day type, then discard detectors whose
speed variance is below threshold (constant speed → not useful for training).

```bash
sumo -c config/sumo_chattanooga_weekdays.sumocfg --mesosim true --end 3600
sumo -c config/sumo_chattanooga.sumocfg          --mesosim true --end 3600

python3 script/filter_detectors.py --day-type weekday --threshold 1.0
python3 script/filter_detectors.py --day-type weekend --threshold 1.0
```

Produces `detectors_filtered_weekday.add.xml` / `detectors_filtered_weekend.add.xml`.
All downstream scripts fall back to the unfiltered files if these don't exist yet.

### 0d — Full 30-Day SUMO Runs

**Script:** `script/run_simulation.py`

Runs SUMO mesosim once per January day. For each day:
- Selects the correct weekday/weekend detector `.add.xml`
- Uses that day's `routes/january_2026/routes_YYYY-MM-DD.rou.xml`
- Writes 4 output files to `output/jan2026/YYYY-MM-DD/`

| Output file | Contents |
|---|---|
| `detector_output.xml` | E1 speed + flow at every sensor, 5-min intervals |
| `queue_output.xml` | Per-edge queue lengths throughout the day |
| `tripinfos.xml` | Per-vehicle depart / arrive / route stats |
| `stats.xml` | Simulation-wide summary (vehicles, teleports, etc.) |

```bash
python3 script/run_simulation.py            # all 30 days
python3 script/run_simulation.py --dry-run  # preview without running
python3 script/run_simulation.py --start 8  # resume from day 8
python3 script/run_simulation.py --only 3   # single day
```

---

## Phase 1 — Parse Detector Output → HDF5

**Script:** `script/parse_detector_output.py`

Converts SUMO's XML detector measurements into a single unified speed matrix
and stores it as HDF5.

```
Rows    : one per 5-minute interval, anchored to real calendar dates
Columns : one per graph node (= one per edge that has detectors)
Values  : mean speed across all lanes of that edge, in mph
```

**Lane aggregation:** since all lanes of an edge share the same midpoint
position, their speed readings are averaged to produce one value per node.
SUMO's `speed=-1` (no vehicle observed) is replaced with the lane speed limit.

**Primary mode — January 2026 directory:**
```bash
python3 script/parse_detector_output.py \
    --jan2026-dir output/jan2026 \
    --output      data/traffic/speed_jan2026.h5
```

**Output:** `data/traffic/speed_jan2026.h5`
- Shape: `(8,640 × N)` — 288 intervals/day × 30 days, N nodes
- Index: `DatetimeIndex` with real timestamps (2026-01-01 … 2026-01-30)

---

## Phase 2 — Build Node Location Table

**Script:** `script/build_sensor_locations.py`

Maps each graph node (edge) to GPS coordinates (lat, lon) using sumolib's
network projection. Uses the first lane's midpoint as the representative
position — all lanes share the same midpoint.

**Canonical node set:** intersection of weekday and weekend filtered detectors
(nodes with meaningful traffic variation on both day types). Use `--union` to
include all nodes.

```bash
python3 script/build_sensor_locations.py           # intersection (default)
python3 script/build_sensor_locations.py --union   # all nodes
```

**Outputs:**

| File | Columns | Description |
|---|---|---|
| `data/graph/nodes.csv` | index, node_id, latitude, longitude, lane_count, detector_ids | One row per edge/node |
| `data/graph/node_ids.txt` | comma-separated node IDs | Canonical node list for downstream scripts |

`node_id` = edge ID (e.g. `123456789`). `lane_count` = how many lane detectors
collapsed into this node. `detector_ids` = pipe-separated detector IDs.

---

## Phase 3 — Build Road-Network Distance Matrix

**Script:** `script/build_distances.py`

Runs forward Dijkstra from every graph node through the SUMO network to find
all reachable nodes within 2,000 m. Distances are **directed** (A→B ≠ B→A on
one-way roads), reflecting actual congestion propagation paths.

```bash
python3 script/build_distances.py                        # default 2,000 m
python3 script/build_distances.py --max-distance 3000    # wider search
```

**Output:** `data/graph/edges.csv`

| Column | Description |
|---|---|
| `from_node` | Source edge ID |
| `to_node` | Target edge ID |
| `distance_m` | Road-network distance in metres |

Use this file to construct any model's adjacency / weight matrix, e.g.:
- Gaussian kernel: `W_ij = exp(−d² / σ²)`, threshold at 0.1
- Binary: `W_ij = 1` if `d < threshold`
- Normalised Laplacian for spectral GCN

---

## Phase 4 — Model Training

With the three dataset files ready, build the model-specific inputs:

```
data/graph/nodes.csv          → node feature / position encoding
data/graph/edges.csv          → adjacency matrix
data/traffic/speed_jan2026.h5 → sliding-window (x, y) arrays
```

**Suggested train/val/test split:** 70% / 10% / 20% by day
(21 weekdays + 4 weekends training; 3 days val; 6 days test).

The pipeline is compatible with any spatial-temporal model:
DCRNN, GraphWaveNet, STGCN, ASTGCN, or custom GNNs.

---

## End-to-end command sequence

```bash
# ── Phase 0: Simulation (sumo_3/) ─────────────────────────────────────────────

python3 script/generate_detectors.py                # midpoint placement ✓
python3 script/build_routes.py                      # 30-day demand ✓

# Trial runs → variance filter
sumo -c config/sumo_chattanooga_weekdays.sumocfg --mesosim true --end 3600
sumo -c config/sumo_chattanooga.sumocfg          --mesosim true --end 3600
python3 script/filter_detectors.py --day-type weekday --threshold 1.0
python3 script/filter_detectors.py --day-type weekend --threshold 1.0

# Full 30-day runs → output/jan2026/YYYY-MM-DD/
python3 script/run_simulation.py

# ── Phases 1–3: Preprocessing (sumo_3/) ──────────────────────────────────────

python3 script/parse_detector_output.py \
    --jan2026-dir output/jan2026 \
    --output      data/traffic/speed_jan2026.h5

python3 script/build_sensor_locations.py
python3 script/build_distances.py

# ── Phase 4: Model ─────────────────────────────────────────────────────────────

# Inputs ready:
#   data/graph/nodes.csv
#   data/graph/edges.csv
#   data/traffic/speed_jan2026.h5
```
