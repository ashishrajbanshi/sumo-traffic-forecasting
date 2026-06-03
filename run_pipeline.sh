#!/bin/bash
# Full pipeline: routes → simulation → parse → train
set -e   # stop on first error

cd "$(dirname "$0")"

echo "========================================"
echo "Step 1/6  Build routes"
echo "========================================"
python script/build_routes.py

echo "========================================"
echo "Step 2/6  Run SUMO simulation"
echo "========================================"
python script/run_simulation.py --workers 16

echo "========================================"
echo "Step 3/6  Parse detector output"
echo "========================================"
python script/parse_detector_output.py --jan2026-dir output/jan2026

echo "========================================"
echo "Step 4/6  Generate training data"
echo "========================================"
python DCRNN/scripts/generate_training_data.py \
    --traffic_df_filename data/traffic/traffic_jan2026.h5 \
    --output_dir          DCRNN/data/chattanooga

echo "========================================"
echo "Step 5/6  Build adjacency matrix"
echo "========================================"
python DCRNN/scripts/gen_adj_mx.py \
    --sensor_ids_filename data/graph/node_ids.txt \
    --distances_filename  data/graph/edges.csv \
    --output_pkl_filename DCRNN/data/sensor_graph/adj_mx_chattanooga.pkl

echo "========================================"
echo "Step 6/6  Train DCRNN"
echo "========================================"
cd DCRNN
python dcrnn_train_pt.py \
    --config_filename data/model/dcrnn_chattanooga.yaml \
    --num-workers 8

echo "========================================"
echo "Pipeline complete."
echo "========================================"
