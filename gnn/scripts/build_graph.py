#!/usr/bin/env python3
"""
build_graph.py — Build detector graph using road network distances (DeepSUMO style).

Reads edges.csv (precomputed shortest-path distances between detector pairs),
applies a distance threshold to create a binary adjacency matrix.

edges.csv was precomputed using Dijkstra on the SUMO road network. It stores
all detector-to-detector shortest-path distances ≤ 2000m.

Run from gnn/:
    python scripts/build_graph.py [--threshold 500] [--config config.yaml]

Outputs (saved to data/):
    graph.pkl — dict:
        adj_matrix  : (N, N) float32 binary adjacency
        node_ids    : list[str] length N
        num_nodes   : int
        num_edges   : int
        threshold_m : float
        lats, lons  : np.ndarray for reference
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import yaml


def build_graph(nodes_csv: str, edges_csv: str, threshold_m: float, output_dir: str):
    nodes = pd.read_csv(nodes_csv)
    edges = pd.read_csv(edges_csv)

    node_ids = nodes['node_id'].tolist()
    lats     = nodes['latitude'].values.astype(np.float64)
    lons     = nodes['longitude'].values.astype(np.float64)
    N        = len(node_ids)

    # Index map: node_id → row index
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    print(f'Building network-distance graph for {N} detectors, threshold={threshold_m}m ...')
    print(f'  Loaded {len(edges):,} precomputed shortest-path pairs from edges.csv')

    # ── Filter edges below threshold and build adjacency ──────────────────────
    sub = edges[edges['distance_m'] <= threshold_m].copy()

    adj = np.zeros((N, N), dtype=np.float32)
    skipped = 0
    for _, row in sub.iterrows():
        i = id_to_idx.get(row['from_node'])
        j = id_to_idx.get(row['to_node'])
        if i is None or j is None:
            skipped += 1
            continue
        adj[i, j] = 1.0
        adj[j, i] = 1.0   # undirected: add both directions

    if skipped:
        print(f'  Skipped {skipped} edges (nodes not in nodes.csv)')

    np.fill_diagonal(adj, 0.0)   # no self-loops

    n_edges  = int(adj.sum())
    isolated = int((adj.sum(axis=1) == 0).sum())

    print(f'  Edges (undirected): {n_edges // 2}  ({n_edges} directed)')
    print(f'  Avg degree        : {n_edges / N:.1f}')
    print(f'  Isolated nodes    : {isolated}')

    result = {
        'adj_matrix':  adj,
        'node_ids':    node_ids,
        'num_nodes':   N,
        'num_edges':   n_edges,
        'threshold_m': threshold_m,
        'lats':        lats,
        'lons':        lons,
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'graph.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(result, f)
    print(f'  Saved → {out_path}')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',    default='config.yaml')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Override network distance threshold in meters')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    threshold = args.threshold or cfg['data']['network_threshold_m']
    build_graph(
        nodes_csv   = cfg['data']['nodes_csv'],
        edges_csv   = cfg['data']['edges_csv'],
        threshold_m = threshold,
        output_dir  = cfg['data']['output_dir'],
    )


if __name__ == '__main__':
    main()
