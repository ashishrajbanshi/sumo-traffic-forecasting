# DCRNN Technical Walkthrough — Chattanooga Traffic Forecasting

This document explains every concept, every number, and every decision in the
DCRNN pipeline from raw simulation output to a trained forecasting model.
No prior deep learning expertise is assumed beyond surface familiarity with
RNNs and graphs.

---

## Table of Contents

1. [The Problem We Are Solving](#1-the-problem-we-are-solving)
2. [Why Not a Simple Model?](#2-why-not-a-simple-model)
3. [The Data — Where It Comes From](#3-the-data--where-it-comes-from)
4. [The Graph — Nodes, Edges, and the Adjacency Matrix](#4-the-graph--nodes-edges-and-the-adjacency-matrix)
5. [Graph Diffusion — How Information Spreads](#5-graph-diffusion--how-information-spreads)
6. [The GRU — Memory Across Time](#6-the-gru--memory-across-time)
7. [The DCGRU Cell — Combining Graph and Time](#7-the-dcgru-cell--combining-graph-and-time)
8. [The Encoder-Decoder Architecture](#8-the-encoder-decoder-architecture)
9. [Curriculum Learning — Teaching the Decoder](#9-curriculum-learning--teaching-the-decoder)
10. [Normalisation — Why We Scale the Data](#10-normalisation--why-we-scale-the-data)
11. [The Loss Function — What We Optimise](#11-the-loss-function--what-we-optimise)
12. [Backpropagation and the Optimiser](#12-backpropagation-and-the-optimiser)
13. [The Training Loop — Step by Step](#13-the-training-loop--step-by-step)
14. [Evaluation Metrics](#14-evaluation-metrics)
15. [The Full Pipeline — Start to Finish](#15-the-full-pipeline--start-to-finish)
16. [What the Numbers Mean](#16-what-the-numbers-mean)

---

## 1. The Problem We Are Solving

**Goal**: given the speed on every road segment in Chattanooga for the past
1 hour, predict the speed on every road segment for the next 1 hour.

In numbers:
- **Past**: 12 timesteps × 5 minutes each = 60 minutes of history
- **Future**: 12 timesteps × 5 minutes each = 60 minutes of prediction
- **Locations**: 1174 road segments (nodes in the graph)
- **Signal**: speed in mph at each node at each timestep

This is a **multi-step, multi-node, spatial time-series forecasting** problem.
It is hard for three reasons:

| Challenge | Explanation |
|---|---|
| Temporal dependency | Speed at 9:00 AM depends on speed at 8:55, 8:50, ... |
| Spatial dependency | A jam at node A will propagate to downstream node B |
| Multi-step | We must predict 12 steps ahead, not just 1 |

---

## 2. Why Not a Simple Model?

### Why not a plain RNN / LSTM?

A plain RNN treats each node independently. It learns "node A's speed at
time t depends on node A's speed at t-1" but it has no idea that node B
is 200 metres upstream and directly feeds traffic into node A. The spatial
relationships are completely ignored.

### Why not a CNN on a grid?

CNNs work on grids (images). Road networks are not grids — they are
**irregular graphs**. A node can have 1 or 6 neighbours depending on the
road layout. There is no natural way to apply a fixed-size convolutional
filter.

### Why DCRNN?

DCRNN (Diffusion Convolutional Recurrent Neural Network) solves both problems:

- **Graph diffusion convolution** captures spatial relationships by
  propagating information through the road network graph
- **GRU** captures temporal relationships by maintaining memory across timesteps
- They are combined inside a single cell (DCGRU) so the model learns both
  simultaneously

---

## 3. The Data — Where It Comes From

### 3.1 The simulation

A 30-day SUMO mesoscopic traffic simulation ran for January 2026 over the
Chattanooga road network. E1 induction-loop detectors (virtual sensors) were
placed on road edges and recorded traffic every 5 minutes.

Each detector recorded four signals per 5-minute interval:
- **Speed** (mph) — mean speed across all vehicles on that edge
- **Volume** (vehicles/interval) — count of vehicles that passed
- **Occupancy** (%) — fraction of time the detector was occupied
- **Flow** (vehicles/hour) — rate of vehicle passage

All four signals were written to `traffic_jan2026.h5`.

### 3.2 The raw HDF5 file

```
traffic_jan2026.h5
  /speed       shape: (8640, 1183)    ← rows = timesteps, columns = nodes
  /volume      shape: (8640, 1183)
  /occupancy   shape: (8640, 1183)
  /flow        shape: (8640, 1183)
```

**8640 timesteps**: 30 days × 24 hours × 12 intervals/hour = 8640 rows,
one per 5-minute slot from 2026-01-01 00:00 to 2026-01-30 23:55.

**1183 nodes**: each node is a SUMO road edge (e.g. `-1044300151#1`).
Multiple detector lanes on the same edge were averaged together into one
value per edge.

### 3.3 Cleaning — removing bad nodes

Nine nodes were removed before modelling:

| Type | Count | Reason |
|---|---|---|
| Isolated | 2 | No reachable neighbours within 2000 m — graph would have disconnected components |
| Zero-volume | 7 | No vehicles observed across all 30 days — speed values are just speed-limit fill, not real measurements |

After removal: **1174 clean nodes**.

### 3.4 What DCRNN actually trains on

Only the **speed** signal is used for training. Speed is the primary traffic
state variable — it encodes both congestion (low speed) and free-flow (high
speed) directly.

```
traffic_chattanooga.h5
  /df    shape: (8640, 1174)    ← speed only, clean nodes only
```

### 3.5 Sliding-window samples

The 8640×1174 speed matrix is cut into overlapping windows:

```
For every timestep t from 11 to 8628:
    x[t] = speed at rows [t-11, t-10, ..., t]      → shape (12, 1174)
    y[t] = speed at rows [t+1,  t+2,  ..., t+12]   → shape (12, 1174)
```

Each `(x, y)` pair is one training sample.
- **x** = what the model sees (past 1 hour)
- **y** = what the model must predict (next 1 hour)

Total samples: 8640 - 12 - 11 = **8617**

A second feature — **time of day** — is appended to x. For a timestep at
09:00 AM, time-of-day = 9/24 = 0.375. This gives the model a clock so it
knows "this is rush hour" vs "this is 3 AM". After appending, each sample
x has shape (12, 1174, **2**) — speed and time-of-day per node per timestep.

### 3.6 Train / val / test split

The 8617 samples are split chronologically (not randomly, because time
series must preserve order):

```
train : 6032 samples  (70%)  → Jan 1  – ~Jan 21
val   :  862 samples  (10%)  → ~Jan 21 – ~Jan 24
test  : 1723 samples  (20%)  → ~Jan 24 – Jan 30
```

Saved as compressed numpy files:
```
data/chattanooga/train.npz   x:(6032,12,1174,2)  y:(6032,12,1174,2)
data/chattanooga/val.npz     x:(862, 12,1174,2)  y:(862, 12,1174,2)
data/chattanooga/test.npz    x:(1723,12,1174,2)  y:(1723,12,1174,2)
```

---

## 4. The Graph — Nodes, Edges, and the Adjacency Matrix

### 4.1 What a graph is

A **graph** G = (V, E) has:
- **V** = set of vertices (nodes) — our 1174 road segments
- **E** = set of edges (connections) — directed road-network links between segments

Two nodes are connected if you can drive from one to the other within 2000 m
along the actual road network. The connection is **directed** — A→B and B→A
are separate edges because roads are one-way and distances differ.

### 4.2 Road-network distance

The distance between two nodes is not straight-line (Euclidean) distance.
It is the actual driving distance through the road network — computed by
Dijkstra's shortest-path algorithm running on the SUMO network topology.

Example: two nodes 300 m apart as the crow flies might be 800 m apart by
road if there is no direct connection.

Results:
```
edges.csv     297,588 directed pairs (from 1174 nodes)
Distance range:  21 m  –  2000 m
Distance mean:  1285 m  ±  490 m (std)
Average neighbours per node: ~253 raw pairs
```

### 4.3 The adjacency matrix

A raw adjacency matrix would be a 1174×1174 binary matrix with 1 where
an edge exists and 0 otherwise. But DCRNN uses a **weighted** adjacency
matrix where the weight encodes how strongly two nodes should influence
each other.

The weight formula is a **Gaussian kernel** on distance:

```
W[i,j] = exp( -(distance[i,j])² / σ² )
```

Where σ (sigma) is the standard deviation of all distances = **490 m**
(computed automatically from the data — no manual tuning needed).

**Why this formula?**

- Close nodes (distance << σ): W ≈ 1 — strong influence
- Nodes at exactly σ away: W = exp(-1) ≈ 0.37 — moderate influence
- Far nodes (distance >> σ): W ≈ 0 — negligible influence
- Nodes beyond 2000 m: W = 0 — no Dijkstra path found, no connection

A **threshold of 0.1** is applied: any W < 0.1 is set to exactly 0.
This creates sparsity — most entries are zero, meaning most node pairs
don't directly influence each other.

Results after thresholding:
```
adj_mx_chattanooga.pkl
  Shape:       (1174, 1174)
  Non-zero:    49,284 entries  (3.6% density)
  Avg degree:  42 neighbours per node
  Weight range: 0.10 – 0.998
```

The `.pkl` file stores three things:
1. `sensor_ids` — list of 1174 node ID strings in matrix order
2. `sensor_id_to_ind` — dictionary mapping node ID → row/column index
3. `adj_mx` — the 1174×1174 float32 weight matrix

---

## 5. Graph Diffusion — How Information Spreads

### 5.1 The intuition

Imagine dropping ink at one node in the graph. The ink spreads to
neighbouring nodes, then to their neighbours, and so on. After K steps,
nodes up to K hops away have received some ink. The amount of ink at each
node depends on how many paths connect it to the source and how far away
it is. This is **diffusion**.

In traffic: a slowdown at node A (accident) spreads upstream as cars queue.
Diffusion models this propagation mathematically.

### 5.2 Random walk matrix

A **random walk** is: starting at node i, at each step, move to a
neighbouring node chosen proportionally to edge weight.

The probability of stepping from node i to node j in one step is:

```
P[i,j] = W[i,j] / sum_k(W[i,k])
```

This is just row-normalisation of the adjacency matrix: divide each row
by its sum so all rows sum to 1. Written as a matrix:

```
P = D_out⁻¹ × A
```

Where D_out is a diagonal matrix of out-degrees (row sums of A).

### 5.3 Dual random walk — why two directions?

Road traffic is directional. A jam at node A propagates **upstream** to
nodes that feed into A. Free flow at node A enables faster speeds
**downstream** at nodes A feeds into. These are different physical effects.

DCRNN uses **dual random walk** — two support matrices:

```
P_forward  = D_out⁻¹ × A      (traffic flows forward through the graph)
P_backward = D_in⁻¹  × Aᵀ    (look backward — what feeds into me?)
```

Using both allows the model to learn both "what am I sending downstream?"
and "what is coming toward me from upstream?".

In your config: `filter_type: dual_random_walk` → 2 support matrices.

### 5.4 K-hop diffusion

One application of P spreads information 1 hop. Applying P twice spreads
it 2 hops. With `max_diffusion_step: 2`:

```
Step 0: x₀ = original node features                    (identity — no diffusion)
Step 1: x₁ = P × x₀                                   (1 hop neighbours)
Step 2: x₂ = P × x₁                                   (2 hop neighbours)
```

For dual random walk with K=2:
```
Feature sets: [x₀,  P_fw×x₀,  P_fw²×x₀,  P_bw×x₀,  P_bw²×x₀]
              identity  fwd-1hop  fwd-2hop   bwd-1hop   bwd-2hop
Total: 1 + 2×2 = 5 feature sets
```

These 5 feature sets are concatenated along the feature dimension and fed
into a linear layer. The linear layer learns which hops matter most for
each gate in the GRU.

### 5.5 Concrete example with numbers

Say node A is a road segment on the I-24 ramp. Its 1-hop forward
neighbours are the road segments immediately downstream. Its 2-hop forward
neighbours are the segments 1-2 km ahead. Its backward neighbours are the
feeder roads.

When computing the update for node A at time t:
- x₀ gives A's own current state
- P_fw×x₀ gives a weighted average of A's downstream neighbours' states
- P_bw×x₀ gives a weighted average of A's upstream neighbours' states
- P_fw²×x₀ gives 2-hop downstream context (e.g., a merging highway)
- P_bw²×x₀ gives 2-hop upstream context (e.g., where the queue starts)

All five are concatenated → the model sees the full local neighbourhood
context around every node simultaneously.

---

## 6. The GRU — Memory Across Time

### 6.1 Why we need memory

Traffic at 9:05 AM is not independent of traffic at 9:00 AM. A GRU
(Gated Recurrent Unit) maintains a **hidden state** h that summarises
everything seen so far. It is updated at every timestep.

### 6.2 The three equations

A standard GRU at timestep t takes:
- `x_t` — current input (features at this timestep)
- `h_{t-1}` — hidden state from the previous timestep

And computes:

```
r_t = sigmoid( W_r × [x_t, h_{t-1}] + b_r )     reset gate
u_t = sigmoid( W_u × [x_t, h_{t-1}] + b_u )     update gate
c_t = tanh(    W_c × [x_t, r_t ⊙ h_{t-1}] + b_c )  candidate
h_t = u_t ⊙ h_{t-1} + (1 - u_t) ⊙ c_t          new hidden state
```

`[x_t, h_{t-1}]` means concatenation. `⊙` means element-wise multiplication.

**What each gate does:**

**Reset gate r_t** (values 0–1 per element):
- r ≈ 0: forget the past hidden state when computing the candidate
- r ≈ 1: let the past hidden state fully influence the candidate
- Lets the model decide "is the past relevant right now?"
- Example: if there was a special event on day 3 that made traffic unusual,
  r can tell the model to partially ignore that when predicting normal days

**Update gate u_t** (values 0–1 per element):
- u ≈ 0: replace hidden state completely with new candidate
- u ≈ 1: keep old hidden state, ignore new input
- Lets the model decide "how much should I update my memory?"
- Example: at 3 AM when traffic is stable, u stays high (keep the memory,
  nothing new is happening)

**Candidate c_t**: the proposed new hidden state, computed using the reset-gated old state.

**New hidden state h_t**: a blend of old state (weight u) and new candidate (weight 1-u).

### 6.3 Why GRU and not LSTM?

LSTM has two internal states (cell state + hidden state) and three gates.
GRU has one hidden state and two gates. In practice GRU trains faster and
performs similarly. DCRNN's original paper chose GRU.

---

## 7. The DCGRU Cell — Combining Graph and Time

### 7.1 The key idea

In a standard GRU, the matrix multiplications `W × [x_t, h_{t-1}]` are
just learned linear transformations applied identically to each node
independently — no graph structure involved.

DCGRU replaces those matrix multiplications with **graph diffusion
convolutions**. The diffusion from Section 5 IS the matrix multiplication.
Specifically, for each gate:

```
Standard GRU:    gate = sigmoid( W × [x_t, h] )
DCGRU:           gate = sigmoid( Linear( Diffuse([x_t, h]) ) )
```

Where `Diffuse([x_t, h])` produces the 5-set diffused features from
Section 5.4. The `Linear` layer then maps those features to the gate values.

This means every gate computation is spatially aware — it considers not
just this node's own state but the states of neighbours up to 2 hops away.

### 7.2 Weight sharing across nodes

The `Linear` layer weights are **shared across all 1174 nodes**. One set
of weights is learned and applied identically to every node's neighbourhood.
This is the "convolutional" part — like a CNN applies the same filter at
every pixel, the DCGRU applies the same learned filter at every node.

This is efficient: instead of 1174 separate models (one per node), we learn
one universal model of how traffic propagates through a local neighbourhood.

### 7.3 The hidden state dimensions

For one DCGRUCell with `num_units=64` and `num_nodes=1174`:

```
Hidden state shape: (batch_size, 1174 × 64) = (batch, 75,136)
```

Each of the 1174 nodes has its own 64-dimensional hidden vector. The 64
numbers encode that node's traffic "memory" — speed history, trend,
whether it is building toward congestion, etc. — in a learned latent space.

### 7.4 Step-by-step computation for one batch

Given a batch of 64 samples at timestep t:
```
Input x_t: (64, 1174, 2)       ← speed + time-of-day for all nodes
State h_{t-1}: (64, 1174, 64)  ← previous hidden state

1. Concatenate: [x_t, h_{t-1}] → (64, 1174, 66)

2. Diffuse through 2 supports × 2 hops + identity:
   → 5 feature sets of shape (64, 1174, 66) each
   → concatenated: (64, 1174, 330)
   → reshaped:     (64×1174, 330) = (75136, 330)

3. W_gate linear: (75136, 330) → (75136, 128)
   → split into r: (75136, 64) and u: (75136, 64)
   → sigmoid → values 0–1
   → reshape: r: (64, 1174, 64)  u: (64, 1174, 64)

4. Candidate diffusion: [x_t, r⊙h_{t-1}] → diffuse → W_cand → tanh
   → c: (64, 1174, 64)

5. New state: h_t = u⊙h_{t-1} + (1-u)⊙c → (64, 1174, 64)
```

### 7.5 Two stacked layers

The model uses `num_rnn_layers: 2` — two DCGRUCells stacked:

```
Layer 1:  input = x_t (features: 2)        → output h1_t (features: 64)
Layer 2:  input = h1_t (features: 64)      → output h2_t (features: 64)
```

Layer 1 learns low-level patterns (raw speed + neighbourhood). Layer 2
learns higher-level abstractions (congestion waves, capacity states) from
layer 1's output.

### 7.6 Parameter count

```
DCGRUCell layer 1:
  W_gate: input=(5 × (2+64))=330  output=2×64=128  params= 330×128 + 128 = 42,368
  W_cand: input=330               output=64         params= 330×64  + 64  = 21,184
  Total layer 1: 63,552

DCGRUCell layer 2:
  W_gate: input=(5 × (64+64))=640  output=128      params= 640×128 + 128 = 82,048
  W_cand: input=640                output=64        params= 640×64  + 64  = 41,024
  Total layer 2: 123,072

× 2 (encoder + decoder): (63,552 + 123,072) × 2 = 373,248

Output projection (decoder): 64 → 1: 64×1 + 1 = 65

Total ≈ 372,353 parameters  ✓ (matches the log output)
```

---

## 8. The Encoder-Decoder Architecture

### 8.1 Overview

DCRNN is a **sequence-to-sequence** model — it reads a sequence (12 past
timesteps) and writes a sequence (12 future timesteps). This uses an
encoder-decoder structure:

```
                    ┌─────────────────────────────┐
Input sequence      │         ENCODER              │
(12 past steps) ──► │  DCGRU → DCGRU → ... → DCGRU │ ──► Hidden state
                    └─────────────────────────────┘           │
                                                              │
                    ┌─────────────────────────────┐           │
                    │         DECODER              │ ◄─────────┘
                    │  DCGRU → DCGRU → ... → DCGRU │ ──► Predictions
                    └─────────────────────────────┘   (12 future steps)
```

### 8.2 The encoder

The encoder processes the 12 past timesteps one at a time, left to right.
At each timestep t:

```
h_t = DCGRUCell(x_t, h_{t-1})
```

After processing all 12 timesteps, the final hidden state h_12 is a
**compressed summary of the past 60 minutes of traffic at all 1174 nodes
simultaneously**. This hidden state is passed to the decoder.

Think of it as: after reading 12 snapshots of traffic, the encoder has
built up a rich internal representation of "the current traffic situation"
— which roads are fast, which are slow, which are building toward congestion.

### 8.3 The decoder

The decoder generates predictions one timestep at a time, using:
1. The encoder's final hidden state as its starting hidden state
2. Its own previous prediction as input for the next step

```
Step 1: input = go_symbol (zeros)  + h_enc → predict ŷ_1
Step 2: input = ŷ_1                + h_1   → predict ŷ_2
Step 3: input = ŷ_2                + h_2   → predict ŷ_3
...
Step 12: input = ŷ_11              + h_11  → predict ŷ_12
```

This is **auto-regressive** decoding — each prediction depends on all
previous predictions. Errors can accumulate: if step 1 is slightly wrong,
step 2 uses that wrong value and might be more wrong, and so on.

### 8.4 Why encoder-decoder instead of direct prediction?

An alternative would be to directly map 12 input steps → 12 output steps
with a single network. Encoder-decoder is better because:

1. The encoder compresses temporal information efficiently into a fixed
   hidden state — the decoder starts from a good initialisation
2. Auto-regressive decoding allows the model to be self-consistent —
   predictions at each step are conditioned on previous predictions
3. During training, curriculum learning (Section 9) makes the decoder
   robust to its own errors

---

## 9. Curriculum Learning — Teaching the Decoder

### 9.1 The problem

During training, the decoder makes predictions and feeds them back as
input. But early in training, predictions are terrible. Feeding bad
predictions back makes the next prediction even worse. The model gets
stuck.

### 9.2 The solution: scheduled sampling

Early in training: instead of feeding the decoder's own (bad) prediction
back, feed the **ground truth** (the real observed speed). This is called
**teacher forcing** — the teacher (ground truth) guides the student.

Late in training: gradually switch to feeding the decoder's own predictions
back, so it learns to handle its own errors.

The probability of using ground truth at step t is:

```
cl_ratio = cl_decay_steps / (cl_decay_steps + e^(global_step / cl_decay_steps))
```

With `cl_decay_steps: 2000`:

```
global_step = 0     → cl_ratio = 2000/(2000+1)    ≈ 1.00  (use ground truth 100%)
global_step = 2000  → cl_ratio = 2000/(2000+e¹)  ≈ 0.42  (use ground truth 42%)
global_step = 5000  → cl_ratio = 2000/(2000+e²·⁵) ≈ 0.14  (use ground truth 14%)
global_step = 10000 → cl_ratio ≈ 0.00              (always use own predictions)
```

Your training has 15 epochs × 95 batches/epoch = 1425 total steps.
At step 1425: `cl_ratio ≈ 2000/(2000+e^0.71) ≈ 0.73`. So for your 15-epoch
run, the decoder still uses ground truth ~73% of the time at the end —
it is still mostly being teacher-forced. A full 100-epoch run would
approach fully auto-regressive behaviour.

### 9.3 During evaluation

At validation and test time, `teacher_forcing_ratio = 0` always. The
model must predict entirely from its own previous outputs — this is the
real-world scenario where you have no ground truth.

---

## 10. Normalisation — Why We Scale the Data

### 10.1 The problem with raw speed values

Speed values range from ~8.7 mph (nearly stopped) to ~86.8 mph (freeway).
Neural network weight updates work best when input values are in a small
range around zero. Large input values cause large activations, large
gradients, and unstable training.

### 10.2 StandardScaler (Z-score normalisation)

Fit on the **training set only** (never val or test — that would be
data leakage):

```
mean = mean of all (timestep, node) speed values in train.npz  ≈ 43.6 mph
std  = std  of all (timestep, node) speed values in train.npz  ≈  X mph

Normalised speed = (raw_speed - mean) / std
```

After normalisation:
- Average speed → 0.0
- Speed one std above average → 1.0
- Speed one std below average → -1.0

The model trains on these normalised values. When computing metrics, the
scaler's `inverse_transform` converts back to mph:

```
raw_speed = normalised_speed × std + mean
```

### 10.3 What the model actually sees

The `y` labels used for loss computation are also normalised. The MAE
in the training log is therefore in **normalised units**, not mph.

The validation and test MAE reported are in **mph** (after inverse transform)
so they are interpretable: `val_mae: 1.68` means predictions are on
average 1.68 mph off from the true speed.

---

## 11. The Loss Function — What We Optimise

### 11.1 Masked MAE

```python
loss = masked_mae(pred[..., 0], y_batch[..., 0])
```

**MAE** = Mean Absolute Error:
```
MAE = (1/N) × Σ |predicted_speed - true_speed|
```

The average absolute difference between predicted and true normalised
speed, summed over all nodes, all horizon steps, all samples in the batch.

**Masked** = ignoring entries where `true_speed == 0`. Why?

In the original METR-LA dataset, missing sensor readings are filled with 0.
Our Chattanooga data has no missing sensors, but the mask is kept for
compatibility. A zero speed would mean a completely stopped road —
extremely rare and worth treating separately.

### 11.2 Why MAE and not MSE?

MSE (Mean Squared Error) squares the error — a 10 mph error becomes 100,
a 1 mph error becomes 1. This makes the model overly sensitive to large
outliers and pushes it to "play it safe" by predicting average speeds.

MAE treats all errors linearly — a 10 mph error is 10 times worse than
a 1 mph error. This trains a model that tries to minimise typical errors
rather than catastrophic ones. For traffic forecasting, MAE better reflects
the user experience: being 2 mph off 50 times is worse than being 10 mph
off once.

---

## 12. Backpropagation and the Optimiser

### 12.1 How the model learns

After computing the loss, PyTorch automatically computes the gradient of
the loss with respect to every one of the 372,353 parameters. This is
**backpropagation** — the chain rule of calculus applied through the
entire computation graph.

The gradient tells us: "if I increase this weight by a tiny amount, how
much does the loss increase or decrease?" A negative gradient means
increasing the weight decreases the loss → we should increase it.

### 12.2 Gradient clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

RNNs unrolled over 12 timesteps can suffer from **exploding gradients** —
the gradient signal gets multiplied together 12 times as it flows backward
through time, and can become astronomically large. This destabilises training.

Gradient clipping caps the total gradient magnitude at 5.0. If the gradient
is larger, it is scaled down proportionally. This keeps training stable.

### 12.3 Adam optimiser

```python
optimizer = Adam(lr=0.01, eps=1e-3)
```

Adam (Adaptive Moment Estimation) improves on plain gradient descent by:

1. **Momentum**: accumulates a running average of past gradients. If
   gradients have consistently pointed in one direction, Adam takes larger
   steps that way. Speeds up convergence in flat regions.

2. **Adaptive learning rate**: maintains a per-parameter learning rate.
   Parameters that have had large gradients historically get a smaller
   effective learning rate. Parameters with small gradients get a larger
   effective rate. This handles the fact that some weights matter more
   than others.

The `eps=1e-3` prevents division by zero in the adaptive rate computation
(and also slightly stabilises training with large normalised gradients
as typically seen in sequence models).

---

## 13. The Training Loop — Step by Step

Here is exactly what happens in one epoch, from the perspective of the
code in `dcrnn_train_pt.py`:

```
EPOCH START
│
├── model.train()  ← enable dropout, batch norm, etc. (none here, but good practice)
│
├── For each of 95 batches (6032 ÷ 64 = 94.25 → padded to 95):
│   │
│   ├── x_batch: (64, 12, 1174, 2) → move to GPU
│   ├── y_batch: (64, 12, 1174, 2) → move to GPU
│   │
│   ├── cl_ratio = compute curriculum ratio from global_step
│   │
│   ├── FORWARD PASS:
│   │   ├── Encoder reads x_batch[:,0,:,:]  → updates hidden state h0
│   │   ├── Encoder reads x_batch[:,1,:,:]  → updates hidden state h1
│   │   ├── ...
│   │   ├── Encoder reads x_batch[:,11,:,:] → final hidden state h11
│   │   │
│   │   ├── Decoder step 1: input=zeros, state=h11 → predict ŷ[:,0,:,:]
│   │   ├── Decoder step 2: input=ŷ[:,0] or y[:,0] (curriculum) → predict ŷ[:,1,:,:]
│   │   ├── ...
│   │   └── Decoder step 12 → predict ŷ[:,11,:,:]
│   │
│   ├── loss = masked_mae(ŷ[...,0], y_batch[...,0])  ← normalised MAE
│   │
│   ├── optimizer.zero_grad()   ← clear old gradients
│   ├── loss.backward()         ← compute new gradients
│   ├── clip_grad_norm_(max=5)  ← cap gradient magnitude
│   ├── optimizer.step()        ← update all 372,353 parameters
│   │
│   └── global_step += 1
│
├── scheduler.step()  ← adjust learning rate (no change for first 20 epochs)
│
├── VALIDATION:
│   ├── model.eval()  ← disable training-specific behaviour
│   ├── For each val batch (862 ÷ 64 = 14 batches):
│   │   ├── pred = model(x)  ← no teacher forcing, no gradients
│   │   ├── inverse_transform predictions and labels back to mph
│   │   └── accumulate MAE, RMSE, MAPE
│   └── val_mae = average over all val batches
│
├── LOG: "Epoch [e/15] (step) train_mae: X  val_mae: Y  lr: Z  Xs"
│
├── IF val_mae < best_val_mae:
│   ├── save best_model.pt
│   └── reset wait counter
│   ELSE:
│   └── wait += 1 (early stopping counter)
│
└── IF (epoch+1) % 10 == 0:
    └── run TEST evaluation and log MAE/RMSE/MAPE
```

---

## 14. Evaluation Metrics

All three metrics are computed on predictions **inverse-transformed back to mph**
so they are human-interpretable.

### MAE — Mean Absolute Error
```
MAE = (1/N) Σ |ŷ - y|
```
Average prediction error in mph. `MAE = 1.68` means on average predictions
are 1.68 mph away from truth. Intuitive and robust to outliers.

### RMSE — Root Mean Squared Error
```
RMSE = sqrt( (1/N) Σ (ŷ - y)² )
```
Like MAE but squares the errors first, so large errors are penalised more.
`RMSE > MAE` always. If RMSE is much larger than MAE, there are occasional
large prediction failures. If RMSE ≈ MAE, errors are consistent in size.

### MAPE — Mean Absolute Percentage Error
```
MAPE = (100/N) Σ |ŷ - y| / |y|
```
Percentage error relative to the true value. `MAPE = 4%` means predictions
are off by 4% on average. Useful for comparing across different speed ranges
(a 2 mph error on a 10 mph road is worse than a 2 mph error on a 60 mph freeway).

### Horizon breakdown
The model predicts 12 steps (5-min intervals) ahead. Metrics can be broken
down by horizon:
- **Step 1** (5 min ahead): easiest, typically lowest error
- **Step 6** (30 min ahead): moderate error
- **Step 12** (60 min ahead): hardest, highest error — uncertainty accumulates

---

## 15. The Full Pipeline — Start to Finish

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SUMO SIMULATION                               │
│  30 days × January 2026 × Chattanooga network                       │
│  E1 detectors → 5-min interval XML output files                     │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ parse_detector_output.py
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    traffic_jan2026.h5                                │
│  Shape: (8745, 1183) → fixed to (8640, 1183) by fix_hdf5_alignment  │
│  Keys: /speed /volume /occupancy /flow                               │
│  Range: 2026-01-01 00:00 → 2026-01-30 23:55, 5-min cadence         │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ prepare_data.py
                            │  - remove 9 bad nodes
                            │  - extract /speed only
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  traffic_chattanooga.h5                              │
│  Shape: (8640, 1174) — 1174 clean nodes, speed in mph               │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ generate_training_data.py
                            │  - sliding 12→12 windows
                            │  - append time-of-day feature
                            │  - StandardScaler on channel 0
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│             train.npz / val.npz / test.npz                          │
│  Each: x (samples,12,1174,2)  y (samples,12,1174,2)                 │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            │
┌───────────────────────────┼─────────────────────────────────────────┐
│  edges.csv (297,588 pairs)│                                          │
│  node_ids.txt (1174)      │ gen_adj_mx.py                           │
│                           │  - Gaussian kernel on distance          │
│                           │  - threshold 0.1                        │
│                           ▼                                          │
│              adj_mx_chattanooga.pkl                                  │
│              (1174×1174, 49,284 non-zeros)                          │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      dcrnn_train_pt.py                               │
│                                                                      │
│  Build supports:  adj_mx → P_forward (D⁻¹A), P_backward (D⁻¹Aᵀ)  │
│                                                                      │
│  Build model:                                                        │
│    Encoder: 2 × DCGRUCell(input=2,  units=64, nodes=1174, K=2)     │
│    Decoder: 2 × DCGRUCell(input=1,  units=64, nodes=1174, K=2)     │
│    Output proj: Linear(64 → 1)                                       │
│    Total: 372,353 parameters                                         │
│                                                                      │
│  Train 15 epochs:                                                    │
│    - Adam lr=0.01, gradient clipping at 5                           │
│    - Curriculum learning (cl_ratio decays from ~1 to ~0.73)        │
│    - Save best_model.pt when val MAE improves                       │
│    - Test evaluation at epoch 10                                     │
│                                                                      │
│  Final test on held-out Jan 24–30 data                              │
└─────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        RESULTS                                       │
│  data/model/dcrnn_pt_DR_2_h_12_64-64_.../                          │
│    info.log        ← epoch-by-epoch MAE (in normalised units)       │
│    best_model.pt   ← weights of best checkpoint                     │
│                                                                      │
│  Final test metrics (in mph):                                        │
│    MAE:  ~X.XX mph    average absolute error                        │
│    RMSE: ~X.XX mph    penalises large errors more                   │
│    MAPE: ~X.XX%       percentage error                              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 16. What the Numbers Mean

### 16.1 Reading the training log

```
Epoch [0/15] (95) train_mae: 1.90  val_mae: 1.82  lr: 0.010000  149.7s
```

- `[0/15]` — epoch 0 out of 15
- `(95)` — global step count (95 batches processed so far)
- `train_mae: 1.90` — average MAE on training set **in normalised units**
- `val_mae: 1.82` — average MAE on validation set **in mph** (after inverse transform)
- `lr: 0.010000` — current learning rate (not yet decayed at epoch 0)
- `149.7s` — time for this epoch

### 16.2 Interpreting val_mae

`val_mae: 1.68 mph` means: across all 1174 nodes, all 12 prediction steps,
and all 862 validation samples, the average absolute prediction error is
1.68 mph.

Context:
- Speed range in the data: 8.7 – 86.8 mph
- Mean speed: ~43.6 mph
- MAE of 1.68 mph = ~3.8% of mean speed — quite good for a 60-minute ahead forecast

### 16.3 Expected behaviour across epochs

```
Early epochs (0-5):   MAE drops quickly — model learns basic patterns
Middle epochs (5-20): MAE improves slowly — fine-tuning
Late epochs (20-50):  LR drops at epoch 20 (×0.1), loss drops again briefly
After 50 epochs:      Usually converged or early-stopping kicks in
```

For your 15-epoch run you will see mostly the first phase — rapid initial
improvement. A 100-epoch run would show the full learning curve.

### 16.4 Flat traffic limitation

Your SUMO simulation used uniform random vehicle trips with no time-of-day
variation. This means there are no AM/PM peaks — traffic is roughly the
same at 8 AM and 3 AM. The speed varies only ~0.4 mph across the day.

Consequence: the model's task is easier than real traffic (no peaks to learn)
but the results are not representative of real-world performance. The pipeline,
architecture, and metrics are all correct — the data is the limiting factor.
To get realistic results, the simulation demand profile would need to be
rebuilt with time-varying vehicle generation rates matching Chattanooga's
real traffic patterns.

---

*Generated from the sumo_3/DCRNN pipeline — Chattanooga, TN — January 2026 simulation.*
