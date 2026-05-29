"""
stgat.py — Spatio-Temporal Graph Attention Network (DeepSUMO style).

Architecture (Zhang et al. 2019, adapted):
  Spatial  : stacked GATLayer (multi-head graph attention)
  Temporal : LSTM over the sequence of GAT-encoded node embeddings
  Output   : Linear projection → speed predictions

No torch_geometric required — implemented with dense adjacency matrix.

Input/output convention:
  x    : (B, T, N, F)   — batch, timesteps, nodes, features
  adj  : (N, N)          — binary adjacency (on device)
  → out: (B, T_pred, N, 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """
    Single graph attention layer with K heads (Velickovic et al., 2018).

    Uses dense (N×N) adjacency masking rather than edge-index iteration,
    which runs faster on Blackwell GPU than sparse kernels.
    """

    def __init__(self, in_dim: int, out_dim: int, n_heads: int,
                 dropout: float = 0.0, concat: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.concat  = concat

        self.W     = nn.Linear(in_dim, n_heads * out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(1, n_heads, out_dim))
        self.a_dst = nn.Parameter(torch.empty(1, n_heads, out_dim))
        self.leaky = nn.LeakyReLU(0.2, inplace=True)
        self.drop  = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x   : (B, N, in_dim)   B may be batch*time after vectorisation
            adj : (N, N) binary adjacency WITH self-loops already added

        Returns:
            (B, N, n_heads * out_dim)  if concat
            (B, N, out_dim)            if not concat
        """
        B, N, _ = x.shape
        H, D    = self.n_heads, self.out_dim

        Wx    = self.W(x).view(B, N, H, D)                      # (B, N, H, D)
        e_src = (Wx * self.a_src).sum(dim=-1)                   # (B, N, H)
        e_dst = (Wx * self.a_dst).sum(dim=-1)                   # (B, N, H)
        e     = self.leaky(e_src.unsqueeze(2) + e_dst.unsqueeze(1))  # (B, N, N, H)

        # adj already has self-loops; mask only true non-edges
        mask  = (adj == 0).unsqueeze(0).unsqueeze(-1)            # (1, N, N, 1)
        e     = e.masked_fill(mask, float('-inf'))

        alpha = F.softmax(e, dim=2)                              # (B, N, N, H)
        alpha = self.drop(alpha)

        # Weighted aggregation: (B, H, N, N) @ (B, H, N, D) → (B, H, N, D)
        Wx_t    = Wx.permute(0, 2, 1, 3)                        # (B, H, N, D)
        alpha_t = alpha.permute(0, 3, 1, 2)                     # (B, H, N, N)
        out     = torch.matmul(alpha_t, Wx_t).permute(0, 2, 1, 3)  # (B, N, H, D)

        if self.concat:
            out = out.reshape(B, N, H * D)
        else:
            out = out.mean(dim=2)                                # (B, N, D)

        return F.elu(out)


class STGAT(nn.Module):
    """
    Spatio-Temporal Graph Attention Network.

    Pipeline per forward pass:
      1. All T timesteps encoded simultaneously through stacked GATLayers
         (reshape trick: (B,T,N,F) → (B*T,N,F) → one GAT pass → reshape back)
      2. Stack timestep embeddings → feed to LSTM per node
      3. Project last LSTM hidden state → n_pred speed values
    """

    def __init__(
        self,
        n_nodes:     int,
        n_features:  int,
        n_hist:      int,
        n_pred:      int,
        gat_heads:   int   = 2,
        gat_hidden:  int   = 32,
        gat_layers:  int   = 2,
        lstm_hidden: int   = 128,
        dropout:     float = 0.3,
    ):
        super().__init__()
        self.n_nodes    = n_nodes
        self.n_hist     = n_hist
        self.n_pred     = n_pred

        gat_list = []
        in_d = n_features
        for i in range(gat_layers):
            is_last = (i == gat_layers - 1)
            concat  = not is_last
            gat_list.append(
                GATLayer(in_d, gat_hidden, n_heads=gat_heads,
                         dropout=dropout, concat=concat)
            )
            in_d = gat_hidden * gat_heads if concat else gat_hidden

        self.gat         = nn.ModuleList(gat_list)
        self.gat_out_dim = in_d
        self.spatial_drop = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size  = self.gat_out_dim,
            hidden_size = lstm_hidden,
            num_layers  = 1,
            batch_first = True,
        )
        self.output_proj = nn.Linear(lstm_hidden, n_pred)

        for name, p in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x   : (B, n_hist, N, F)
            adj : (N, N)  binary adjacency (no self-loops required — added here)

        Returns:
            (B, n_pred, N, 1)  — normalised speed predictions
        """
        B, T, N, F = x.shape

        # Self-loops prevent softmax(all -inf) → NaN for isolated nodes
        adj_self = (adj + torch.eye(N, device=adj.device)).clamp(0, 1)  # (N, N)

        # Vectorised spatial encoding: one GAT pass over all B*T "frames"
        h = x.reshape(B * T, N, F)          # (B*T, N, F)
        for layer in self.gat:
            h = layer(h, adj_self)           # (B*T, N, gat_out_dim)
        h = self.spatial_drop(h)

        # Back to (B, T, N, D), then (B*N, T, D) for LSTM
        D   = h.shape[-1]
        seq = h.reshape(B, T, N, D).permute(0, 2, 1, 3).reshape(B * N, T, D)

        lstm_out, _  = self.lstm(seq)        # (B*N, T, lstm_hidden)
        last_hidden  = lstm_out[:, -1, :]   # (B*N, lstm_hidden)

        pred_flat = self.output_proj(last_hidden)          # (B*N, n_pred)
        pred      = pred_flat.view(B, N, self.n_pred)
        pred      = pred.permute(0, 2, 1).unsqueeze(-1)   # (B, n_pred, N, 1)
        return pred
