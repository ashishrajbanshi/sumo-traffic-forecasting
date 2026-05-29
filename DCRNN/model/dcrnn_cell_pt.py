import torch
import torch.nn as nn


class DCGRUCell(nn.Module):
    """
    Diffusion Convolutional GRU cell.

    Replaces the standard GRU's matrix multiplications with graph diffusion
    convolutions.  For each gate/candidate the cell:
      1. Concatenates inputs x and hidden state h  → shape (B, N, F)
      2. Diffuses through each support matrix K times → num_matrices feature sets
      3. Flattens to (B*N, num_matrices*F) and applies a learned linear layer
      4. Reshapes output back to (B, N, num_units)

    Supports are registered as buffers so they automatically move to GPU.
    """

    def __init__(self, input_dim, num_units, num_nodes, supports, max_diffusion_step):
        """
        Args:
            input_dim:           features per node per timestep (e.g. 2: speed + time-of-day)
            num_units:           hidden size per node (rnn_units in config)
            num_nodes:           number of graph nodes
            supports:            list of scipy sparse matrices (D^{-1}A, D^{-1}A^T)
            max_diffusion_step:  K — number of hops per support (K=1 → 1 hop)
        """
        super().__init__()
        self._input_dim = input_dim
        self._num_units = num_units
        self._num_nodes = num_nodes
        self._max_diffusion_step = max_diffusion_step
        self._num_supports = len(supports)

        # Register each support as a non-trainable buffer (moves with .to(device))
        for i, s in enumerate(supports):
            self.register_buffer(f'support_{i}', s)

        # Number of feature sets after diffusion:
        #   1 (original x0) + K steps × num_supports
        num_matrices = self._num_supports * max_diffusion_step + 1
        gconv_in = num_matrices * (input_dim + num_units)

        # One linear for reset+update gates combined (split after sigmoid)
        self.W_gate = nn.Linear(gconv_in, 2 * num_units)
        # One linear for the candidate hidden state
        self.W_cand = nn.Linear(num_matrices * (input_dim + num_units), num_units)

        # High initial gate bias helps the cell pass gradients early in training
        nn.init.constant_(self.W_gate.bias, 1.0)
        nn.init.zeros_(self.W_cand.bias)

    # ── Diffusion ───────────────────────────────────────────────────────────────

    def _diffuse(self, x, h):
        """
        Concatenate x and h, apply graph diffusion, return flat features.

        Args:
            x: (B, N, input_dim)
            h: (B, N, num_units)   — either full h or r*h for candidate

        Returns:
            (B*N, num_matrices * (input_dim + num_units))
        """
        B, N = x.shape[0], self._num_nodes
        inp = torch.cat([x, h], dim=-1)   # (B, N, F)
        F   = inp.shape[-1]

        x0 = inp                          # (B, N, F) — zero-hop (identity)
        feature_sets = [x0]

        for i in range(self._num_supports):
            sup = getattr(self, f'support_{i}')   # dense (N, N)
            x_k = x0                              # reset for each support
            for _ in range(self._max_diffusion_step):
                # Apply dense (N,N) @ (B,N,F):
                #   reshape → (N, B*F), matmul, reshape back
                x_k = (sup @ x_k.permute(1, 0, 2).reshape(N, B * F)
                        ).reshape(N, B, F).permute(1, 0, 2)   # (B, N, F)
                feature_sets.append(x_k)

        # (B, N, num_matrices*F) → (B*N, num_matrices*F)
        return torch.cat(feature_sets, dim=-1).reshape(B * N, -1)

    # ── Forward pass ───────────────────────────────────────────────────────────

    def forward(self, inputs, state):
        """
        Args:
            inputs: (B, num_nodes * input_dim)
            state:  (B, num_nodes * num_units)

        Returns:
            new_state: (B, num_nodes * num_units)
        """
        B, N, U = inputs.shape[0], self._num_nodes, self._num_units

        x = inputs.reshape(B, N, self._input_dim)
        h = state.reshape(B, N, U)

        # Gates — shared diffusion of [x, h]
        feat_gate = self._diffuse(x, h)                          # (B*N, gconv_in)
        gates     = torch.sigmoid(self.W_gate(feat_gate))        # (B*N, 2*U)
        r, u      = gates.split(U, dim=-1)                       # each (B*N, U)
        r = r.reshape(B, N, U)
        u = u.reshape(B, N, U)

        # Candidate — diffusion of [x, r*h]
        feat_cand = self._diffuse(x, r * h)
        c = torch.tanh(self.W_cand(feat_cand)).reshape(B, N, U)  # (B, N, U)

        new_h = u * h + (1.0 - u) * c
        return new_h.reshape(B, N * U)

    @property
    def hidden_size(self):
        return self._num_nodes * self._num_units
