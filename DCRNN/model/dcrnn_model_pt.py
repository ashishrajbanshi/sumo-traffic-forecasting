import torch
import torch.nn as nn

from model.dcrnn_cell_pt import DCGRUCell


class DCRNNEncoder(nn.Module):
    """Stack of DCGRUCells unrolled over the input sequence."""

    def __init__(self, input_dim, num_units, num_nodes, supports,
                 max_diffusion_step, num_layers):
        super().__init__()
        self._num_nodes  = num_nodes
        self._num_units  = num_units
        self._num_layers = num_layers

        # First layer receives raw input; deeper layers receive previous layer's output
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            in_dim = input_dim if layer == 0 else num_units
            self.cells.append(
                DCGRUCell(in_dim, num_units, num_nodes, supports, max_diffusion_step)
            )

    def forward(self, inputs, hidden=None):
        """
        Args:
            inputs: (seq_len, B, num_nodes, input_dim)
            hidden: list of (B, num_nodes * num_units) per layer, or None

        Returns:
            hidden: list of final hidden states, one per layer
        """
        B = inputs.shape[1]
        if hidden is None:
            hidden = [torch.zeros(B, self._num_nodes * self._num_units,
                                  device=inputs.device)
                      for _ in self.cells]

        seq_len = inputs.shape[0]
        for t in range(seq_len):
            x = inputs[t]                                   # (B, N, input_dim)
            x = x.reshape(B, self._num_nodes * x.shape[-1]) # (B, N*F)
            new_hidden = []
            for layer_idx, cell in enumerate(self.cells):
                h = cell(x, hidden[layer_idx])              # (B, N*U)
                new_hidden.append(h)
                x = h                                       # feed into next layer
            hidden = new_hidden

        return hidden


class DCRNNDecoder(nn.Module):
    """
    Autoregressive decoder with optional scheduled sampling (curriculum learning).

    During training, with probability `cl_decay` the decoder feeds its own
    output back as the next input instead of the ground-truth value.
    At inference time it always feeds its own output back.
    """

    def __init__(self, output_dim, num_units, num_nodes, supports,
                 max_diffusion_step, num_layers, horizon):
        super().__init__()
        self._num_nodes  = num_nodes
        self._num_units  = num_units
        self._num_layers = num_layers
        self._horizon    = horizon
        self._output_dim = output_dim

        # Input to decoder: output_dim (prediction fed back)
        # First layer takes output_dim, deeper layers take num_units
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            in_dim = output_dim if layer == 0 else num_units
            self.cells.append(
                DCGRUCell(in_dim, num_units, num_nodes, supports, max_diffusion_step)
            )

        # Project hidden state → output (speed prediction per node)
        self.output_proj = nn.Linear(num_units, output_dim)

    def forward(self, encoder_hidden, labels=None, teacher_forcing_ratio=0.0):
        """
        Args:
            encoder_hidden:       list of (B, N*U) final encoder states per layer
            labels:               (horizon, B, N, output_dim) ground truth, or None
            teacher_forcing_ratio: probability of using ground truth at each step

        Returns:
            outputs: (horizon, B, N, output_dim)
        """
        B = encoder_hidden[0].shape[0]
        N = self._num_nodes
        device = encoder_hidden[0].device

        # Start token: zeros (go symbol)
        go = torch.zeros(B, N * self._output_dim, device=device)

        hidden  = list(encoder_hidden)
        outputs = []
        x = go

        for t in range(self._horizon):
            new_hidden = []
            for layer_idx, cell in enumerate(self.cells):
                h = cell(x, hidden[layer_idx])
                new_hidden.append(h)
                x = h
            hidden = new_hidden

            # Project top-layer hidden to output
            out = self.output_proj(
                hidden[-1].reshape(B * N, self._num_units)
            ).reshape(B, N, self._output_dim)              # (B, N, output_dim)

            outputs.append(out)

            # Scheduled sampling: choose next input
            if labels is not None and torch.rand(1).item() < teacher_forcing_ratio:
                x = labels[t].reshape(B, N * self._output_dim)
            else:
                x = out.reshape(B, N * self._output_dim).detach()

        return torch.stack(outputs, dim=0)   # (horizon, B, N, output_dim)


class DCRNNModel(nn.Module):
    """Full encoder-decoder DCRNN."""

    def __init__(self, input_dim, output_dim, num_units, num_nodes,
                 supports, max_diffusion_step, num_layers, seq_len, horizon):
        super().__init__()
        self.encoder = DCRNNEncoder(
            input_dim, num_units, num_nodes, supports, max_diffusion_step, num_layers
        )
        self.decoder = DCRNNDecoder(
            output_dim, num_units, num_nodes, supports, max_diffusion_step, num_layers, horizon
        )

    def forward(self, x, y=None, teacher_forcing_ratio=0.0):
        """
        Args:
            x:  (B, seq_len, N, input_dim)   — encoder input
            y:  (B, horizon, N, output_dim)  — decoder labels (training only)
            teacher_forcing_ratio: float [0,1]

        Returns:
            outputs: (B, horizon, N, output_dim)
        """
        # Encoder expects (seq_len, B, N, input_dim)
        x = x.permute(1, 0, 2, 3)

        enc_hidden = self.encoder(x)

        # Decoder labels: (horizon, B, N, output_dim)
        if y is not None:
            labels = y[..., :self.decoder._output_dim].permute(1, 0, 2, 3)
        else:
            labels = None

        out = self.decoder(enc_hidden, labels, teacher_forcing_ratio)
        return out.permute(1, 0, 2, 3)   # (B, horizon, N, output_dim)
