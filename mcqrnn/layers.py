import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int):
    """
    Utility function to set the random seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MCQRNN(nn.Module):
    def __init__(
        self, m_input_size, i_input_size, hidden_size, seed=None, dropout_rate=0.4
    ):
        """
        Monotone MLP with two hidden layers.
        - Monotonic part (m_input_size) uses softplus for learned parameters
            to ensure monotonicity w.r.t. the scalar input x_m.
        - Non-monotonic part (i_input_size) is unconstrained.

        hidden_size: number of neurons in the hidden layer
        dropout_rate: dropout rate for dropout layer
        """
        super(MCQRNN, self).__init__()
        if seed is not None:
            set_seed(seed)
        self.m_matrix = nn.Parameter(torch.randn(hidden_size, m_input_size))
        self.i_matrix = nn.Parameter(torch.randn(hidden_size, i_input_size))
        self.out_m = nn.Parameter(torch.randn(1, hidden_size))
        self.bias_h = nn.Parameter(torch.zeros(hidden_size))
        self.bias_o = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x_m, x_i):
        """
        Forward pass with two hidden layers.

        Args:
            x_m: shape [batch_size], the scalar monotonic input (quantile).
            x_i: shape [batch_size, i_input_size], the non-monotonic inputs.

        Returns:
            A tensor of shape [batch_size].
        """
        x_m = x_m.unsqueeze(1)
        m_output = F.linear(x_m, F.softplus(self.m_matrix), None)
        i_output = F.linear(x_i, self.i_matrix, None)
        combined_output = m_output + i_output
        hidden_output = self.dropout(
            torch.tanh(combined_output + self.bias_h.unsqueeze(0))
        )
        # hidden_output = torch.tanh(combined_output + self.bias_h.unsqueeze(0))
        out_output = F.linear(hidden_output, F.softplus(self.out_m), bias=self.bias_o)
        return out_output.flatten()
