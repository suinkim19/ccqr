import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

def set_seed(seed: int):
    """
    Utility function to set the random seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MCQRNN_Dataset(Dataset):
    """
    A PyTorch Dataset to expand X and y by the given quantiles.
    Repeats X for each quantile and tiles y similarly.
    """
    def __init__(self, X, y, quantiles):
        """
        X : np.ndarray of shape [N, D]
        y : np.ndarray of shape [N]
        quantiles : np.ndarray of shape [Q, 1]
        """
        # data_m (monotonic input) has shape [N*Q]
        self.data_m = torch.tensor(
            np.repeat(np.array(quantiles), len(y)), dtype=torch.float32
        )
        # data_i (main input) has shape [N*Q, D]
        self.data_i = torch.tensor(np.tile(X, (len(quantiles), 1)), dtype=torch.float32)
        # labels has shape [N*Q]
        self.labels = torch.tensor(np.tile(y, len(quantiles)), dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data_m[idx], self.data_i[idx, :], self.labels[idx]


class MCQRNN(nn.Module):
    """
    Monotone MLP with two hidden layers.
    - Monotonic part (m_input_size) uses softplus for learned parameters 
      to ensure monotonicity w.r.t. the scalar input x_m.
    - Non-monotonic part (i_input_size) is unconstrained.

    hidden_size1: number of neurons in the first hidden layer
    hidden_size2: number of neurons in the second hidden layer
    """

    def __init__(
        self,
        m_input_size: int,
        i_input_size: int,
        hidden_size1: int,
        hidden_size2: int,
        seed: int = None,
        dropout_rate: float = 0.4,
    ):
        super(MCQRNN, self).__init__()
        if seed is not None:
            set_seed(seed)

        # --------------------------
        # First hidden layer
        # --------------------------
        # Monotonic part: shape [hidden_size1, m_input_size]
        self.m_matrix_1 = nn.Parameter(torch.randn(hidden_size1, m_input_size))
        # Non-monotonic part: shape [hidden_size1, i_input_size]
        self.i_matrix_1 = nn.Parameter(torch.randn(hidden_size1, i_input_size))
        self.bias_h1 = nn.Parameter(torch.zeros(hidden_size1))

        # --------------------------
        # Second hidden layer
        # --------------------------
        # shape [hidden_size2, hidden_size1]
        self.hidden2_matrix = nn.Parameter(torch.randn(hidden_size2, hidden_size1))
        self.bias_h2 = nn.Parameter(torch.zeros(hidden_size2))

        # --------------------------
        # Output layer
        # --------------------------
        # Monotonic part for final output: shape [1, hidden_size2]
        self.out_m = nn.Parameter(torch.randn(1, hidden_size2))
        self.bias_o = nn.Parameter(torch.zeros(1))

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x_m: torch.Tensor, x_i: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with two hidden layers.

        Args:
            x_m: shape [batch_size], the scalar monotonic input (quantile).
            x_i: shape [batch_size, i_input_size], the non-monotonic inputs.

        Returns:
            A tensor of shape [batch_size].
        """
        # Expand x_m to shape [batch_size, 1] for matmul
        x_m = x_m.unsqueeze(1)

        # ========== First Hidden Layer ==========
        # Softplus to enforce monotonic connections w.r.t. x_m
        m_output_1 = F.linear(x_m, F.softplus(self.m_matrix_1), bias=None)
        # Non-monotonic part, no constraint
        i_output_1 = F.linear(x_i, self.i_matrix_1, bias=None)

        combined_1 = m_output_1 + i_output_1 + self.bias_h1.unsqueeze(0)
        hidden_1 = torch.tanh(combined_1)
        hidden_1 = self.dropout(hidden_1)

        # ========== Second Hidden Layer ==========
        # Unconstrained, but we can do a standard linear + tanh
        combined_2 = F.linear(hidden_1, self.hidden2_matrix, bias=None) + self.bias_h2
        hidden_2 = torch.tanh(combined_2)
        hidden_2 = self.dropout(hidden_2)

        # ========== Output Layer ==========
        # Again apply softplus to out_m for monotonic connection
        out_output = F.linear(hidden_2, F.softplus(self.out_m), bias=self.bias_o)

        return out_output.flatten()
