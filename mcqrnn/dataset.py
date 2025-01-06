import torch
import numpy as np
from torch.utils.data import Dataset


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
