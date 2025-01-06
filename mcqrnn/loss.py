import torch
import torch.nn as nn

class MCQRNNCheckLoss(nn.Module):
    """
    Monotone Composite Quantile Regression Neural Network (MCQRNN) "check loss"
    with Huber-style smoothing.

    Args:
        quantile (float): The quantile, sometimes referred to as tau, in [0, 1].
        alpha (float): The threshold for applying the quadratic vs. linear term (Huber).

    Forward call:
        predictions (torch.Tensor): [batch_size, ...]
        targets (torch.Tensor): [batch_size, ...]

    Returns:
        A scalar tensor (mean loss).
    """
    def __init__(self, quantile: float, alpha: float):
        super(MCQRNNCheckLoss, self).__init__()
        
        # Convert to tensor and register as buffer so they'll move with the module (e.g. .to(device))
        self.register_buffer('tau', torch.tensor(quantile, dtype=torch.float32))
        self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the MCQRNN check loss with a Huber-style smoothing.

        Returns:
            check_loss (torch.Tensor): scalar mean of the loss.
        """
        errors = targets - predictions
        
        # Huber portion
        # We'll break it down for clarity:
        # 1) Quadratic term for |error| <= alpha
        # 2) Linear term for |error| > alpha
        # We'll combine them elementwise via boolean masks.
        huber_quad = 0.5 * errors.pow(2) / self.alpha
        huber_lin = errors.abs() - 0.5 * self.alpha
        
        mask_quad = (errors.abs() <= self.alpha).float()
        mask_lin  = (errors.abs() >  self.alpha).float()

        huber = huber_quad * mask_quad + huber_lin * mask_lin

        # Separate out positive vs. negative errors for the “check” part
        # check_loss = sum( tau * huber if error >= 0 else (1 - tau) * huber ) / batch_size
        check_loss_pos = (self.tau * huber)[errors >= 0]
        check_loss_neg = ((1 - self.tau) * huber)[errors < 0]
        
        # Combine and normalize by batch size
        check_loss = (check_loss_pos.sum() + check_loss_neg.sum()) / huber.size(0)

        return check_loss