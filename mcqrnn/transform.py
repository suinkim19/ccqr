import torch
from typing import Optional, Tuple, Union

def _mcqrnn_transform(
    x: torch.Tensor,
    taus: torch.Tensor,
    y: Optional[torch.Tensor] = None,
) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
           Tuple[torch.Tensor, torch.Tensor]]:
    """
    Transform x, y, taus into the trainable form (monotone composite quantile style)
    
    Args:
        x (torch.Tensor): Input of shape [N, d].
        taus (torch.Tensor): Quantiles of shape [T].
        y (Optional[torch.Tensor]): Target of shape [N, 1] (or [N]), if provided.
    
    Returns:
        If y is provided:
            (x_trans, y_trans, tau_trans): All expanded to shape [N*T, ...].
        Otherwise:
            (x_trans, tau_trans).
    """
    # Number of original samples and quantiles
    len_x = x.shape[0]
    len_taus = taus.shape[0]
    
    # Repeat x along the first dimension: [N -> N*T]
    # e.g., if x is [N, d], x_trans becomes [N*T, d].
    x_trans = x.repeat(len_taus, 1)
    
    # Tile or repeat taus along x dimension
    # taus: [T] -> [N*T], then make it [N*T, 1]
    tau_trans = taus.repeat(len_x).unsqueeze(-1)  # shape [N*T, 1]
    
    # Optionally transform y the same way
    if y is not None:
        # Ensure y is [N, 1], then repeat
        if y.ndim == 1:
            y = y.unsqueeze(-1)  # make sure it's [N, 1]
        y_trans = y.repeat(len_taus, 1)  # [N, 1] -> [N*T, 1]
        return x_trans, y_trans, tau_trans
    else:
        return x_trans, tau_trans


class DataTransformer:
    """
    A class to transform data into trainable form (MCQRNN-style).
    
    Args:
        x (torch.Tensor): Input features of shape [N, d].
        taus (torch.Tensor): Quantiles [T].
        y (Optional[torch.Tensor]): Targets of shape [N, 1] or [N].
    
    Methods:
        __call__():
            Return the transformed (x_trans, y_trans, tau_trans) or (x_trans, tau_trans).
        transform(x, input_taus):
            Return a re-transformed version of x with new quantiles.
    """

    def __init__(
        self,
        x: torch.Tensor,
        taus: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ):
        self.x = x
        self.y = y
        self.taus = taus

        # Apply the internal transform once at construction
        out = _mcqrnn_transform(x=self.x, taus=self.taus, y=self.y)
        
        if y is not None:
            self.x_trans, self.y_trans, self.tau_trans = out
        else:
            self.x_trans, self.tau_trans = out
            self.y_trans = None

    def __call__(self) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                                Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns the pre-transformed data from initialization.
        """
        if self.y_trans is not None:
            return self.x_trans, self.y_trans, self.tau_trans
        else:
            return self.x_trans, self.tau_trans

    def transform(
        self,
        x: torch.Tensor,
        input_taus: torch.Tensor
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
               Tuple[torch.Tensor, torch.Tensor]]:
        """
        Transform a new batch of x with a new set of quantiles.
        
        Args:
            x (torch.Tensor): [M, d]
            input_taus (torch.Tensor): [K]
        
        Returns:
            (x_trans, tau_trans) or (x_trans, y_trans, tau_trans) if y is provided.
            In this minimal example, we do not provide y, so typically it is
            just (x_trans, tau_trans).
        """
        return _mcqrnn_transform(x=x, taus=input_taus)
