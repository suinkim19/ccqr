# ccqr/conformity/utils.py

import numpy as np

def generate_quantile_grid(quantiles, d, K):
    """
    Generates a grid of quantile pairs for calibration.
    """
    q_low_start, q_high_start = quantiles
    grid_q_low = np.linspace(q_low_start, q_low_start + d, K).reshape(-1, 1)
    grid_q_high = np.linspace(q_high_start, q_high_start - d, K).reshape(-1, 1)
    return np.concatenate((grid_q_low, grid_q_high), axis=1)
    
def generate_asymmetric_quantile_grid(quantiles, d1, d2, K):
    """
    Generates a grid of quantile pairs for calibration.
    """
    q_low_start, q_high_start = quantiles
    grid_q_low = np.linspace(q_low_start, q_low_start + d1, K).reshape(-1, 1)
    grid_q_high = np.linspace(q_high_start, q_high_start - d2, K).reshape(-1, 1)
    return np.concatenate((grid_q_low, grid_q_high), axis=1)
