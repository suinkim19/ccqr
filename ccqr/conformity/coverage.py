# ccqr/conformity/coverage.py

import numpy as np

def compute_asymmetric_quantiles(
    lower_scores,
    upper_scores,
    y_calib,
    grid_q,
    coverage_quantiles_storage,
    strategy,
    alpha
):
    """
    Updates 'coverage_quantiles_storage' in-place with (q, coverage_quantile) pairs.
    """
    if strategy == "quantile":
        for q, lower_score, upper_score in zip(grid_q, lower_scores, upper_scores):
            lower_coverage_quantile = np.quantile(
                lower_score,
                (1 - alpha / 2) * (1 + 1 / len(y_calib))
            )
            upper_coverage_quantile = np.quantile(
                upper_score,
                (1 - alpha / 2) * (1 + 1 / len(y_calib))
            )
            coverage_quantiles_storage.append((
                (q[0], q[1]),
                np.array([lower_coverage_quantile, upper_coverage_quantile])
            ))
    elif strategy == "score":
        # Example: average across all pairs, then apply quantile
        lower_mean = np.mean(np.array(lower_scores), axis=0)
        upper_mean = np.mean(np.array(upper_scores), axis=0)

        lower_coverage_quantile = np.quantile(
            lower_mean,
            (1 - alpha / 2) * (1 + 1 / len(y_calib))
        )
        upper_coverage_quantile = np.quantile(
            upper_mean,
            (1 - alpha / 2) * (1 + 1 / len(y_calib))
        )
        coverage_quantile = np.array([lower_coverage_quantile, upper_coverage_quantile])

        for q in grid_q:
            coverage_quantiles_storage.append(((q[0], q[1]), coverage_quantile))


def compute_symmetric_quantiles(
    calibration_scores,
    y_calib,
    grid_q,
    coverage_quantiles_storage,
    strategy,
    alpha
):
    """
    Updates 'coverage_quantiles_storage' in-place with (q, coverage_quantile) pairs.
    """
    if strategy == "quantile":
        for q, score in zip(grid_q, calibration_scores):
            coverage_quantile = np.quantile(
                score,
                (1 - alpha) * (1 + 1 / len(y_calib))
            )
            coverage_quantiles_storage.append(((q[0], q[1]), coverage_quantile))
    elif strategy == "score":
        # Example: average across all pairs, then apply quantile
        mean_score = np.mean(np.array(calibration_scores), axis=0)
        coverage_quantile = np.quantile(
            mean_score,
            (1 - alpha) * (1 + 1 / len(y_calib))
        )
        for q in grid_q:
            coverage_quantiles_storage.append(((q[0], q[1]), coverage_quantile))
