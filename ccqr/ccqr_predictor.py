# ccqr/ccqr_predictor.py

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

# Import your helper functions
from .conformity.utils import generate_quantile_grid
from .conformity.scores import (
    calculate_asymmetric_scores,
    calculate_symmetric_scores
)
from .conformity.coverage import (
    compute_asymmetric_quantiles,
    compute_symmetric_quantiles
)

class CCQRpredictor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        model,
        quantiles=[0.05, 0.95],
        d=0.05,
        alpha=None,
        K=5,
        symmetric=True,
        strategy="score",
        adaptive=False,
    ):
        """
        Initializes the Split Composite Conformal Regressor (SCCRegressor).
        """
        self.model = model
        self.quantiles = quantiles
        self.alpha = (
            1 - (quantiles[1] - quantiles[0]) if alpha is None else alpha
        )
        self.d = d
        self.K = K
        self.strategy = strategy
        self.symmetric = symmetric
        self.adaptive = adaptive

        # Will store ( (q_low, q_high), coverage_quantile ) pairs
        self.coverage_quantiles = []

        # Generate the grid once
        if self.symmetric:
            self.grid_q = generate_quantile_grid(self.quantiles, self.d, self.K)
        else:
            self.grid_q = generate_asymmetric_quantile_grid(self.quantiles, self.d[0], self.d[1], self.K)

    def fit(self, X, y):
        """Fit the model."""
        self.model.fit(X, y)
        return self

    def calibrate(self, X_calib, y_calib):
        """Calibrate using a separate calibration set."""
        if self.symmetric:
            calibration_scores = calculate_symmetric_scores(
                self.model, X_calib, y_calib, self.grid_q,
                self.quantiles, self.adaptive
            )
            compute_symmetric_quantiles(
                calibration_scores, y_calib,
                self.grid_q, self.coverage_quantiles,
                self.strategy, self.alpha
            )
        else:
            lower_scores, upper_scores = calculate_asymmetric_scores(
                self.model, X_calib, y_calib, self.grid_q,
                self.quantiles, self.adaptive
            )
            compute_asymmetric_quantiles(
                lower_scores, upper_scores, y_calib,
                self.grid_q, self.coverage_quantiles,
                self.strategy, self.alpha
            )
        return self

    def predict(self, X):
        """Predict the conformal prediction intervals."""
        def get_bounds(y_pred_quantiles, y_pred_target, coverage_quantile):
            # In adaptive mode, scale_factor depends on the predicted range.
            if self.adaptive:
                scale_factor = y_pred_target[:, 1] - y_pred_target[:, 0]
            else:
                scale_factor = 1.0

            y_lower_pred = y_pred_quantiles[:, 0]
            y_upper_pred = y_pred_quantiles[:, 1]

            if self.symmetric:
                lower_bound = y_lower_pred - coverage_quantile * scale_factor
                upper_bound = y_upper_pred + coverage_quantile * scale_factor
            else:
                lower_bound = y_lower_pred - coverage_quantile[0] * scale_factor
                upper_bound = y_upper_pred + coverage_quantile[1] * scale_factor

            return lower_bound, upper_bound

        y_pred_target = self.model.predict(X, quantiles=self.quantiles)

        # Compute intervals for each pair (q, coverage_quantile)
        bounds = [
            get_bounds(
                self.model.predict(X, quantiles=[q_low, q_high]),
                y_pred_target,
                coverage_quantile
            )
            for (q_low, q_high), coverage_quantile in self.coverage_quantiles
        ]

        # Collect all lower/upper bounds
        all_lower_bounds, all_upper_bounds = zip(*bounds)

        # Average them
        cqr_lower_bound = np.mean(all_lower_bounds, axis=0)
        cqr_upper_bound = np.mean(all_upper_bounds, axis=0)

        return np.column_stack((cqr_lower_bound, cqr_upper_bound))
