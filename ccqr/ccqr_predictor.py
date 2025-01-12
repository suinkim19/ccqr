from .calibration import generate_quantile_grid, calculate_asymmetric_scores, calculate_symmetric_scores
from .prediction import compute_asymmetric_quantiles, compute_symmetric_quantiles

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

        Parameters:
        - model: A scikit-learn compatible regression model.
        - quantiles: A list containing two quantiles for prediction intervals (default is [0.05, 0.95]).
        - strategy: The strategy to select the coverage quantile ('average', 'widest', 'narrowest', 'asymmetric', 'score', 'quantile').
        - adaptive: Whether to use adaptive conformity functions (default is False).
        """
        self.model = model
        self.quantiles = quantiles
        self.alpha = (
            1 - (quantiles[1] - quantiles[0]) if alpha == None else alpha
        )  # Calculate alpha based on quantiles
        self.d = d
        self.K = K
        self.coverage_quantiles = []
        self.strategy = strategy
        self.symmetric = symmetric
        self.adaptive = adaptive
        self.grid_q = self._generate_quantile_grid()

    def fit(self, X, y):
        """
        Fits the base model on the training data.

        Parameters:
        - X: Training features.
        - y: Training labels.
        """
        self.model.fit(X, y)
        return self

    def calibrate(self, X_calib, y_calib):
        """
        Calibrates the model using a separate calibration set to compute conformity scores.
        Computes coverage quantiles for all pairs in grid_q.

        Parameters:
        - X_calib: Calibration features.
        - y_calib: Calibration labels.
        """
        # Calculate coverage quantiles for all grid pairs
        if self.symmetric:
            calibration_scores = self._calculate_symmetric_scores(X_calib, y_calib)
            self._compute_symmetric_quantiles(calibration_scores, y_calib)
        else:
            lower_scores, upper_scores = self._calculate_asymmetric_scores(
                X_calib, y_calib
            )
            self._compute_asymmetric_quantiles(lower_scores, upper_scores, y_calib)

        return self

    def predict(self, X):
        """
        Predicts the conformal prediction intervals for new data.

        Parameters:
        - X: New input features.

        Returns:
        - A tuple (lower_bound, upper_bound) representing the prediction intervals.
        """

        def get_bounds(y_pred_quantiles, y_pred_target, coverage_quantile):
            if self.adaptive:
                scale_factor = y_pred_target[:, 1] - y_pred_target[:, 0]
            else:
                scale_factor = 1
            y_lower_pred, y_upper_pred = y_pred_quantiles[:, 0], y_pred_quantiles[:, 1]
            if self.symmetric:
                lower_bound = y_lower_pred - coverage_quantile * scale_factor
                upper_bound = y_upper_pred + coverage_quantile * scale_factor
            else:
                lower_bound = y_lower_pred - coverage_quantile[0] * scale_factor
                upper_bound = y_upper_pred + coverage_quantile[1] * scale_factor
            return lower_bound, upper_bound

        y_pred_target = self.model.predict(X, quantiles=self.quantiles)
        bounds = [
            get_bounds(
                self.model.predict(X, quantiles=[q[0], q[1]]),
                y_pred_target,
                coverage_quantile,
            )
            for q, coverage_quantile in self.coverage_quantiles
        ]

        all_lower_bounds, all_upper_bounds = zip(*bounds)

        # Average the coverage quantiles
        cqr_lower_bound = np.mean(all_lower_bounds, axis=0)
        cqr_upper_bound = np.mean(all_upper_bounds, axis=0)

        return np.column_stack((cqr_lower_bound, cqr_upper_bound))