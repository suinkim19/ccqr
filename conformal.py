from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
from sklearn.model_selection import train_test_split
import copy
import warnings
import torch
from tqdm.autonotebook import tqdm


def symmetric_conformity_score(y_true, y_pred_q):
  # Conformity score for CCQR
    y_lower_pred, y_upper_pred = y_pred_q[:, 0], y_pred_q[:, 1]
    return np.maximum(y_lower_pred - y_true, y_true - y_upper_pred)


def symmetric_adaptive_conformity_score(y_true, y_pred_q, y_pred_target):
  # Conformity score for adaptive CCQR
    y_lower_pred, y_upper_pred = y_pred_q[:, 0], y_pred_q[:, 1]
    scale_factor = y_pred_target[:, 1] - y_pred_target[:, 0]
    return np.maximum(
        (y_lower_pred - y_true) / scale_factor, (y_true - y_upper_pred) / scale_factor
    )

def asymmetric_conformity_score(y_true, y_pred_q):
    # Conformity score for asymmetric CCQR
    lower_conformity_scores = y_pred_q[:, 0] - y_true
    upper_conformity_scores = y_true - y_pred_q[:, 1]
    return lower_conformity_scores, upper_conformity_scores

# Conformal predicion with mean regressor
class CPpredictor(BaseEstimator, RegressorMixin):
    def __init__(self, model, quantiles=[0.05, 0.95]):
        """
        Initializes the Split Conformal Mean Regressor.
        
        Parameters:
        - model: A scikit-learn compatible regression model.
        - quantiles: A list containing two quantiles for prediction intervals (default is [0.05, 0.95]).
        """
        self.model = model
        self.quantiles = quantiles
        self.alpha = 1 - (quantiles[1] - quantiles[0])

    def calibrate(self, X_calib, y_calib):
        """
        Parameters:
        - X_calib: Calibration features.
        - y_calib: Calibration labels.
        """
        # Predict lower and upper quantiles using the base model
        y_pred = self.model.predict(X_calib)
        # Compute conformity scores
        self.calibration_scores = np.abs(y_pred - y_calib)
        # Determine the empirical quantile for the desired coverage level
        self.coverage_quantile = np.quantile(
            self.calibration_scores,
            (1 - self.alpha) * (1 + 1 / len(self.calibration_scores)),
        )
        return self

    def predict(self, X):
        """
        Parameters:
        - X: New input features.
        """
        # Predict lower and upper quantiles using the base model
        y_pred = self.model.predict(X)

        # Create lower and upper bounds using the coverage quantile
        lower_bound = y_pred - self.coverage_quantile
        upper_bound = y_pred + self.coverage_quantile
        return np.column_stack((lower_bound, upper_bound))


# Conformalized quantile regression
class CQRpredictor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        model,
        quantiles=[0.05, 0.95],
        alpha=None,
        CV=False,
        range_vals=0.05,
        num_vals=5,
    ):
        """
        Initializes the Split Conformalized Quantile Regression.

        Parameters:
        - model: A scikit-learn compatible regression model.
        - conformity_score_func: A function to compute conformity scores.
        - quantiles: A list containing two quantiles for prediction intervals (default is [0.05, 0.95]).
        - CV: Boolean indicating whether to perform cross-validation for quantile tuning.
        - range_vals: The range within which to adjust quantiles for cross-validation (default is 0.05).
        - num_vals: The number of quantile values to consider in cross-validation (default is 5).
        """
        self.model = model
        self.conformity_score_func = symmetric_conformity_score
        self.quantiles = quantiles
        self.alpha = ( 1 - (quantiles[1] - quantiles[0]) if alpha == None else alpha )
        self.CV = CV
        self.range_vals = range_vals
        self.num_vals = num_vals
        self.coverage_quantile = None # Initialization
        self.calibration_scores = None # Initialization
        self.optimal_quantiles = quantiles  # Initialize with provided quantiles

    def fit(self, X, y):
        """
        Fits the base model on the training data.

        Parameters:
        - X: Training features.
        - y: Training labels.
        """
        if self.CV:
            self.cross_validate(X, y)
        self.model.fit(X, y)
        return self

    def calibrate(self, X_calib, y_calib):
        """
        Calibrates the model using a separate calibration set to compute conformity scores.
        If CV is enabled, it performs cross-validation to find the optimal quantiles.

        Parameters:
        - X_calib: Calibration features.
        - y_calib: Calibration labels.
        """

        # Use the optimal quantiles if CV was performed, else use the initial quantiles
        quantiles_to_use = self.optimal_quantiles if self.CV else self.quantiles

        # Predict lower and upper quantiles using the base model
        y_pred_quantiles = self.model.predict(
            X_calib, quantiles=[quantiles_to_use[0], quantiles_to_use[1]]
        )
        y_lower_pred, y_upper_pred = y_pred_quantiles[:, 0], y_pred_quantiles[:, 1]

        # Compute conformity scores
        self.calibration_scores = np.maximum(
            y_lower_pred - y_calib, y_calib - y_upper_pred
        )

        # Determine the empirical quantile for the desired coverage level
        self.coverage_quantile = np.quantile(
            self.calibration_scores,
            (1 - self.alpha) * (1 + 1 / len(self.calibration_scores)),
        )
        return self

    def cross_validate(self, X, y, test_ratio=0.3, random_state=0):
        """
        Parameters:
        - X: Training features.
        - y: Training labels.
        - test_ratio: The ratio for validation data.
        - random_state: Seed (default is 0).
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_ratio, random_state=random_state
        )

        grid_q_low = np.linspace(
            self.quantiles[0], self.quantiles[0] + self.range_vals, self.num_vals
        ).reshape(-1, 1)
        grid_q_high = np.linspace(
            self.quantiles[1], self.quantiles[1] - self.range_vals, self.num_vals
        ).reshape(-1, 1)
        grid_q = np.concatenate((grid_q_low, grid_q_high), axis=1)

        best_interval_width = float("inf")
        cv_model = copy.deepcopy(self.model)
        cv_model.fit(X_train, y_train)
        # Iterate through the grid of quantile pairs
        for q in grid_q:
            # Predict with the candidate quantiles
            y_pred_quantiles = cv_model.predict(X_test, quantiles=[q[0], q[1]])
            y_lower_pred, y_upper_pred = y_pred_quantiles[:, 0], y_pred_quantiles[:, 1]
            coverage = np.mean((y_test >= y_lower_pred) & (y_test <= y_upper_pred))
            # Calculate interval width
            interval_width = np.mean(y_upper_pred - y_lower_pred)
            # Update best quantiles if this interval is shorter
            if coverage >= 1 - self.alpha and interval_width < best_interval_width:
                best_interval_width = interval_width
                self.optimal_quantiles = q

    def predict(self, X):
        """
        Parameters:
        - X: New input features.
        """
        # Use the optimal quantiles if CV was performed, else use the initial quantiles
        quantiles_to_use = self.optimal_quantiles if self.CV else self.quantiles

        # Predict lower and upper quantiles using the base model
        y_pred_quantiles = self.model.predict(
            X, quantiles=[quantiles_to_use[0], quantiles_to_use[1]]
        )
        y_lower_pred, y_upper_pred = y_pred_quantiles[:, 0], y_pred_quantiles[:, 1]

        # Create lower and upper bounds using the coverage quantile
        lower_bound = y_lower_pred - self.coverage_quantile
        upper_bound = y_upper_pred + self.coverage_quantile
        return np.column_stack((lower_bound, upper_bound))


# Conformalized composite quantile regression
class CCQRpredictor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        model,
        quantiles=[0.05, 0.95],
        d=0.2,
        alpha=None,
        K=5,
        symmetric=True,
        average="score",
        adaptive=False,
    ):
        """
        Initializes the Split Conformalized Composite Quantile Regression.

        Parameters:
        - model: A scikit-learn compatible regression model.
        - quantiles: A list containing two quantiles for prediction intervals (default is [0.05, 0.95]).
        - d: The range of combining quantiles (default is 0.2).
        - alpha: The desired coverage rate (default is 1 - (quantiles[1] - quantiles[0])).
        - K: The number of combining quantiles (default is 9).
        - symmetric: Whether to use symmetric conformity fucntions (default is True).
        - average: The type of averaging strategy (default is score).
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
        self.average = average
        self.symmetric = symmetric
        self.adaptive = adaptive
        self.grid_q = self._generate_quantile_grid()

    def fit(self, X, y):
        """
        Parameters:
        - X: Training features.
        - y: Training labels.
        """
        self.model.fit(X, y)
        return self

    def calibrate(self, X_calib, y_calib):
        """
        Parameters:
        - X_calib: Calibration features.
        - y_calib: Calibration labels.
        """
  
        if self.symmetric:
            calibration_scores = self._calculate_symmetric_scores(X_calib, y_calib)
            self._compute_symmetric_quantiles(calibration_scores, y_calib)
        else:
            lower_scores, upper_scores = self._calculate_asymmetric_scores(
                X_calib, y_calib
            )
            self._compute_asymmetric_quantiles(lower_scores, upper_scores, y_calib)

        return self

    def _generate_quantile_grid(self):
        """
        Generates a grid of quantile pairs for calibration.
        """
        grid_q_low = np.linspace(
            self.quantiles[0],
            self.quantiles[0] + self.d,
            self.K,
        ).reshape(-1, 1)
        grid_q_high = np.linspace(
            self.quantiles[1],
            self.quantiles[1] - self.d,
            self.K,
        ).reshape(-1, 1)
        return np.concatenate((grid_q_low, grid_q_high), axis=1)

    def _calculate_asymmetric_scores(self, X_calib, y_calib):
        """
        Calculates asymmetric conformity scores for calibration.
        """
        lower_scores, upper_scores = [], []
        for q in self.grid_q:
            y_pred_quantiles = self.model.predict(X_calib, quantiles=[q[0], q[1]])
            y_pred_target = self.model.predict(X_calib, quantiles=self.quantiles)
            if self.adaptive:
                lower_score, upper_score = asymmetric_adaptive_conformity_score(
                    y_true=y_calib,
                    y_pred_q=y_pred_quantiles,
                    y_pred_target=y_pred_target,
                )
            else:
                lower_score, upper_score = asymmetric_conformity_score(
                    y_true=y_calib, y_pred_q=y_pred_quantiles
                )
            lower_scores.append(lower_score)
            upper_scores.append(upper_score)
        return lower_scores, upper_scores

    def _calculate_symmetric_scores(self, X_calib, y_calib):
        """
        Calculates symmetric conformity scores for calibration.
        """
        calibration_scores = []
        for q in self.grid_q:
            y_pred_quantiles = self.model.predict(X_calib, quantiles=[q[0], q[1]])
            if self.adaptive:
                y_pred_target = self.model.predict(X_calib, quantiles=self.quantiles)
                calibration_score = symmetric_adaptive_conformity_score(
                    y_true=y_calib,
                    y_pred_q=y_pred_quantiles,
                    y_pred_target=y_pred_target,
                )
            else:
                calibration_score = symmetric_conformity_score(
                    y_true=y_calib, y_pred_q=y_pred_quantiles
                )
            calibration_scores.append(calibration_score)
        return calibration_scores

    def _compute_asymmetric_quantiles(self, lower_scores, upper_scores, y_calib):
        """
        Computes asymmetric coverage quantiles based on the average.
        """
        if self.average == "quantile":
            for q, lower_score, upper_score in zip(
                self.grid_q, lower_scores, upper_scores
            ):
                lower_coverage_quantile = np.quantile(
                    lower_score, (1 - self.alpha / 2) * (1 + 1 / len(y_calib))
                )
                upper_coverage_quantile = np.quantile(
                    upper_score, (1 - self.alpha / 2) * (1 + 1 / len(y_calib))
                )
                coverage_quantile = np.array(
                    [lower_coverage_quantile, upper_coverage_quantile]
                )
                self.coverage_quantiles.append((q, coverage_quantile))
        elif self.average == "score":
            lower_coverage_quantile = np.quantile(
                np.mean(np.array(lower_scores), axis=0),
                (1 - self.alpha / 2) * (1 + 1 / len(y_calib)),
            )
            upper_coverage_quantile = np.quantile(
                np.mean(np.array(upper_scores), axis=0),
                (1 - self.alpha / 2) * (1 + 1 / len(y_calib)),
            )
            coverage_quantile = np.array(
                [lower_coverage_quantile, upper_coverage_quantile]
            )
            for q in self.grid_q:
                self.coverage_quantiles.append((q, coverage_quantile))

    def _compute_symmetric_quantiles(self, calibration_scores, y_calib):
        """
        Computes symmetric coverage quantiles based on the average.
        """
        if self.average == "quantile":
            for q, score in zip(self.grid_q, calibration_scores):
                coverage_quantile = np.quantile(
                    score, (1 - self.alpha) * (1 + 1 / len(y_calib))
                )
                self.coverage_quantiles.append((q, coverage_quantile))
        elif self.average == "score":
            coverage_quantile = np.quantile(
                np.mean(np.array(calibration_scores), axis=0),
                (1 - self.alpha) * (1 + 1 / len(y_calib)),
            )
            for q in self.grid_q:
                self.coverage_quantiles.append((q, coverage_quantile))

    def predict(self, X):
        """
        Parameters:
        - X: New input features.
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



class DCPpredictor(BaseEstimator, RegressorMixin):

    def __init__(
        self, model, quantiles=[0.05, 0.95], grid_q=np.arange(0.01, 1, 0.01).tolist()
    ):
        """
        Initializes Distributional Conformal Prediction

        Parameters:
        - model: A quantile regression model.
        - quantiles: A list containing two quantiles for prediction intervals (default is [0.05, 0.95]).
        - grid_q: A list containing quantiles used in DCP (default is [0.01, 0.02, ..., 0.99]).
        """
        self.model = model
        self.quantiles = quantiles
        self.alpha = 1 - (
            quantiles[1] - quantiles[0]
        )  # Calculate alpha based on quantiles
        self.grid_q = grid_q

    def calibrate(self, X_calib, y_calib):
        """
        Parameters:
        - X_calib: Calibration features.
        - y_calib: Calibration labels.
        """

        # Predict lower and upper quantiles using the base model
        Q_yx = self.model.predict(X_calib, quantiles=self.grid_q)
        u_hat = np.mean((Q_yx <= y_calib[:, None]), axis=1)
        cs = np.abs(u_hat - 0.5)
        # Compute conformity scores
        self.calibration_scores = cs

        # Determine the empirical quantile for the desired coverage level
        self.threshold = np.quantile(cs, (1 - self.alpha) * (1 + 1 / len(cs)))
        return self

    def predict(self, X):
        """
        Parameters:
        - X: New input features.
        """
      
        Q_yx = self.model.predict(X, quantiles=self.grid_q)

        ci_grid = np.abs(np.array(self.grid_q) - 0.5)

        lower_bound = np.full(X.shape[0], np.nan)
        upper_bound = np.full(X.shape[0], np.nan)
        for i in range(X.shape[0]):
            ci = Q_yx[i, ci_grid <= self.threshold]
            if len(ci) > 0:
                lower_bound[i] = np.min(ci)
                upper_bound[i] = np.max(ci)

        # Create lower and upper bounds using the coverage quantile
        return np.column_stack((lower_bound, upper_bound))
