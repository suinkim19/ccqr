from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
from sklearn.model_selection import train_test_split
import copy
import warnings
import torch
from tqdm.autonotebook import tqdm

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
