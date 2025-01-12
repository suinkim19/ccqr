# Conformalized Composite Quantile Regression (CCQR)

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**CCQR** (Conformalized Composite Quantile Regression) is a Python package for generating reliable prediction intervals using a combination of multiple quantile regression models and conformal predictions.

## Installation

1. **Clone this repository (optional if installing directly from GitHub)**:
    ```bash
    git clone https://github.com/suinkim19/ccqr.git
    cd ccqr
    ```
2. **Install via `pip`**:
    ```bash
    pip install .
    ```
    Or directly from GitHub:
    ```bash
    pip install git+https://github.com/suinkim19/ccqr.git
    ```
    
## Basic Usage

Below is a quick-start example demonstrating how to use the `CCQRpredictor`:

```python
import numpy as np
from ccqr import CCQRpredictor

# Suppose you have a scikit-learn-compatible quantile regression model
# which can predict quantiles via .predict(X, quantiles=[q1, q2]).
from your_quantile_model import YourQuantileRegressor

# 1. Initialize your base quantile regressor
base_model = YourQuantileRegressor() # Some scikit-like regressor supporting predict().

# 2. Initialize CCQRpredictor
ccqr_est = CCQRpredictor(
    model=base_model,
    quantiles=[0.05, 0.95],
    d=0.05,
    alpha=None,
    K=5,
    symmetric=True,
    strategy="score",
    adaptive=False
)

# 3. Fit the model
X_train = np.random.randn(100, 10)  # Dummy training features
y_train = np.random.randn(100)      # Dummy training targets
base_model.fit(X_train, y_train) # or you can use ccqr_est.fit(X_train, y_train).

# 4. Calibrate the model
X_calib = np.random.randn(50, 10)  # Dummy calibration features
y_calib = np.random.randn(50)      # Dummy calibration targets
ccqr_est.calibrate(X_calib, y_calib)

# 5. Predict intervals on new data
X_test = np.random.randn(20, 10)   # Dummy test features
intervals = ccqr_est.predict(X_test)

print("Lower bounds:", intervals[:, 0])
print("Upper bounds:", intervals[:, 1])
```

## Project Structure

A possible directory structure for the **ccqr** package might be:

```
ccqr/
├── __init__.py
├── ccqr_predictor.py
└── conformity/
    ├── __init__.py
    ├── coverage.py
    ├── scores.py
    └── utils.py
```

- **`ccqr_predictor.py`**  
  Contains the main `CCQRpredictor` class, which provides the user-facing methods (`fit`, `calibrate`, `predict`).

- **`conformity/`**  
  - **`scores.py`**: Functions for computing conformity scores (symmetric vs. asymmetric).  
  - **`coverage.py`**: Functions for aggregating conformity scores into coverage quantiles based on the chosen strategy.  
  - **`utils.py`**: Helper functions (e.g., generating quantile grids).
