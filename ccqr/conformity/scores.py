# ccqr/conformity/scores.py

import numpy as np

# If you have your adaptive vs. non-adaptive scoring in separate functions:
# from .my_adaptive_module import ...
# or define them inline here.

def asymmetric_adaptive_conformity_score(y_true, y_pred_q, y_pred_target):
    # Your logic here
    pass

def asymmetric_conformity_score(y_true, y_pred_q):
    # Your logic here
    pass

def symmetric_adaptive_conformity_score(y_true, y_pred_q, y_pred_target):
    # Your logic here
    pass

def symmetric_conformity_score(y_true, y_pred_q):
    # Your logic here
    pass


def calculate_asymmetric_scores(
    model, X_calib, y_calib, grid_q, target_quantiles, adaptive
):
    lower_scores, upper_scores = [], []
    for q in grid_q:
        y_pred_quantiles = model.predict(X_calib, quantiles=[q[0], q[1]])
        if adaptive:
            y_pred_target = model.predict(X_calib, quantiles=target_quantiles)
            lower_score, upper_score = asymmetric_adaptive_conformity_score(
                y_true=y_calib,
                y_pred_q=y_pred_quantiles,
                y_pred_target=y_pred_target
            )
        else:
            lower_score, upper_score = asymmetric_conformity_score(
                y_true=y_calib,
                y_pred_q=y_pred_quantiles
            )
        lower_scores.append(lower_score)
        upper_scores.append(upper_score)
    return lower_scores, upper_scores


def calculate_symmetric_scores(
    model, X_calib, y_calib, grid_q, target_quantiles, adaptive
):
    calibration_scores = []
    for q in grid_q:
        y_pred_quantiles = model.predict(X_calib, quantiles=[q[0], q[1]])
        if adaptive:
            y_pred_target = model.predict(X_calib, quantiles=target_quantiles)
            score = symmetric_adaptive_conformity_score(
                y_true=y_calib,
                y_pred_q=y_pred_quantiles,
                y_pred_target=y_pred_target
            )
        else:
            score = symmetric_conformity_score(
                y_true=y_calib,
                y_pred_q=y_pred_quantiles
            )
        calibration_scores.append(score)
    return calibration_scores
