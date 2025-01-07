import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import logging
from chr.methods import CHR
from sklearn.linear_model import LassoCV
from conformal import CPpredictor, CQRpredictor, CCQRpredictor, DCPpredictor
import pickle
import os
import argparse

def f_setting_1(x, random_state):
    """
    Generate synthetic data for setting 1.
    """
    np.random.seed(random_state)
    ax = 0 * x
    for i in range(len(x)):
        ax[i] = (
            np.random.poisson(np.sin(x[i])**2 + 0.1)
            + 0.03 * x[i] * np.random.randn(1)
        )
        ax[i] += 25 * (np.random.uniform(0, 1, 1) < 0.01) * np.random.randn(1)
    return ax.astype(np.float32)

def f_setting_2(x, random_state):
    """
    Generate synthetic data for setting 2.
    """
    np.random.seed(random_state)
    epsilon = np.random.chisquare(df=3, size=x.shape)
    y = (1 - x + 2 * x**2) * np.exp(-0.5 * x**2) + (1 + 0.2 * x) / 5 * epsilon
    outliers = (np.random.uniform(0, 1, size=x.shape) < 0.05)
    y[outliers] += 15 * np.random.randn(np.sum(outliers))
    return y.astype(np.float32)

def f_setting_3(x, random_state):
    """
    Generate synthetic data for setting 3.
    """
    def g(t):
        return 2 * np.sin(np.pi * t) + np.pi * t
    np.random.seed(random_state)
    epsilon = np.random.normal(0, 1, size=(x.shape[0],))
    beta = np.array([1, 1, 1, 1] + [0] * (x.shape[1] - 4))
    beta_X = np.dot(x, beta)
    y = g(beta_X) + epsilon * np.sqrt(1 + beta_X**2)
    return y.astype(np.float32)

def f_setting_4(x, random_state):
    """
    Generates synthetic data for setting 4.
    """
    np.random.seed(random_state)
    
    # Extract first two covariates (X1 and X2) for the nonlinear model
    X1, X2 = x[:, 0], x[:, 1]
    
    # Generate noise terms
    epsilon_1 = np.random.standard_t(3, size=x.shape[0]) * (1 + np.sqrt(X1**2 + X2**2))
    epsilon_2 = np.random.standard_t(3, size=x.shape[0]) * (1 + np.sqrt(X1**4 + X2**4))

    # Conditional response Y generation based on the nonlinear model
    y = np.random.poisson(np.sin(X1)**2 + np.cos(X2)**4 + 0.01) + 0.03 * X1 * epsilon_1 + 25 * (np.random.rand(x.shape[0]) < 0.01) * epsilon_2
    
    return y.astype(np.float32)

SETTING_FUNCTIONS = {
    1: f_setting_1,
    2: f_setting_2,
    3: f_setting_3,
    4: f_setting_4,
}

SETTING_PARAMETERS = {
    1: {'d': 1, 'lam' : 0, 'n_estimators': 500, 'max_features': 1},
    2: {'d': 1, 'lam' : 0, 'n_estimators': 500, 'max_features': 1},
    3: {'d': 100, 'lam' : 0.0001, 'n_estimators': 1000, 'max_features': 10},
    4: {'d': 100, 'lam' : 0.0001, 'n_estimators': 1000, 'max_features': 10}
}

def get_base_model(args, seed):
    """
    Retrieve the base model according to the given model name.

    Args:
        base_model_name (str): Name of the base model (e.g., 'MCQRNN').
        seed (int): Random seed for reproducibility.

    Returns:
        A PyTorch model or an equivalent object that supports `.fit()` and `.predict()`.
    """

    base_model_name = args.base_model
    if base_model_name.upper() == "MCQRNN":
        from mcqrnn.model import MCQRNN_model
        model = MCQRNN_model(m_input_size=1, i_input_size=SETTING_PARAMETERS[args.setting]['d'], hidden_size = 256, quantiles = np.arange(0.05, 1, 0.05), seed=seed, dropout_rate=0.2, early_stopping_round=20)

    elif base_model_name.upper() == "QRF":
        from quantile_forest import RandomForestQuantileRegressor
        model = RandomForestQuantileRegressor(n_estimators=SETTING_PARAMETERS[args.setting]['n_estimators'], min_samples_leaf=100, max_features = SETTING_PARAMETERS[args.setting]['max_features'], random_state=seed)

    elif base_model_name.upper() == "QERT":
        from quantile_forest import ExtraTreesQuantileRegressor
        model = ExtraTreesQuantileRegressor(n_estimators=SETTING_PARAMETERS[args.setting]['n_estimators'], min_samples_leaf=100, max_features = SETTING_PARAMETERS[args.setting]['max_features'], random_state=seed)
    return model


def simulation_pred(model, test_X, test_y):
    """
    Predicts the conformal intervals for the given test set and
    calculates coverage and the mean interval length.

    Args:
        model: The fitted model (with `.predict()` method returning intervals).
        test_X (numpy.ndarray): Test features.
        test_y (numpy.ndarray): True labels for test data.

    Returns:
        tuple: (coverage, average_interval_length)
    """
    prediction_intervals = model.predict(test_X)
    y_lower = prediction_intervals[:, 0]
    y_upper = prediction_intervals[:, 1]
    in_the_range = np.mean((test_y >= y_lower) & (test_y <= y_upper))
    return (in_the_range, np.mean(y_upper - y_lower))


def main(args):
    """
    Main function to run the simulation with different conformal prediction methods.
    It loads data according to the chosen setting, trains a base model, and evaluates
    various conformal prediction methods.

    Args:
        args (argparse.Namespace): Parsed arguments containing `setting`, `base_model`, and 'save'.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    base_model_name = args.base_model

    column_name = [
        'Bandwidth',
        'CHR',
        'DCP',
        'LASSO',
        "CQR",
        "CCQR-SS",
        "CCQR-SSA", #Adaptive
        "CCQR-SAS", #Asymmetric
        "CCQR-SASA" #Asymmetric adaptive
    ]
    coverage_df = pd.DataFrame(columns=column_name)
    interval_df = pd.DataFrame(columns=column_name)

    # Select the function for the chosen setting
    f = SETTING_FUNCTIONS[args.setting]

    # Set constants
    n_train = 1000
    n_test = 5000
    quantiles = np.arange(0.05, 1, 0.05)

    seeds = range(100)

    for seed in seeds:
        # Generate data depending on the chosen setting
        if args.setting == 1:
            train_X = np.random.uniform(0, 5, size=n_train).astype(np.float32).reshape(-1, 1)
            test_X = np.random.uniform(0, 5, size=n_test).astype(np.float32).reshape(-1, 1)

        elif args.setting == 2:
            train_X = np.random.uniform(-4, 4, size=n_train).astype(np.float32).reshape(-1, 1)
            test_X = np.random.uniform(-4, 4, size=n_test).astype(np.float32).reshape(-1, 1)

        elif args.setting == 3:
            train_X = np.random.uniform(0, 1, size=(n_train, 100)).astype(np.float32)
            test_X = np.random.uniform(0, 1, size=(n_test, 100)).astype(np.float32)

        elif args.setting == 4:
            def multivariate_t_rvs(df, Sigma, n):
                d = Sigma.shape[0]
                
                # Step 1: Cholesky decomposition of Sigma
                L = np.linalg.cholesky(Sigma)
                
                # Step 2: Sample chi-square(df) and convert to the scaling factor sqrt(df / chi2)
                chi2_samples = np.random.chisquare(df, size=n)
                scaling = np.sqrt(df / chi2_samples)  # shape = (n,)

                # Step 3: Draw from multivariate Normal(0, Sigma).
                # We first draw Z ~ N(0, I) and then multiply by L to get correlation structure.
                Z = np.random.normal(size=(n, d))    # shape (n, d)
                
                # Step 4: Scale each row of Z by the corresponding scaling, then apply L
                # Equivalent to:  X = (Z * scaling[:, None]) @ L.T
                # shape of X will be (n, d)
                X = (Z * scaling[:, None]) @ L.T
                
                return X
            Sigma = 0.5 * np.ones((100, 100))
            np.fill_diagonal(Sigma, 1.0)  # set diag=1
            train_X = multivariate_t_rvs(3, Sigma, n_train).astype(np.float32)
            test_X  = multivariate_t_rvs(3, Sigma, n_test).astype(np.float32)


        train_y = f(train_X, random_state=seed).flatten()
        test_y = f(test_X, random_state=seed).flatten()

        # Split the train set into train/calibration
        tr_X, cal_X, tr_y, cal_y = train_test_split(
            train_X, train_y, test_size=0.5, random_state=seed
        )

        # Get the base model
        base_model = get_base_model(
            args.base_model, 
            seed=seed
        )
        optimizer = torch.optim.Adam(base_model.model.parameters(), lr=0.01)

        # Fit the base model
        
        if base_model_name.upper() == 'MCQRNN':
            base_model.fit(
                tr_X,
                tr_y,
                optimizer=optimizer,
                batch_size=5000,
                num_epochs=3000,
                verbose=False,
                l1_lambda=0,
            )
        else:
            base_model.fit(tr_X, tr_y)

        # 1) CHR
        grid_quantiles = np.arange(0.01, 1.0, 0.01).tolist()
        chr_model = CHR(base_model, ymin=-3, quantiles=grid_quantiles,ymax=20, y_steps=200, delta_alpha=0.001, randomize=False)
        chr_model.calibrate(cal_X, cal_y, 0.1)
        cr1, len1 = simulation_pred(chr_model, test_X, test_y)

        # 2) DCP
        dcp = DCPpredictor(base_model, quantiles=[0.05, 0.95])
        dcp.calibrate(cal_X, cal_y)
        cr2, len2 = simulation_pred(dcp, test_X, test_y)

        # 3) CP with Lasso
        lasso_reg = LassoCV(cv=5, max_iter=10000)
        lasso_reg.fit(tr_X, tr_y)
        lasso = CPpredictor(lasso_reg, quantiles=[0.05, 0.95])
        lasso.calibrate(cal_X, cal_y)
        cr3, len3 = simulation_pred(lasso, test_X, test_y)

        # 4) CQR
        cqr = CQRpredictor(model=base_model, quantiles=[0.05, 0.95], CV=True)
        cqr.calibrate(cal_X, cal_y)
        cr4, len4 = simulation_pred(cqr, test_X, test_y)

        # 5) CCQR variants
        for bw in range(5, 45, 5):
            ccqr_ss = CCQRpredictor(
                model=base_model,
                quantiles=[0.05, 0.95],
                bandwidth=bw / 100,
                K=9,
                strategy="score",
                symmetric=True,
                adaptive=False,
            )
            ccqr_ssa = CCQRpredictor(
                model=base_model,
                quantiles=[0.05, 0.95],
                bandwidth=bw / 100,
                K=9,
                strategy="score",
                symmetric=True,
                adaptive=True,
            )
            ccqr_sas = CCQRpredictor(
                model=base_model,
                quantiles=[0.05, 0.95],
                bandwidth=bw / 100,
                K=9,
                strategy="score",
                symmetric=False,
                adaptive=False,
            )
            ccqr_sasa = CCQRpredictor(
                model=base_model,
                quantiles=[0.05, 0.95],
                bandwidth=bw / 100,
                K=9,
                strategy="score",
                symmetric=False,
                adaptive=True,
            )

            ccqr_ss.calibrate(cal_X, cal_y)
            ccqr_ssa.calibrate(cal_X, cal_y)
            ccqr_sas.calibrate(cal_X, cal_y)
            ccqr_sasa.calibrate(cal_X, cal_y)

            cr5, len5 = simulation_pred(ccqr_ss, test_X, test_y)
            cr6, len6 = simulation_pred(ccqr_ssa, test_X, test_y)
            cr7, len7 = simulation_pred(ccqr_sas, test_X, test_y)
            cr8, len8 = simulation_pred(ccqr_sasa, test_X, test_y)

            # Record results
            coverage_df = pd.concat(
                [
                    coverage_df,
                    pd.DataFrame(
                        [[bw / 100, cr1, cr2, cr3, cr4, cr5, cr6, cr7, cr8]],
                        columns=column_name,
                    ),
                ],
                axis=0,
            )
            interval_df = pd.concat(
                [
                    interval_df,
                    pd.DataFrame(
                        [[bw / 100, len1, len2, len3, len4, len5, len6, len7, len8]],
                        columns=column_name,
                    ),
                ],
                axis=0,
            )

            logging.info(
                "[Coverage] seed=%d, bw=%.2f, CHR=%.4f, DCP=%.4f, Lasso=%.4f, CQR=%.4f, CCQR=%.4f",
                seed, bw/100, cr1, cr2, cr3, cr4, cr5
            )
            logging.info(
                "[Interval] seed=%d, bw=%.2f, CHR=%.4f, DCP=%.4f, Lasso=%.4f, CQR=%.4f, CCQR=%.4f",
                seed, bw/100, len1, len2, len3, len4, len5
            )

    # Save final results
    with open(f"{args.base_model}_coverage_sim{args.setting}.pkl", "wb") as file:
        pickle.dump(coverage_df, file)
    with open(f"{args.base_model}_interval_sim{args.setting}.pkl", "wb") as file:
        pickle.dump(interval_df, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--setting",
        type=int,
        choices=[1, 2, 3, 4],
        required=True,
        help="The setting number to run (1, 2, 3, or 4).",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="MCQRNN",
        help="The name of the base model to use (e.g., 'MCQRNN').",
    )
    args = parser.parse_args()

    main(args)
