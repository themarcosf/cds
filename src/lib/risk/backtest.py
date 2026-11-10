################################################################################
# Author: Marcos Florencio
# Date: 2026-09-17
################################################################################
import logging
import math
import time

import numpy as np
import pandas as pd
import yfinance as yf

from collections.abc import Callable

from lib.risk.cds import CausalDomainShear
from lib.risk.engine import RiskEngine


################################################################################
# Setup
################################################################################
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


################################################################################
# Private Interface
################################################################################
def evaluate(sigma: pd.DataFrame, X_target: pd.DataFrame) -> dict:
    """
    Two Gaussian scores of sigma on the target window, both with zero mean.

    Window mean (logdet, mahal, nll): scores the single vector X_target.mean().
    The mean of n rows has covariance sigma / n, so this score is minimized in
    expectation by sigma / n, not by sigma, and it favours matrices with small
    determinant. Only compare estimators with equal logdet on it.

    Per observation (mahal_obs, nll_obs): the average negative log-likelihood
    of the n rows. A proper score, minimized in expectation by the true 
    covariance; mahal_obs is close to N when sigma is calibrated.
    """
    X = X_target[sigma.columns].to_numpy(dtype=float)
    mu = X.mean(axis=0)

    N = len(sigma.columns)

    _, logdet = np.linalg.slogdet(sigma)

    mahal = mu @ np.linalg.solve(sigma, mu)
    mahal_obs = np.einsum("ti,it->t", X, np.linalg.solve(sigma, X.T)).mean()

    const = N * math.log(2 * math.pi)

    return dict(
        logdet=logdet,
        mahal=mahal,
        nll=0.5 * (const + logdet + mahal),
        mahal_obs=mahal_obs,
        nll_obs=0.5 * (const + logdet + mahal_obs)
    )


def get_daily_returns(
    tickers: list[str], 
    period: str, 
    max_attempts: int = 5
) -> pd.DataFrame:
    for attempt in range(max_attempts):
        df = yf.download(
            tickers=tickers,
            period=period,
            interval="1d",
            keepna=True
        )

        if df is not None and not df.empty:
            df = df[["Close"]].pct_change().iloc[1:]

            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel(0, axis=1)

            if not df.isna().any().any():
                return df

        if attempt < max_attempts - 1:
            time.sleep(2**(attempt + 1))

    raise RuntimeError


################################################################################
# Public Interface
################################################################################
def backtest_risk_engine(
    tickers: list[str],
    estimators: dict[str, Callable[[pd.DataFrame], pd.DataFrame]],
    period: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = get_daily_returns(tickers, period)

    T, N = df.shape

    cutoff = min(round(T / 3 * 2), T - 50)

    i = 0

    weights = {}

    test_results = []
    eval_results = []

    while True:
        logger.info(f"Iteration: {i}")

        i_train = i + 10 * N
        X_train = df.iloc[i:i_train]

        i_test = i_train + N
        X_test = df.iloc[i_train:i_test]

        i_eval = i_test + N
        X_eval = df.iloc[i_test:i_eval]

        engine = RiskEngine(X_train, X_test)

        sigma_hat = {}

        # parallelize
        for method, estimator in estimators.items():
            sigma_hat[method] = engine.estimate(estimator)

        weights, ensemble = engine.ensemble(sigma_hat, weights)
        sigma_hat["ensemble"] = ensemble

        sigma_hat["cds"] = CausalDomainShear(ensemble, X_test).optimize()

        for method, sigma in sigma_hat.items():
            if i_train < cutoff:
                evaluation = evaluate(sigma, X_test)
                test_results.append(dict(
                    train_start=X_train.index[0],
                    train_end=X_train.index[-1],
                    test_start=X_test.index[0],
                    test_end=X_test.index[-1],
                    method=method,
                    logdet=evaluation["logdet"],
                    mahal=evaluation["mahal"],
                    nll=evaluation["nll"],
                    mahal_obs=evaluation["mahal_obs"],
                    nll_obs=evaluation["nll_obs"]
                ))
            else:
                evaluation = evaluate(sigma, X_eval)
                eval_results.append(dict(
                    train_start=X_train.index[0],
                    train_end=X_train.index[-1],
                    test_start=X_test.index[0],
                    test_end=X_test.index[-1],
                    eval_start=X_eval.index[0],
                    eval_end=X_eval.index[-1],
                    method=method,
                    logdet=evaluation["logdet"],
                    mahal=evaluation["mahal"],
                    nll=evaluation["nll"],
                    mahal_obs=evaluation["mahal_obs"],
                    nll_obs=evaluation["nll_obs"]
                ))

        if i_eval == T:
            break

        i += 1

    return pd.DataFrame(test_results), pd.DataFrame(eval_results)