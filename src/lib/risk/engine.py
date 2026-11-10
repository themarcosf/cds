################################################################################
# Author: Marcos Florencio
# Date: 2026-07-14
################################################################################
import logging

import numpy as np
import pandas as pd

from collections.abc import Callable

from scipy.optimize import minimize

################################################################################
# Setup
################################################################################
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


################################################################################
# Public Interface
################################################################################
class RiskEngine:
    def __init__(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        k: int = 10
    ):
        """
        -- "For the covariance matrix, small sample problems occur unless T is
            at least one order of magnitude larger than N."
            (Ledoit and Wolf, 2001)
        """
        T, N = X_train.shape

        if T < k * N:
            raise ValueError

        self.T = T
        self.N = N
        self.X_train = X_train
        self.X_test = X_test
        self.columns = X_train.columns

    def estimate(
        self,
        estimator: Callable[[pd.DataFrame], pd.DataFrame]
    ) -> pd.DataFrame:
        return estimator(self.X_train.copy())
    
    def ensemble(
        self, 
        sigma_hat: dict[str, pd.DataFrame],
        weights: dict[str, float],
        lambda_: float = 0.1
    ) -> tuple[dict[str, float], pd.DataFrame]:
        weights_ = self._fit_weights(sigma_hat)

        if not (weights or weights_):
            raise RuntimeError

        if not weights:
            assert weights_ is not None
            weights = weights_.copy()
        elif weights_:
            weights = {
                k: (1 - lambda_) * weights[k] + lambda_ * weights_[k]
                for k in weights_
            }

        A = pd.DataFrame(
            0.0,
            index=self.columns,
            columns=self.columns
        )

        for w, s in zip(weights.values(), sigma_hat.values()):
            A += w * s

        return weights, (A + A.T) / 2
        
    def _fit_weights(
        self,
        sigma_hat: dict[str, pd.DataFrame]
    ) -> dict[str, float] | None:
        """
        Weights w on the simplex that maximize the Gaussian likelihood of the
        test observations under A(w) = sum_k w_k Sigma_k:

            minimize logdet A(w) + tr(A(w)^-1 C)

        Only A(w)^-1 is needed, which exists for any convex combination of PD
        estimates.

        With no evidence the minimizer is equal weights; the weights are moved
        away from that in proportion to their share of the data.
        """
        keys = list(sigma_hat)
        K = len(keys)

        M = np.stack([
            sigma_hat[key]
            .loc[self.columns, self.columns]
            .to_numpy(dtype=float)
            for key in keys
        ])

        X = self.X_test[self.columns].to_numpy(dtype=float)
        n = len(X)

        scale = np.trace(M.mean(axis=0)) / self.N
        M = M / scale

        k = n / (n + self.T)
        C = (1 - k) * M.mean(axis=0) + k * (X.T @ X) / (n * scale)

        def objective(w: np.ndarray) -> tuple[float, np.ndarray]:
            A = np.tensordot(w, M, axes=1)
    
            sign, logdet = np.linalg.slogdet(A)
    
            if sign <= 0:
                return np.inf, np.zeros(K)
    
            A_inv = np.linalg.inv(A)
            G = A_inv - A_inv @ C @ A_inv
    
            return (
                logdet + np.trace(A_inv @ C),
                np.einsum("ij,kji->k", G, M)
            )

        try:
            result = minimize(
                objective,
                x0=np.full(K, 1.0 / K),
                jac=True,
                method="SLSQP",
                bounds=[(0.0, 1.0)] * K,
                constraints=[{
                    "type": "eq",
                    "fun": lambda w: w.sum() - 1.0,
                    "jac": lambda w: np.ones(K)
                }],
                options={
                    "ftol": 1e-10,
                    "maxiter": 200
                }
            )
        except np.linalg.LinAlgError as e:
            logger.info(f"Weight optimization failed: {e}")
            return None

        if not result.success or not np.isfinite(result.x).all():
            logger.info(
                "Weight optimization did not converse. "
                f"Solver message: {result.message}"
            )
            return None

        w = np.clip(result.x, 0.0, None)
        w = w / w.sum()

        return {k: float(v) for k, v in zip(keys, w)}