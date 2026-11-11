################################################################################
# Author: Marcos Florencio
# Date: 2026-11-10
################################################################################
import numpy as np
import pandas as pd


################################################################################
# Private Interface
################################################################################
def _unit_upper_ldl(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A = U diag(d) U^T with U unit upper triangular."""
    L = np.linalg.cholesky(A[::-1, ::-1])
    G = L[::-1, ::-1]
    g = np.diag(G)
    return G / g, g ** 2


def _shear_fit(
    A: np.ndarray,
    C: np.ndarray,
    weight: float
) -> np.ndarray:
    """Likelihood-optimal unit-triangular shear of A towards C."""
    if not 0.0 <= weight < 1.0:
        raise ValueError

    if weight == 0.0:
        return A.copy()

    _, d_A = _unit_upper_ldl(A)
    U_M, _ = _unit_upper_ldl((1.0 - weight) * A + weight * C)

    sheared = (U_M * d_A) @ U_M.T

    return (sheared + sheared.T) / 2


################################################################################
# Public Interface
################################################################################
class CausalDomainShear:
    """
    A causal domain is a collection of things linked by mutual cause-and-effect
    relationships. One cannot really draw hard and fast boundaries between 
    causal domains, as it depends on what level of approximation one is willing
    to put up with, but it is nevertheless useful to distinguish them when
    interactions between them are non-negligible at the scale of interest.
    
    In portfolio theory, the ordinary intuition is that assets share a common 
    temporal framework. Our intuition is that assets can be connected without 
    being synchronized in time, making their temporal evolution partially 
    independent.
    
    In other words, covariance arises from strong temporal coupling, while 
    causal domains arise from weak temporal coupling. If different causal 
    domains are not synchronized, the notion of shared covariances may be less 
    meaningful than assumed for portfolio diversification.

    In the absence of evidence to the contrary, all assets are assumed to 
    belong to independent causal domains, except for a minimal slippage 
    coefficient. This coefficient is then optimized with respect to a volume-
    preserving shear transformation of a covariance matrix.

    Model. Each asset is its own causal domain with its own innovation 
    variance. In a causal ordering, asset i is driven by its own innovation 
    plus slippage from the domains that come after it.

    Fit. With the volume fixed, the Gaussian log-likelihood of n observations
    with second moment C depends on the shear only through tr(...). The 
    minimizer is closed-form: with no evidence of causal interference the base
    matrix is returned unchanged.
    """
    def __init__(
        self, 
        sigma: pd.DataFrame,
        X_test: pd.DataFrame,
        *,
        n_prior: int | None = None,
        order: str | list = "variance"
    ):
        """
        sigma   Base covariance.
        X_test  Recent returns used as evidence. Never the evaluation window.
        n_prior Number of observations the base matrix is worth. Defaults to
                10 * N, the training length RiskEngine requires. The evidence
                weight is n / (n + n_prior) with n = len(X_test).
        order   Causal order.
                "variance": low-variance domains drive high-variance ones; the
                            result does not depend on the column order.
                "given":    column order, later columns drive earlier ones.
                list:       explicit labels, from most driven to most driving.
        """
        self.index = sigma.index
        self.columns = sigma.columns
        self.n = sigma.shape[0]

        A = sigma.to_numpy(dtype=float)
        self.A = (A + A.T) / 2

        X = X_test[list(self.columns)].to_numpy(dtype=float)

        # uncentered second moment: with n ~ N the sample mean is noise
        self.C = X.T @ X / len(X)

        n_prior = 10 * self.n if n_prior is None else n_prior

        if n_prior <= 0:
            raise ValueError

        self.weight = len(X) / (len(X) + n_prior)

        if isinstance(order, str):
            if order == "variance":
                self.perm = np.argsort(-np.diag(self.A), kind="stable")
            elif order == "given":
                self.perm = np.arange(self.n)
            else:
                raise ValueError
        else:
            labels = list(self.columns)
            self.perm = np.array([labels.index(c) for c in order])
            if sorted(self.perm) != list(range(self.n)):
                raise ValueError

    @property
    def causal_order(self) -> list:
        return list(self.columns[self.perm])

    def optimize(self) -> pd.DataFrame:
        p = self.perm
        q = np.argsort(p)

        sheared = _shear_fit(
            self.A[np.ix_(p, p)],
            self.C[np.ix_(p, p)],
            self.weight
        )

        return pd.DataFrame(
            sheared[np.ix_(q, q)],
            index=self.index,
            columns=self.columns
        )