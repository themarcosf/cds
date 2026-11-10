################################################################################
# Author: Marcos Florencio
# Date: 2026-11-10
################################################################################
import logging

import cvxpy as cvx
import numpy as np
import pandas as pd

from collections.abc import Iterator

################################################################################
# Setup
################################################################################
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


################################################################################
# Private Interface
################################################################################
class CombinationProblem:
    def __init__(self, keys: list[str], n: int, window: int):
        self.A = cvx.Parameter((n * window, len(keys)))

        self.weights = cvx.Variable(len(keys), name="weights")

        self.L_p = cvx.Parameter((len(keys), len(keys)))

        self.objective = (
            cvx.sum(cvx.log(self.A @ self.weights)) - 
            0.5 * cvx.sum_squares(self.L_p.T @ self.weights)
        )

        self.contraints = [cvx.sum(self.weights) == 1, self.weights >= 0]

        self.problem = cvx.Problem(
            cvx.Maximize(self.objective),
            self.contraints
        )

    def solve(self):
        return self.problem.solve()


class IEWMA:
    def __init__(
        self, 
        timestamp: pd.Timestamp, 
        mean: pd.Series, 
        covariance: pd.DataFrame, 
        volatility: pd.Series
    ):
        self.timestamp = timestamp
        self.mean = mean
        self.covariance = covariance
        self.volatility = volatility


class Result:
    def __init__(
        self, 
        timestamp: pd.Timestamp, 
        mean: pd.Series, 
        covariance: pd.DataFrame, 
        weights: cvx.Variable
    ):
        self.timestamp = timestamp
        self.mean = mean
        self.covariance = covariance
        self.weights = weights.value


class ConvexCombination:
    """
    Compare multiple return-mean and covariance estimation horizons using 
    precision-transformed residuals.

    Each estimation horizon provides a time series of expected returns and
    covariance matrices. The class transforms both the estimated means and
    observed returns using the Cholesky factor of the inverse covariance 
    matrix, then computes pairwise inner products between the resulting 
    residual vectors.

    The Cholesky factors and transformed means are shifted forward by one row
    before residuals are computed. This is intended to ensure that the estimate
    associated with a timestamp uses information available at the preceding
    timestamp.
    """
    def __init__(
        self, 
        returns: pd.DataFrame, 
        mu: dict[str, dict[pd.Timestamp, pd.Series]],
        sigma: dict[str, dict[pd.Timestamp, pd.DataFrame]]
    ):
        """
        Performs the following operations:

        1. Compute the Cholesky factor of each inverse covariance matrix.
        2. Transform each available estimated mean using its Cholesky factor.
        3. Shift both collections by one row to impose a one-period lag.
        4. Compute transformed residuals and their cross-method Gram matrices.
        """
        self.r = returns
        self.mu = mu
        self.sigma = sigma

        L = self._cholesky_decomposition()
        mu_tilde = self._transform_means(L)

        self.L = L.shift(1).dropna()
        self.mu_tilde = mu_tilde.shift(1).dropna()
        self.Q = self._transform_residuals()

    def _cholesky_decomposition(self) -> pd.DataFrame:
        """
        Computes Cholesky factors of the inverse covariance matrices.

        For each estimation horizon k and timestamp t, calculate the precision
        matrix

            P[k, t] = inverse(Sigma[k, t])

        and its lower-triangular Cholesky factor L[k, t], satisfying

            P[k, t] = L[k, t] @ L[k, t].T
        """
        return pd.DataFrame({
            k: {
                t: np.linalg.cholesky(np.linalg.inv(matrix.values))
                for t, matrix in sigma.items()
            }
            for k, sigma in self.sigma.items()
        })

    def _transform_means(self, cholesky: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms estimated mean-return vectors using precision factors.

        For each estimation horizon k and timestamp t, calculate

            mu_tilde[k, t] = L[k, t].T @ mu[k, t],

        where L[k, t] is the Cholesky factor of the inverse covariance matrix
        and mu[k, t] is the estimated mean-return vector.

        This transformation expresses the mean vector in the precision-induced
        coordinate system used for the residuals.
        """
        return pd.DataFrame({
            k: {
                t: matrix.T @ self.mu[k][t]
                for t, matrix in L.items()
                if 
                    isinstance(k, str) and 
                    isinstance(t, pd.Timestamp) and
                    t in self.mu[k]
            }
            for k, L in cholesky.items()
        })
    
    def _transform_residuals(self) -> pd.Series:
        """
        Computes cross-method Gram matrices of transformed residuals.

        For each estimation horizon k and timestamp t, calculate

            e[k, t] = L[k, t].T @ r[t] - mu_tilde[k, t],

        where r[t] is the observed return vector, mu[k, t] is the estimated
        mean-return vector, and L[k, t] is the Cholesky factor of the inverse 
        covariance matrix.

        Stack the residual vectors for all available horizons into a matrix

            E[t] = column_stack(e[1, t], ..., e[K, t])

        and calculate the Gram matrix

            Q[t] = E[t].T @ E[t].

        Thus, Q[t][i, j] is the inner product between the transformed residual
        vectors of horizons i and j. It is symmetric and positive semidefinite
        and its diagonal contains squared residual norms, while its off-
        diagonal entries measure pairwise residual alignment.
        """
        e = pd.DataFrame({
            k: {
                t: (
                    m.T @ 
                    self.r.loc[t].values - 
                    self.mu_tilde[k][t]
                )
                for t, m in L.items()
                if isinstance(t, pd.Timestamp)
            }
            for k, L in self.L.items()
        })

        E = {
            t: np.column_stack(e.loc[t]) 
            for t in e.index
        }

        return pd.Series({t: e.T @ e for t, e in E.items()})

    def solve(self, window: int) -> Iterator[Result | None]:
        ts = self.Q.index

        P = {
            ts[i]: sum(self.Q.loc[ts[i - window + 1]:ts[i]]) 
            for i in range(window - 1, len(ts))
        }

        L_p = {t: np.linalg.cholesky(m) for t, m in P.items()}

        D = pd.DataFrame({
            k: {
                t: np.diag(m) 
                for t, m in L.items()
            }
            for k, L in self.L.items()
        })
        
        A = {
            ts[i]: np.column_stack([
                np.stack(
                    D.
                    truncate(ts[i - window + 1], ts[i])
                    .iloc[:, j]
                    .to_list()
                ).ravel() 
                for j in range(D.shape[1])
            ])
            for i in range(window - 1, len(ts))
        }

        problem = CombinationProblem(
            keys=list(self.mu.keys()), 
            n=len(self.r.columns), 
            window=window
        )

        for t, m in A.items():
            problem.A.value = m
            problem.L_p.value = L_p[t]

            try:
                yield self._solve(t, problem)
            except cvx.SolverError:
                logger.info(f"Solver did not converge at time {t}.")
                yield None

    def _solve(
        self, 
        t: pd.Timestamp, 
        problem: CombinationProblem
    ) -> Result:
        _ = problem.solve()

        L_pred = sum(self.L.loc[t].to_numpy() * problem.weights.value)
        mu_pred = sum(self.mu_tilde.loc[t].to_numpy() * problem.weights.value)

        assert isinstance(L_pred, np.ndarray)

        mean = pd.Series(
            data=np.linalg.inv(L_pred.T) @ mu_pred,
            index=self.r.columns
        )

        covariance = pd.DataFrame(
            data=np.linalg.inv(L_pred @ L_pred.T),
            index=self.r.columns, 
            columns=self.r.columns
        )

        return Result(
            timestamp=t, 
            mean=mean, 
            covariance=covariance, 
            weights=problem.weights
        )


class CM_IEWMA:
    def __init__(self):
        self.pairs = [(10, 21), (21, 63), (63, 125), (125, 250), (250, 500)]

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        iewmas = {
            f"{pair[0]}-{pair[1]}": {
                iewma.timestamp: iewma
                for iewma in self._iterated_ewma(
                    df,
                    vol_halflife=pair[0],
                    cov_halflife=pair[1]
                )
            }
            for pair in self.pairs
            if len(df) > pair[1]
        }

        sigma, mu = (
            {
                k: {
                    t: getattr(r, attr)
                    for t, r in iewma.items()
                }
                for k, iewma in iewmas.items()
            }
            for attr in ("covariance", "mean")
        )

        cc = ConvexCombination(
            returns=df, 
            mu=mu,
            sigma=sigma
        )

        cov_matrix = {}
        weights = {}
        
        for result in cc.solve(window=10):
            if result is not None:
                cov_matrix[result.timestamp] = result.covariance
                weights[result.timestamp] = result.weights

        return cov_matrix[max(cov_matrix)]
    
    def _compute_ewma_cov(
        self, 
        df: pd.DataFrame, 
        halflife: int
    ) -> Iterator[tuple[pd.Timestamp, pd.DataFrame]]:
        for t, values in self._iter_ewma_values(df, halflife, outer=True):
            yield t, pd.DataFrame(
                values, 
                index=df.columns,
                columns=df.columns
            )

    def _compute_ewma_mean(
        self, 
        df: pd.DataFrame,
        halflife: int
    ) -> pd.DataFrame:
        return pd.DataFrame(
            (values for _, values in self._iter_ewma_values(df, halflife)),
            index=df.index,
            columns=df.columns,
        )

    def _iterated_ewma(
        self, 
        df: pd.DataFrame,
        vol_halflife: int,
        cov_halflife: int
    ) -> Iterator[IEWMA]:
        mean_returns = self._compute_ewma_mean(df=df, halflife=vol_halflife)
        
        demeaned_returns = df - mean_returns

        variance = self._compute_ewma_mean(
            df=demeaned_returns**2, 
            halflife=vol_halflife
        )

        volatility = (variance ** 0.5).fillna(0.0)

        z_scores = (
            (demeaned_returns / volatility)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

        mean_z_scores = self._compute_ewma_mean(
            df=z_scores, 
            halflife=cov_halflife
        )

        demeaned_z_scores = z_scores - mean_z_scores

        for timestamp, centered_z_scores in self._compute_ewma_cov(
            df=demeaned_z_scores,
            halflife=cov_halflife
        ):
            if timestamp < df.index[cov_halflife]:
                continue

            vol = volatility.loc[timestamp]
            assert isinstance(vol, pd.Series)

            vec1 = mean_returns.loc[timestamp]
            vec2 = mean_z_scores.loc[timestamp]
            mean = vec1 + vol * vec2
            assert isinstance(mean, pd.Series)

            v = 1 / np.sqrt(np.diagonal(centered_z_scores))
            corr = centered_z_scores * np.outer(v, v)
            cov = np.outer(vol, vol) * corr
            assert isinstance(cov, pd.DataFrame)

            yield IEWMA(
                timestamp=timestamp,
                mean=mean,
                covariance=cov,
                volatility=vol
            )

    def _iter_ewma_values(
        self,
        df: pd.DataFrame,
        halflife: int,
        outer: bool = False
    ) -> Iterator[tuple[pd.Timestamp, np.ndarray]]:
        decay = np.exp(-np.log(2) / halflife)
        alpha = 1 - decay

        numerator = np.zeros(
            shape = (df.shape[1], df.shape[1]) if outer else df.shape[1], 
            dtype=float
        )

        for t, values in enumerate(df.to_numpy(dtype=float)):
            observation = np.outer(values, values) if outer else values
            numerator = alpha * observation + decay * numerator
            correction = 1 - decay ** (t + 1)

            yield df.index[t], (numerator / correction).copy()