################################################################################
# Author: Marcos Florencio
# Date: 2026-08-09
################################################################################
import logging

import numpy as np
import pandas as pd

from sklearn import covariance as skcv
from sklearn.model_selection import GridSearchCV

from lib.models.cm_iewma import CM_IEWMA as CM_IEWMA_

################################################################################
# Setup
################################################################################
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


################################################################################
# Private Interface
################################################################################
class BaseCovariance:
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.isna().any().any():
            raise ValueError

        return (df + df.T) / 2


################################################################################
# Public Interface
################################################################################
class Sample(BaseCovariance):
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.validate(df.cov())


class EWMA(BaseCovariance):
    """
    -- "When we think about the variances and covariances as unobserved time-
        varying quantities, it becomes clear that more recent observations of 
        returns contain more information about current values than do older 
        observations"
        (Black and Litterman, Asset Allocation, 1991)
    """

    def __init__(self):
        self.decay_factors = (
            0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.98,
        )

        self.risk_metrics_choice = 0.94

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        decay_factor = self._select_decay_factor(df)

        return self.validate((
            df
            .ewm(alpha=1 - decay_factor)
            .cov(pairwise=True)
            .loc[df.index[-1]]
        ))

    def _select_decay_factor(self, df: pd.DataFrame) -> float:
        scores = {
            decay_factor: self._score(df, decay_factor)
            for decay_factor in self.decay_factors
        }

        return max(
            scores.items(), 
            key=lambda item: item[1]
        )[0]

    def _score(self, df: pd.DataFrame, decay_factor: float) -> float:
        cut = int(min(len(df) * 0.9, len(df.columns) * 1e1))

        scores = []

        for i in range(cut, len(df)):
            context = df.iloc[:i]
            observation = df.iloc[i]

            cov_matrix = (
                context
                .ewm(alpha=1 - decay_factor)
                .cov(pairwise=True)
                .loc[context.index[-1]]
            )

            sign, logdet = np.linalg.slogdet(cov_matrix)

            if sign <= 0:
                continue

            mahal = observation @ np.linalg.solve(cov_matrix, observation)

            n = len(observation)

            scores.append(
                -0.5 * (
                    n * np.log(2 * np.pi)
                    + logdet
                    + mahal
                )
            )

        if not scores:
            return self.risk_metrics_choice

        return float(np.mean(scores))


class Shrunk(BaseCovariance):
    def __init__(self):
        self.param_grid = [{"shrinkage": np.logspace(-2, 0, 30)}]

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        cv = GridSearchCV(
            estimator=skcv.ShrunkCovariance(), 
            param_grid=self.param_grid
        )

        cv.fit(df)

        return self.validate(pd.DataFrame(
            cv.best_estimator_.covariance_,
            index=df.columns,
            columns=df.columns
        ))


class LedoitWolf(BaseCovariance):
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        cov = skcv.LedoitWolf().fit(df)

        return self.validate(pd.DataFrame(
            cov.covariance_, 
            index=df.columns, 
            columns=df.columns
        ))


class OAS(BaseCovariance):
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        cov = skcv.OAS().fit(df)

        return self.validate(pd.DataFrame(
            cov.covariance_, 
            index=df.columns, 
            columns=df.columns
        ))


class CM_IEWMA(BaseCovariance):
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.validate(CM_IEWMA_()(df))