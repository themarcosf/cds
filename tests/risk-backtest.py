################################################################################
# Author: Marcos Florencio
# Date: 2026-09-12
################################################################################
import logging
import os
import sys

from pathlib import Path

import pandas as pd
import pytest

from lib.risk import backtest
from lib.risk import estimators as est
from lib.risk.backtest import backtest_risk_engine


################################################################################
# Setup
################################################################################
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout
)

logger = logging.getLogger(__name__)

tickers = [
    "NVDA",     # AI causal domain candidate - GPU accelerators
    "AMD",      # AI causal domain candidate - data center processors
    "ORCL",     # AI causal domain candidate - data center infrastructure
    "VRT",      # AI causal domain candidate - data center cooling and power management
    "KO",       # consumer staples
    "XOM",      # energy
    "CB",       # financials and insurance 
    "JNJ",      # healthcare and pharmaceuticals
    "UNP",      # industrials (railroad)
    "NEM"       # materials (gold mining)
]

estimators = dict(
    sample=est.Sample(),
    ewma=est.EWMA(),
    shrunk=est.Shrunk(),
    ledoit_wolf=est.LedoitWolf(),
    oas=est.OAS(),
    cm_iewma=est.CM_IEWMA()
)

DATA_SOURCE = os.environ.get("DATA_SOURCE", "live")
SAMPLE_PATH = Path(__file__).parent / "sample.parquet"


################################################################################
# Tests
################################################################################
@pytest.fixture
def daily_returns(monkeypatch):
    if DATA_SOURCE == "offline":
        sample = pd.read_parquet(SAMPLE_PATH)
        assert isinstance(sample.index, pd.DatetimeIndex)
        monkeypatch.setattr(
            backtest,
            "get_daily_returns",
            lambda *args, **kwargs: sample
        )
    elif DATA_SOURCE != "live":
        raise ValueError


@pytest.mark.integration
def test_risk_backtest(daily_returns):
    results = backtest_risk_engine(
        tickers=tickers, 
        estimators=estimators, 
        period="1y"
    )

    for df in results:
        try:
            stats = (
                df.groupby(["method"])
                .agg(
                    logdet_mean=("logdet", "mean"),
                    logdet_std=("logdet", "std"),
                    mahal_mean=("mahal", "mean"),
                    mahal_std=("mahal", "std"),
                    nll_mean=("nll", "mean"),
                    nll_std=("nll", "std"),
                    mahal_obs_mean=("mahal_obs", "mean"),
                    mahal_obs_std=("mahal_obs", "std"),
                    nll_obs_mean=("nll_obs", "mean"),
                    nll_obs_std=("nll_obs", "std")
                )
            )

            logger.info("\n%s", stats.to_string())
        except:
            continue