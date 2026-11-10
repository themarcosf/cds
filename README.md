# Causal Domain Shear

Reference implementation of the **Causal Domain Shear (CDS)** algorithm, a
covariance-matrix correction for portfolio risk estimation, together with a
rolling backtest that compares it against established covariance estimators.

> The paper describing CDS is forthcoming. This repository will be linked from
> it once published.

---

## Motivation

Portfolio theory usually assumes that all assets live in one shared temporal
frame, so that their co-movement is fully described by a covariance matrix.
CDS starts from a different intuition:

- A **causal domain** is a set of things linked by mutual cause-and-effect.
  Assets can be connected without being synchronized in time, so their
  evolution is partly independent.
- **Covariance** comes from *strong* temporal coupling. **Causal domains** come
  from *weak* temporal coupling. If domains are not synchronized, shared
  covariances may matter less for diversification than people assume.

Unless the data shows otherwise, CDS treats each asset as its **own causal
domain**, connected to the others only by a small *slippage*. That slippage is
estimated with a **volume-preserving shear** of a base covariance matrix.

## The algorithm

Implemented in [`src/lib/risk/cds.py`](src/lib/risk/cds.py) as
`CausalDomainShear(sigma, X_test).optimize()`.

**Model.** Assets are put in a causal order. Asset *i* is driven by its own
innovation (with its own variance) plus slippage from the domains that come
after it in that order. This is the structure of a unit-upper-triangular
factorisation

$$\Sigma = U \,\mathrm{diag}(d)\, U^\top, \qquad U \text{ unit upper triangular},$$

where `d` are the per-domain innovation variances and `U` holds the slippage
between domains.

**Fit.**

1. Factor the base covariance `A` as `U_A diag(d_A) U_Aᵀ`.
2. Mix in the evidence. `C = XᵀX / n` is the uncentered second moment of the
   `n` recent returns `X_test`. The mixing weight is
   `w = n / (n + n_prior)`, so `M = (1 − w)·A + w·C`.
3. Take the shear `U_M` from the factorisation of `M`, but keep the base
   innovation variances: `Σ_CDS = U_M diag(d_A) U_Mᵀ`.

Because `det U = 1`, the result has the same determinant ("volume") as `A`.
Only the coupling between domains changes, never the total risk volume. With
volume fixed, the Gaussian log-likelihood depends on the shear only through a
trace term, so the minimizer has a closed form: no iterations and no solver.
With no evidence (`w = 0`), the base matrix comes back unchanged.

**Parameters.**

| Argument  | Default      | Meaning |
|-----------|--------------|---------|
| `sigma`   | —            | Base covariance (a `DataFrame` indexed by asset). |
| `X_test`  | —            | Recent returns used as evidence. Never the evaluation window. |
| `n_prior` | `10 * N`     | How many observations the base matrix is worth. Sets the evidence weight `n / (n + n_prior)`. |
| `order`   | `"variance"` | Causal order. `"variance"`: low-variance domains drive high-variance ones, and the result does not depend on column order. `"given"`: column order, with later columns driving earlier ones. A list of labels: explicit order, from most driven to most driving. |

`CausalDomainShear.causal_order` returns the order that was used.

## Pipeline

```
returns ──► RiskEngine.estimate ──► {sample, ewma, shrunk, ledoit_wolf, oas, cm_iewma}
                                              │
                                              ▼
                                  RiskEngine.ensemble  (likelihood-weighted mix)
                                              │
                                              ▼
                                  CausalDomainShear    (shear towards evidence)
```

### Base estimators: [`src/lib/risk/estimators.py`](src/lib/risk/estimators.py)

Each estimator is a callable `DataFrame of returns → covariance DataFrame`, and
each output is symmetrised and checked for NaNs.

| Name          | Description |
|---------------|-------------|
| `Sample`      | Plain sample covariance. |
| `EWMA`        | Exponentially weighted covariance. The decay factor is picked from 0.80–0.98 by out-of-sample Gaussian log-likelihood, with RiskMetrics' 0.94 as fallback. |
| `Shrunk`      | Shrinkage towards a scaled identity, with the intensity cross-validated over a log grid. |
| `LedoitWolf`  | Ledoit–Wolf optimal shrinkage. |
| `OAS`         | Oracle Approximating Shrinkage. |
| `CM_IEWMA`    | Combined multiple iterated EWMA ([`src/lib/models/cm_iewma.py`](src/lib/models/cm_iewma.py)): several half-life pairs, mixed by a convex optimisation (cvxpy) over precision-transformed residuals. |

### Risk engine: [`src/lib/risk/engine.py`](src/lib/risk/engine.py)

`RiskEngine(X_train, X_test)` holds one training/test split. It requires at
least `10 · N` training rows for `N` assets, following Ledoit and Wolf (2001):
*"small sample problems occur unless T is at least one order of magnitude
larger than N"*.

- `estimate(estimator)` fits a base estimator on the training window.
- `ensemble(sigma_hat, weights)` finds simplex weights `w` that maximise the
  Gaussian likelihood of the test window under `A(w) = Σ_k w_k Σ_k`:

  $$\min_{w \in \Delta} \; \log\det A(w) + \mathrm{tr}\!\left(A(w)^{-1} C\right)$$

  This is solved with SLSQP using analytic gradients. The target `C` shrinks
  towards the average estimate in proportion to how much test data there is,
  so with no evidence the weights stay equal. Across backtest steps the
  weights are smoothed (`λ = 0.1`) so they don't jump around.

The ensemble matrix is then the base `sigma` passed to `CausalDomainShear`.

### Backtest: [`src/lib/risk/backtest.py`](src/lib/risk/backtest.py)

`backtest_risk_engine(tickers, estimators, period)` downloads daily close
returns from Yahoo Finance and walks forward one day at a time over three
adjacent windows:

| Window | Length | Used for |
|--------|--------|----------|
| train  | `10·N` | fitting the base estimators |
| test   | `N`    | ensemble weights, CDS evidence, and scoring in the first part of the sample |
| eval   | `N`    | held-out scoring in the last part of the sample (after the cutoff at ≈ 2/3 of the data) |

Every step scores the six estimators plus `ensemble` and `cds` with:

| Metric                   | Scores | Notes |
|--------------------------|--------|-------|
| `logdet`                 | `log det Σ` | Risk "volume". |
| `mahal`, `nll`           | the window **mean** vector | Minimised by `Σ / n`, so it favours small determinants. Only compare estimators with equal `logdet` on it. |
| `mahal_obs`, `nll_obs`   | each **observation** in the window | A proper scoring rule, minimised in expectation by the true covariance. `mahal_obs ≈ N` when `Σ` is calibrated. |

It returns two DataFrames: test-window results and eval-window results.

---

## Getting started

### Environment

The project ships as a VS Code **Dev Container** (Python 3.13, see
[`.devcontainer.json`](.devcontainer.json) and
[`docker/Dockerfile`](docker/Dockerfile)). Open the folder in VS Code and pick
**Reopen in Container**. Dependencies (`numpy`, `pandas`, `scipy`,
`scikit-learn`, `cvxpy`, `yfinance`, `pyarrow`, `pytest`) are compiled from
[`src/requirements.in`](src/requirements.in) and installed automatically.

### Usage: `run.sh`

Everything runs through `run.sh`. The image doesn't support shebang execution,
so call it through `bash`:

```bash
bash run.sh --help            # show usage
bash run.sh --test live       # backtest on fresh market data from Yahoo Finance
bash run.sh --test offline    # backtest on the bundled snapshot, no network
```

| Command                  | What it does |
|--------------------------|--------------|
| `--help`                 | Prints usage and exits. |
| `--test live`            | Runs `tests/risk-backtest.py` on one year of daily returns downloaded from Yahoo Finance. Results change from day to day as the window moves. |
| `--test offline`         | Runs the same test on `tests/sample.parquet`, a saved copy of `get_daily_returns` output. Results are reproducible and no network is needed. |

Any other argument, or a missing `live`/`offline` value, prints the help and
exits with status 1.

Both modes run the integration test with live logging. For each result
set (test window and eval window), it prints a table of the mean and standard
deviation of every metric per method, for this universe:

```
AAPL  GOOGL  MSFT  NVDA  TSLA  AMZN  BRK-B  JPM  WMT  XOM
```

### Refreshing the offline snapshot

`tests/sample.parquet` must keep the `DatetimeIndex` returned by
`get_daily_returns`, because `CM_IEWMA` relies on timestamps. To regenerate it:

```python
from lib.risk.backtest import get_daily_returns

tickers = ["AAPL", "GOOGL", "MSFT", "NVDA", "TSLA",
           "AMZN", "BRK-B", "JPM", "WMT", "XOM"]

get_daily_returns(tickers, "1y").to_parquet("tests/sample.parquet")
```

---

## Citation

If you use this code, please cite the Causal Domain Shear paper (forthcoming).
Citation details will be added on publication.
