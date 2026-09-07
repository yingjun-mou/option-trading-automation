# aal-wheel-decision-making

Research code that searches AAL's price and options history for the most
profitable **price-based** options-selling strategy -- cash-secured puts, covered
calls, or the two combined as a Wheel -- and checks whether the result survives
out of sample.

It is a decision-making study, not a trading bot. Nothing here connects to a
broker or runs live. You run it once and read the report.

## The idea

AAL mean-reverts inside a fairly persistent post-COVID price band. So deploy
capital based on where AAL sits in its own recent range, not on the calendar.
Sell puts when it is cheap, size up as it gets cheaper, take assignment, sell
calls only once it has recovered, let it be called away near the top, then wait
in cash for the next cheap window. Cash is allowed to sit idle for years.

## Install

```
pip install -r requirements.txt
```

Python 3.11+ (developed on 3.14).

## Run

```
python scripts/run_all.py            # full grid + walk-forward + report (~8 min)
python scripts/run_all.py --quick    # small grid, fast sanity pass
python scripts/ml_probe.py           # optional: does a simple model beat the price rule?
```

Everything lands in `results/`: `RESEARCH_REPORT.md` (the readable answer),
grid CSVs, walk-forward and regime tables, per-trade log, equity plots.

## Data

Stock prices are real (Yahoo Finance, cached in `data/stock_aal.csv`).

Option prices come from the **ORATS EOD "strikes" schema**. Two folders, one
reader:

- `data/mock/` -- a synthetic chain written on first run in the exact ORATS
  layout (Black-Scholes on the real AAL close plus a modelled IV surface). Used
  only while `data/orats/` is empty.
- `data/orats/` -- drop real ORATS EOD files here (`.csv` or `.parquet`, one big
  file or one per day). As soon as anything is present it takes over and the
  report header changes from `MockOptions` to `OratsFolder`.

Both go through `read_orats_strikes()`, so switching is zero-code. See
`data/README.md` for the column list.

## Strategy legs

Each side of the Wheel is an independent **leg** with its own price-ladder logic:

- `src/csp/` -- cash-secured puts: signal -> target capital allocation, contract
  selection, the "wait for the next dip after a call-away" gate.
- `src/cc/` -- covered calls: signal -> fraction of shares to cover, contract
  selection, the call-away zone, the don't-sell-below-basis rule.

`WheelConfig` carries both legs' parameters plus `enable_csp` / `enable_cc`. The
engine runs any list of legs, so:

| study | config |
| --- | --- |
| full Wheel | `enable_csp=True, enable_cc=True` (default) |
| CSP only | `enable_cc=False` |
| CC overlay | `enable_csp=False, initial_alloc=1.0, cc_rebuy_when_flat=True` |

The report's benchmark table shows all three.

## How the code is organised

The pipeline is a chain of independent steps; each module hands a plain DataFrame
or dataclass to the next.

| module | responsibility |
| --- | --- |
| `datasource.py` | The only place that knows where data comes from. `StockSource` / `OptionSource` interfaces; `YahooStock`, `CsvStock`, `OratsFolder`, `MockOptions` implement them. `load_market_data(stock_source, option_source)` returns a `MarketData`. New provider = one new class. |
| `pricing.py` | Black-Scholes price, greeks, implied-vol solve. |
| `features.py` | Daily features from `MarketData`: rolling 3Y/1Y price percentile (main signal), realised vol, IV percentile, momentum. |
| `csp/`, `cc/` | The two legs. `leg.py` in each holds the ladder function and the `rebalance()` that decides what to sell that day. |
| `wheel.py` | `WheelConfig`, `build_legs(cfg)`, and `run_backtest` = `run_strategy(build_legs(cfg), ...)`. |
| `engine.py` | Instrument-level machinery, no strategy knowledge. `Chain` (fast per-day lookup), `WheelAccount` (cash / shares / positions / assignment / call-away accounting), `run_strategy(legs, ...)` (the daily loop). |
| `metrics.py` | CAGR, drawdown, Sharpe / Sortino / Calmar, yearly returns. |
| `benchmarks.py` | Buy-and-hold curve; mechanical-wheel and price-wheel config factories. |
| `evaluation.py` | `Objective` = "give me a config and a window, I return its metrics". Walk-forward windows (`TRAIN` / `VALID` / `TEST`) and regime windows. The reusable primitive every search method uses. |
| `search.py` | Search decoupled from the objective. `Searcher` has `propose` / `observe` / `finished`; `GridSearcher` is today's only one. A regression- or XGBoost-guided searcher is a new class here that reads the results table in `observe` and returns configs in `propose`. Also `walk_forward_select` (TRAIN shortlist, VALID chooses, TEST untouched). |
| `spaces.py` | The parameter grids and ladder presets. |
| `report.py` | Renders the context dict into `RESEARCH_REPORT.md` and draws the plots. |
| `scripts/run_all.py` | Wires the steps together. |
| `scripts/ml_probe.py` | Optional phase-2 model check. |

## Swapping pieces

- **New data source**: implement `fetch()` on a class in `datasource.py`.
- **Smarter search**: implement `Searcher` in `search.py`, feed it the same
  `Objective`.
- **New strategy leg**: add a package under `src/` with a `rebalance(account,
  chain, date, spot, features_row)` method and list it in `build_legs`.

## Method notes

- Grid search runs on TRAIN (2016-2021) only. VALID (2022-2023) chooses among the
  TRAIN shortlist. TEST (2024-latest) is scored but never optimised on.
- Performance is also split by regime (pre-COVID / COVID / post-COVID).
- Absolute-dollar entry ladders score higher in-sample but embed hindsight about
  AAL's post-COVID range. The report's "defensible" percentile-only variant
  self-calibrates; that is the one to deploy.
- Cash-secured puts only, no margin, no naked options, no rolling, European-style
  assignment, no liquidity cap on position size.
