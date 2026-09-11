# aal-wheel-decision-making

Two parts:

1. **Research** -- search AAL's price and options history for the most profitable
   **price-based** options-selling strategy (cash-secured puts, covered calls, or
   the two combined as a Wheel) and check whether it survives out of sample.
2. **Advisor dashboard** -- feed the same rules a live (currently mock) AAL quote
   and your current holdings, and get one recommendation: sell a CSP/CC (with the
   exact strike, expiry, size), wait, buy-to-close, or roll. A second tab scans
   ~850 liquid US stocks for the richest ATM premium (live Yahoo Finance data).

Nothing here trades. It tells you what to do; you execute manually.

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
python scripts/run_all.py            # research: full grid + walk-forward + report (~8 min)
python scripts/run_all.py --quick    # research: small grid, fast sanity pass
python scripts/ml_probe.py           # research: does a simple model beat the price rule?

python scripts/advise.py             # advisor: print one recommendation to the terminal
python dashboard/app.py              # advisor: live dashboard at http://127.0.0.1:5000
python scripts/build_universe.py     # scanner: (re)build the S&P500+Nasdaq ticker universe
```

Research output lands in `results/`: `RESEARCH_REPORT.md` (the readable answer),
grid CSVs, walk-forward and regime tables, per-trade log, equity plots. The
dashboard reads `results/optimized_config.json` for its rules (falls back to
defaults if you have not run the research yet).

## Data

Two independent data layers, each with a mock backend now and a real one later.

### `historical_data/` -- for the research backtest

- `historical_data/stock_aal.csv` -- real AAL OHLCV (Yahoo Finance, auto-cached).
- `historical_data/mock/` -- synthetic option chain written on first run in the
  exact **ORATS EOD "strikes" schema** (Black-Scholes on the real close + a
  modelled IV surface). Used while `historical_data/orats/` is empty.
- `historical_data/orats/` -- drop real ORATS EOD files here (`.csv`/`.parquet`).
  Picked up automatically; `read_orats_strikes()` parses both, so it is zero-code.

### `realtime_data/` -- for the advisor dashboard

- `realtime_data/mock/` -- fake feed from the last cached close (price wanders so
  the dashboard ticks) + an on-the-fly Black-Scholes chain. Also holds
  `mock/portfolio.json` (your current holdings; copy from `portfolio.example.json`).
- `realtime_data/robinhood_api/` -- TODO: real AAL quote + chain via Robinhood.

`src/realtime.py` defines the `RealtimeSource` interface (`quote()` +
`option_chain()`); `default_realtime_source()` picks the backend.

## Premium scanner (dashboard's second tab)

Ranks a universe of stocks by ATM option premium richness, independent of the
AAL wheel rules:

- **Universe**: S&P 500 union the 500 largest Nasdaq-listed stocks by market cap
  (~850 unique tickers after dedup). Built by `scripts/build_universe.py`
  (Wikipedia + Nasdaq screener API) into `realtime_data/universe.json`; this
  changes slowly, so it is a manual/occasional rebuild, not part of the live loop.
- **Per ticker**: nearest expiry in the 25-45 DTE window, the strike closest to
  spot (ATM), and:
  - `cc_reward_pct` = call bid / spot -- covered-call premium as a fraction of
    the stock's market value (the 100-share contract multiplier cancels).
  - `csp_reward_pct` = put bid / strike -- cash-secured-put premium as a
    fraction of the cash collateral required.
  - `*_annualized` versions divide by dte/365.
  - `cc_thin` / `csp_thin` flags a leg with no open interest or volume today
    (quote may be stale/wide -- shown as &#9888; in the UI, not dropped).
- **Data source**: live yfinance quotes/chains (~15-20min delayed, free, no
  auth). Yahoo's unofficial endpoint rate-limits hard (HTTP 429) well before any
  useful concurrency, so `src/scanner.py` scans **sequentially** with a fixed
  gap between tickers (`REQUEST_GAP`) -- do not reintroduce concurrency without
  re-verifying against the rate limit.
- **Refresh**: `src/scanner_job.py` runs a full scan in a background thread on
  dashboard startup (skipped if the on-disk cache is still fresh), then every
  `REFRESH_SECONDS`, writing `realtime_data/scanner_cache.json`. `/api/scan`
  serves the cache instantly; `/api/scan/refresh` (the dashboard's "Refresh
  now" button) wakes the loop early.

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
| `datasource.py` | Historical data boundary. `StockSource` / `OptionSource` interfaces; `YahooStock`, `CsvStock`, `OratsFolder`, `MockOptions` implement them. `load_market_data(...)` returns a `MarketData`. |
| `realtime.py` | Live data boundary. `RealtimeSource` interface; `MockRealtime` today, `RobinhoodRealtime` later. Returns a `Quote` and a normalised option-chain DataFrame. |
| `advisor.py` | Turns the rules + a live quote + your `PortfolioState` into an `Advice`: `compute_live_features()` (the same feature definitions, one latest row) then `advise()`, which dry-runs the legs and reviews open positions to produce ranked `Recommendation`s (ROLL / CLOSE / SELL_CSP / SELL_CC / WAIT). Rolling parameter selection is a TODO. |
| `dashboard/app.py` + `templates/index.html` | Flask, two tabs. AAL Wheel: `/api/advice` (polled every 6s) + `/api/history`. Premium Scanner: `/api/scan` (cache, polled every 15s) + `/api/scan/refresh` (manual trigger). |
| `scanner.py` | Live cross-sectional ATM premium scan (see "Premium scanner" above). `scan_universe(symbols)` -> DataFrame, one row per ticker. |
| `scanner_job.py` | Background loop that owns the scan cadence, disk cache, and manual-refresh wake-up for the dashboard. |
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

- **New historical source**: implement `fetch()` on a class in `datasource.py`.
- **Real-time feed (Robinhood)**: implement `quote()` + `option_chain()` in
  `src/realtime.py` and return it from `default_realtime_source()`. The advisor
  and dashboard only see the interface.
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
