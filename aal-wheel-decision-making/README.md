# aal-wheel-decision-making

Two parts:

1. **Research** -- search AAL's price and options history for the most profitable
   **price-based** options-selling strategy (cash-secured puts, covered calls, or
   the two combined as a Wheel) and check whether it survives out of sample.
2. **Advisor dashboard** ("Stock Trading Automation" in the UI) -- three tabs. **Macro**
   (opens first): a regime read on QQQ + Nasdaq-100 breadth (trend, ADX, VIX term
   structure). **Premium Scanner**: ~850 liquid US stocks ranked by richest ATM
   premium (live Yahoo Finance data). **AAL Wheel**: feed the same research rules a
   live (currently mock) AAL quote and your current holdings, and get one
   recommendation -- sell a CSP/CC (with the exact strike, expiry, size), wait,
   buy-to-close, or roll.

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
defaults if you have not run the research yet). `optimized_config.json`
specifically is checked into git (unlike the rest of `results/`) so a fresh
deploy shows "OPTIMISED" rules out of the box -- it's just tuned strategy
parameters, no personal data.

## Deploying (so you can reach the dashboard from any computer, not just this one)

The dashboard is a normal Flask app; `Procfile` here plus `render.yaml` at
the **repo root** (one level up -- Render's Blueprint discovery only looks
there, never in a subdirectory, even though this project lives inside a
larger repo) target [Render](https://render.com)'s free tier specifically,
since that's what this project was set up against. Free tier means the
instance **spins down after ~15 min idle** and has **no persistent disk** --
every time you open the dashboard after a gap, it's a cold start:

- **Premium Scanner**: no cached scan, so "Refresh now" kicks off a fresh
  multi-minute scan against Yahoo.
- **AAL Wheel**: its context (historical features + live-quote source) now
  builds lazily on first use rather than blocking the app from starting at
  all (see `_get_aal_context()` in `dashboard/app.py` -- this used to be
  eager, and building it doubled as gunicorn's own startup validation,
  which blocked the port from ever opening long enough that Render's deploy
  outright failed with "no open ports detected"). The first request to this
  tab is still slow, though, for a real reason: `historical_data/mock/`'s
  synthetic options file (~124MB, regenerated via a per-day Black-Scholes
  loop across ~13 years) is too large to commit to git, so a fresh deploy
  regenerates it from scratch. `historical_data/stock_aal.csv` (379KB) *is*
  committed, so at least that half is instant and network-free.
- **Macro**: `MacroJob` starts fetching QQQ/VIX ~10s after the app comes up
  (see `src/macro_job.py`'s `STARTUP_DELAY`) -- a few lightweight yfinance
  calls, nowhere near the AAL Wheel tab's regeneration cost, so this tab is
  usually populated within a few seconds of a cold start. The chart itself
  needs no backend data at all (it's a client-side TradingView embed), so it
  renders immediately regardless.

This was a deliberate choice over paid always-on hosting; see this
project's chat history if you want to revisit that tradeoff later, or want
the AAL Wheel tab's cold-start cost reduced further (e.g. a smaller
synthetic-data window for the dashboard's own use, separate from the full
backtest's).

1. Push this repo to GitHub (already the case if you're reading this from a
   clone of it).
2. Create a free account at [render.com](https://render.com) and connect
   your GitHub account.
3. **New +** -> **Blueprint** -> pick this repo. Render reads `render.yaml`
   (root dir is set to `aal-wheel-decision-making` in that file, since this
   project lives inside a larger repo) and proposes the web service --
   review and confirm.
4. Before the first deploy finishes, set the two secret environment
   variables it will prompt for (also editable later under the service's
   **Environment** tab): `DASHBOARD_USER` and `DASHBOARD_PASSWORD`. These
   gate every page and API route behind HTTP Basic Auth (see `dashboard/app.py`)
   -- required, since this becomes reachable by anyone with the URL once
   deployed. Local `python dashboard/app.py` runs stay completely open
   (these env vars are unset there), unchanged from before.
5. Once deployed, Render gives you a `https://<service-name>.onrender.com`
   URL -- open it from any computer, log in with the username/password from
   step 4.

No Blueprint access, or want to configure it by hand instead: create a new
**Web Service**, point it at this repo with **Root Directory**
`aal-wheel-decision-making`, **Build Command** `pip install -r requirements.txt`,
**Start Command** from `Procfile` (`gunicorn --worker-class gthread --workers 1
--threads 4 --timeout 90 --bind 0.0.0.0:$PORT dashboard.app:app` -- the
`--workers 1` is deliberate: `ScannerJob`/`IvRankJob` are in-process
background threads with in-memory state; a second worker process would run
independent, uncoordinated copies of both), same two env vars as step 4.

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

## Premium scanner (dashboard's second tab, opens after Macro)

Ranks a universe of stocks by ATM option premium richness, independent of the
AAL wheel rules:

- **Universe**: S&P 500 union the 500 largest Nasdaq-listed stocks by market cap
  (~850 unique tickers after dedup). Built by `scripts/build_universe.py`
  (Wikipedia + Nasdaq screener API) into `realtime_data/universe.json`; this
  changes slowly, so it is a manual/occasional rebuild, not part of the live loop.
- **Per ticker**: nearest expiry in the 25-45 DTE window, the strike closest to
  spot (ATM), and:
  - `cc_reward_pct` = call premium / spot -- covered-call premium as a fraction
    of the stock's market value (the 100-share contract multiplier cancels).
  - `csp_reward_pct` = put premium / strike -- cash-secured-put premium as a
    fraction of the cash collateral required.
  - `*_annualized` versions divide by dte/365.
  - **premium = the live bid, nothing else** (`cc_live`/`csp_live` says which).
    With no live bid (routinely true outside market hours -- confirmed AAPL/TSLA
    show bid=ask=0 overnight), falls back to the last trade price, but *only* if
    that trade happened within `MAX_CLOSE_STALENESS_DAYS`. Ground truth for why
    this matters: INIO's `lastPrice` was $7.00 from a real trade 9 days earlier
    at a very different spot, bid/ask both 0 -- reporting that as "today's
    premium" is simply wrong, not optimistic, so a stale lastPrice yields no
    number at all rather than a guess.
  - `call_oi`/`call_volume`/`put_oi`/`put_volume` are the raw open-interest and
    same-day volume, exposed for the dashboard's liquidity filters.
  - `iv_rank_pct`, `price_pct`, `rv_30`, `iv_rv_ratio`, and `iv_rv_pct` are
    filled in from `src/iv_rank.py` (see below) -- `iv_rank_pct` and
    `iv_rv_pct` are both **proxies**, not the true historical-implied-vol
    signals they stand in for.
- **Dashboard filters** (client-side, instant, no re-scan needed): IV Rank %
  range (default 30-70), max bid-ask spread $ (default 0.10), min open interest
  (default 1000), max stock price $ (default 500). The spread/OI filters apply
  per leg -- a stock stays listed if either leg qualifies, with the other
  leg's cells blanked; a stock is dropped entirely only if neither leg
  qualifies, or its price/IV Rank fails the stock-level filters.
- **Price %ile column**: today's *live* spot ranked (0-100%) against its
  trailing ~3-month daily closes -- same definition as `advisor.py`'s
  `pct_1y`/`pct_3y` signals (fraction of the window at or below the current
  price), just on a shorter window suited to "is this a rich day to sell a
  covered call" rather than the wheel's multi-year entry-ladder framing.
  Color-coded in the UI: green at or above the 75th percentile, red at or
  below the 25th, amber between. Deliberately mixes a *live* number (spot,
  refreshed every scan cycle) with a *slow* one (the historical window, see
  below) rather than freezing the percentile to yesterday's close, since the
  whole point is catching an intraday move.
- **IV Rank is a proxy**: true IV Rank needs a 1-year history of the *options
  market's own* implied vol, which no free source provides for an 850-stock
  universe. `src/iv_rank.py` instead computes the percentile rank of trailing
  30-day realized volatility within its own 1-year range -- a reasonable "is
  this name in an elevated-vol regime" signal, but it can diverge from real IV
  Rank, especially around known upcoming events (IV prices those in ahead of
  time; realized vol obviously can't yet). Color-coded in the UI the same as
  IV/RV %ile below: green &ge;67th percentile, red &le;33rd, amber between.
  The 30-day window (not the more
  common 20) is a deliberate match to the scanner's own 25-45 DTE option
  window -- see the IV/RV column below, which is the main reason this window
  matters. The same function also returns each symbol's trailing ~3-month
  closes (`recent_closes`), raw trailing 30-day realized vol (`rv_30`), and
  its 3-month percentile (`iv_rv_pct`, see the IV/RV %ile column below) --
  one batched `yf.download` per symbol feeds all four signals rather than
  fetching history four times. Refreshed on
  its own ~daily cadence (`src/iv_rank_job.py`, `IvRankJob`,
  `realtime_data/iv_rank_cache.json`) since none of them move much within a
  day and every extra Yahoo call is extra rate-limit risk -- decoupled from
  the option scan's cadence (see "Refresh" below) on purpose. The cache file
  carries a `schema_version` (`iv_rank.SCHEMA_VERSION`, bump on any change to
  `compute_market_signals()`'s return shape) -- a version mismatch forces a
  prompt recompute instead of treating a schema-stale but recent-timestamped
  cache as fresh and waiting out the full refresh interval; hit this for real
  during development (rv_20->rv_30, then again adding `iv_rv_pct`) before the
  guard existed. Unlike the
  option scan, this one still runs on its own timer; it's a single
  lightweight `yf.download` batch, not 800+ sequential ticker fetches, so an
  unattended timer is a much smaller rate-limit bet.
- **IV/RV column**: this expiry's ATM implied vol (average of the call and
  put legs' `impliedVolatility`, which can differ slightly due to skew) over
  `rv_30`. >1x means the market is pricing more movement than has actually
  happened lately -- a classic "is premium rich" read, but still a rough one
  even with the 30-day/25-45-DTE tenor match: it's a fixed trailing window,
  not a forecast of realized vol over the option's own remaining life. Also
  floors out `impliedVolatility` below 1% as an unreliable placeholder (same
  failure mode as the stale-`lastPrice` bug: seen live on a thin contract
  reporting IV=0.0016%, obviously not real). Color-coded: red below 0.9x,
  amber 0.9-1.3x, green 1.3-2.0x, blue above 2.0x -- an extra band beyond
  green since >2x is a materially different situation, not just "more of the
  same rich."
- **IV/RV %ile (3mo) column**: where today's IV/RV ratio sits (0-100%)
  against its own trailing 3 months. A **proxy** for the same reason as IV
  Rank -- no free source has 3 months of historical implied vol -- computed
  by `iv_rank._iv_rv_percentile()`, which exploits the fact that
  ratio(t) = IV_now / RV(t) for a *fixed* IV_now is a strictly decreasing
  function of RV(t) alone: ranking the ratio over the past 3 months is
  mathematically identical to *inverse*-ranking `rv_30`'s own 3-month history,
  so this needs no historical IV at all -- just `rv_30`'s rolling series,
  already computed for IV Rank. This assumes IV has stayed roughly constant
  over the window, the same simplification IV Rank makes. Colored the same
  as IV Rank: green &ge;67th percentile, red &le;33rd, amber between.
- **RSI column**: standard 14-day RSI, same Wilder-smoothing formula as
  `features._rsi` (duplicated rather than imported -- this live-scanner
  module tree stays independent of the research/backtest one). Free to
  compute -- same already-downloaded close series as the signals above, no
  extra Yahoo call. Colored **inverted** from the percentile columns: red
  &ge;70 (overbought), green &le;30 (oversold), amber between.
- **F. PE column**: forward P/E from `Ticker.info["forwardPE"]` (Yahoo's
  analyst-consensus forward-earnings estimate). Unlike every other IvRankJob
  signal, this has **no batched fetch** -- each ticker needs its own
  `Ticker.info` round-trip, verified sequentially-pacable the same way as the
  option scan (0/140 failures in testing at a 0.3s gap) but adding real time
  to `IvRankJob`'s own cycle (~8-9 more minutes for the full universe) since
  there's no `yf.download`-style batch endpoint for fundamentals. Kept off
  the option scan's cadence entirely for this reason -- seconds of extra
  latency per ticker times 800+ tickers on a manually-triggered scan the user
  is actively waiting on would be a worse tradeoff than adding it to the
  already-infrequent, unattended IvRankJob cycle. Colored inverted like RSI:
  red &gt;22 (expensive), green &le;15 (cheap), amber between.
- **Data source**: live yfinance quotes/chains (~15-20min delayed, free, no
  auth). Yahoo's unofficial endpoint rate-limits hard (HTTP 429) well before any
  useful concurrency, so `src/scanner.py` scans **sequentially** with a fixed
  gap between tickers (`REQUEST_GAP`) -- do not reintroduce concurrency without
  re-verifying against the rate limit.
- **Refresh is manual, not on a timer**: `src/scanner_job.py` runs one scan in
  a background thread on a cold start (no cache on disk yet, so there's
  something to show), then does nothing further until `/api/scan/refresh`
  (the dashboard's "Refresh now" button) wakes it. Deliberately not on an
  automatic cadence -- 800+ tickers against a rate-limited free API isn't
  something to fire unattended on a timer; a human decides when a rescan is
  worth the ~8-9 minutes and the rate-limit exposure. `/api/scan` always
  serves whatever is cached, updated or not. Each scan also merges in the
  latest `iv_rank_pct` + a live `price_pct` per symbol from `IvRankJob`
  (injected as `iv_rank_provider`, so this module doesn't need to know that
  job's cadence or cache format).

## Macro tab (dashboard's first/default tab)

A regime read on QQQ + the VIX term structure and Nasdaq-100 breadth,
unrelated to any single stock's option chain:

- **Chart**: QQQ candles with 50- and 200-day SMA lines, via a free,
  no-signup TradingView "Advanced Chart" embed (`dashboard/templates/index.html`,
  `initMacroChart()`). The chart and both moving-average lines are rendered
  entirely by TradingView -- this project fetches nothing for it and computes
  none of it.
- **Snapshot table** (`src/macro.py`'s `compute_macro_signals()`, refreshed
  every 30 min by `src/macro_job.py`'s `MacroJob`, `realtime_data/macro_cache.json`):
  - `fifty_dma`/`two_hundred_dma` are Yahoo's own already-computed
    `fiftyDayAverage`/`twoHundredDayAverage` fields (`Ticker.info`) -- fetched,
    not recomputed here, per an explicit "don't compute the moving averages
    yourself" requirement. They can differ slightly from the chart's own
    TradingView-rendered lines (different providers, different
    adjusted-close/timing conventions) -- that's expected, not a bug.
  - `qqq_return_6m`, `price_over_two_hundred`, `fifty_over_two_hundred` are
    plain arithmetic on those already-fetched numbers (a % change and two
    ratios) -- colored green/red/amber (above/below/exactly at 0 or 1).
  - `adx_14` is the **one indicator this project computes itself** on this
    tab: no free, no-signup source exposes ADX (Alpha Vantage and Twelve Data
    both gate it behind an API key), so it's Wilder's-smoothing ADX(14) over
    QQQ's own OHLC history, using the same `.ewm(alpha=1/n, adjust=False)`
    convention already used by this project's RSI (`iv_rank._rsi`). Colored
    red &lt;20 (no trend), green &gt;25 (trending), amber between.
  - `vix`/`vix3m` are plain index closes (`^VIX`/`^VIX3M` via yfinance); `vix`
    is shown uncolored (context only), `vix_over_vix3m` is colored green
    &gt;1 (near-term vol richer than 3-month -- backwardation, often a
    stress signal), red otherwise (normal contango).
  - `nasdaq100_breadth` = % of the Nasdaq-100's ~100 members trading above
    their own 200-day SMA. Constituent list scraped from Wikipedia's
    [List of NASDAQ-100 companies](https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies)
    (same `requests` + `pd.read_html` pattern as `scripts/build_universe.py`'s
    S&amp;P 500 fetch; note the capitalization -- the "Nasdaq-100" article
    itself dropped its constituent table at some point and just hatnotes to
    this page now) and cached for `NASDAQ100_MAX_AGE_DAYS` (7) since index
    membership barely changes. Like ADX, this is the **other indicator
    computed here rather than fetched**: there's no free per-basket "already
    computed 200DMA" source for ~100 names, and checking each one via
    `Ticker.info` would be ~100 HTTP round-trips every 30 minutes -- far more
    Yahoo request volume than the two batched `yf.download` calls this uses
    instead (same chunked pattern as `iv_rank.compute_market_signals`, just a
    plain rolling-mean SMA). Colored green &gt;60%, red &lt;40%, amber
    between.
- **Slow score (tier 1)** and **market regime** (`_slow_score`, `_classify_regime`
  in `src/macro.py`) -- the first tier of a planned 2-tier scoring system:
  - The score sums four independent +1/-1/0 conditions to a **-4 to +4**
    total (0 both in a condition's stated neutral zone and when an input is
    missing): `Price/SMA200` &gt;1.03x (+1) / &lt;0.97x (-1);
    `SMA50/SMA200` &gt;1.01x (+1) / &lt;0.99x (-1); `Slope_200` &gt;+0.5%
    (+1) / &lt;-0.5% (-1); `+DI &gt; -DI` (+1) / `+DI &lt; -DI` (-1) (ADX's own
    two directional components -- whether rising or falling momentum
    currently dominates, both now returned by `_directional_movement`
    alongside ADX itself, refactored from the old single-value `_adx`).
  - This score needs its **own** SMA50/SMA200 rolling calculation, kept
    deliberately separate from the snapshot table's Yahoo-fetched
    `fifty_dma`/`two_hundred_dma` above: `Slope_200` needs the 200-day
    average's value *20 trading days ago*, and Yahoo's `Ticker.info` field
    only ever has *today's* value -- there's no fetching a historical SMA
    series without paying for an indicator API, so the whole score (level
    checks included) uses one internally consistent locally-computed
    SMA50/SMA200 rather than mixing a fetched "today" number with a
    computed "20-days-ago" one for the same average.
  - **Slope_200 direction, flagged explicitly**: implemented as
    `SMA200(t)/SMA200(t-20) - 1` (positive while the average is rising),
    not the literal `SMA200(t-20)/SMA200(t) - 1` as originally specified --
    that version is negative while the average rises and positive while it
    falls, which would flip the "+1 if Slope_200 &gt; +0.5%" rule backwards
    relative to the other three bullish-when-true conditions in the same
    score. Implemented the internally-consistent way; revert `_slope_200`
    in `src/macro.py` if the literal formula was actually intended.
  - **Market regime** combines the score with ADX: Strong Bull (score
    &ge;3, ADX &ge;25), Bull (score &ge;2, ADX &lt;25), Strong Bear (score
    &le;-3, ADX &ge;25), Bear (score &le;-2, ADX &lt;25), else Sideways --
    which, per the literal rules, also catches gaps like a +2 score
    alongside a &ge;25 ADX (strong-trend-confirmed but not a high enough
    score for either Bull tier), not just genuinely flat readings.
  - Unlike `IvRankJob`'s ~daily cadence, `MacroJob` refreshes every 30
    minutes -- QQQ/VIX/breadth genuinely move during the trading day unlike
    the scanner's slower-moving per-stock signals. Still cheap relative to
    the 850-symbol scan: one symbol's history, one `Ticker.info`, two index
    closes, and ~100 more symbols' history via 2 batched downloads for
    breadth.

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
| `dashboard/app.py` + `templates/index.html` | Flask, three tabs. AAL Wheel: `/api/advice` (polled every 6s) + `/api/history`. Premium Scanner: `/api/scan` (cache, polled every 15s) + `/api/scan/refresh` (manual trigger). Macro: `/api/macro` (cache, polled every 60s). |
| `scanner.py` | Live cross-sectional ATM premium scan (see "Premium scanner" above). `scan_universe(symbols)` -> DataFrame, one row per ticker. |
| `scanner_job.py` | Background loop that owns the scan cadence, disk cache, and manual-refresh wake-up for the dashboard; merges in `iv_rank_pct` from an injected provider. |
| `iv_rank.py` | `compute_market_signals(symbols)` -- the realized-vol-percentile proxy for IV Rank, each symbol's trailing closes (for the live Price %ile column), raw `rv_30` (for the IV/RV column), `iv_rv_pct` (that ratio's own 3-month percentile), and `rsi_14`, all batched via one `yf.download` per symbol. `compute_forward_pe(symbols)` -- forward P/E, sequential (no batch endpoint for fundamentals). Carries `SCHEMA_VERSION` for `IvRankJob`'s cache-invalidation check. |
| `iv_rank_job.py` | Background loop maintaining that proxy on its own slow (~daily) cadence, decoupled from the option scan. |
| `macro.py` | `compute_macro_signals()` -- QQQ price + Yahoo's own 50/200-day averages, the 6-month return, ADX(14) + its +DI/-DI, the VIX/VIX3M term structure, Nasdaq-100 breadth, and the tier-1 slow score + market regime classification (ADX/breadth/the slow score's own SMA50/SMA200 are computed here rather than fetched -- see "Macro tab" above). |
| `macro_job.py` | Background loop refreshing that snapshot every 30 minutes, cached to `realtime_data/macro_cache.json`. |
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
