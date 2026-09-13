"""Slow-moving signals for the scanner, computed on `iv_rank_job.IvRankJob`'s
own cadence, decoupled from the option scan:

- `compute_market_signals`: the realized-vol-percentile proxy for "IV Rank",
  the raw 30-day realized vol (rv_30) the scanner divides into each option's
  implied vol for the IV/RV column, that ratio's own trailing-3-month
  percentile (iv_rv_pct), 14-day RSI, and a trailing-closes window the
  scanner turns into a *live* price percentile -- all five derived from ONE
  batched price-history download per symbol.
- `compute_forward_pe`: forward P/E, which (unlike the above) has no batched
  equivalent -- each ticker needs its own `Ticker.info` round-trip -- so it's
  paced sequentially like the option scan, just on this module's much slower
  cadence to keep that extra per-ticker cost off the frequently-run scan.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

# 30 trading days, not 20 -- chosen (per explicit request) to roughly match the
# scanner's own 25-45 DTE option window, since IV/RV only means much when the
# two tenors are in the same ballpark (see the IV/RV column's docstring below).
RV_WINDOW = 30            # trading days for the realized-vol estimate itself
RSI_WINDOW = 14           # standard RSI period
PRICE_WINDOW = 63         # ~3 trading months, for the price-percentile column
LOOKBACK_DAYS = "1y"      # history window all price-based signals are measured against
CHUNK_SIZE = 50
CHUNK_GAP = 1.5

# forwardPE has no batched download -- each Ticker(...).info call is its own
# request. Verified sequential+gap holds up at 140-ticker scale (0/140 failed);
# same defensive pacing discipline as scanner.REQUEST_GAP, just a separate
# constant since this is a different, much less frequent job.
FORWARD_PE_GAP = 0.3
FORWARD_PE_RETRY_BACKOFF = 15.0

# Bump whenever compute_market_signals()'s return shape changes (a key renamed,
# added, or dropped) -- IvRankJob checks this against its on-disk cache to force
# an immediate recompute on a shape change, rather than treating a schema-stale
# but recent-timestamped cache as fresh and waiting up to REFRESH_SECONDS (20h)
# to notice. Bit twice already: rv_20->rv_30 (this constant's addition) sat
# silently stale for what would have been hours, since only the *content*
# changed, not the timestamp.
SCHEMA_VERSION = 4


def _extract_close(data: pd.DataFrame, symbol: str, single: bool) -> pd.Series | None:
    """Pull one symbol's Close series out of a (possibly multi-ticker)
    `yf.download` result; None if `symbol` isn't in it."""
    try:
        if single:
            return data["Close"].dropna()
        return data[symbol]["Close"].dropna()
    except (KeyError, TypeError):
        return None


def _realized_vol(close: pd.Series, window: int) -> tuple[pd.Series | None, float | None]:
    """The trailing annualized realized-vol series and its most recent value
    (rv_30 at the default window), or (None, None) with too little history.

    rv_30's rolling history also backs the "IV Rank" proxy: where today's
    value sits (0-1 percentile) within its own trailing-1Y range.

    That proxy is NOT true IV Rank (which needs a 1Y history of the options
    market's *implied* vol -- Yahoo's free feed has no such series, and nor
    does any other no-cost source for an 850-stock universe). Realized vol
    is a reasonable stand-in for "is this name in an elevated-volatility
    regime right now" but can diverge from true IV Rank, especially around
    known upcoming events (earnings, etc.) that are priced into IV ahead of
    time but obviously can't be "realized" yet."""
    if len(close) < window * 2:
        return None, None
    logret = np.log(close).diff()
    rv = (logret.rolling(window).std() * np.sqrt(252)).dropna()
    if len(rv) < window:
        return None, None
    return rv, float(rv.iloc[-1])


def _rsi(close: pd.Series, n: int = RSI_WINDOW) -> pd.Series:
    """Wilder-smoothed RSI. Same formula as features._rsi, duplicated rather
    than imported -- this live-scanner module tree stays independent of the
    research/backtest one, per this project's existing architecture (see
    scanner.py's module docstring / this project's memory notes)."""
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(50)


def _iv_rv_percentile(rv_series: pd.Series, rv_30: float, window: int) -> float | None:
    """Proxy for "where does today's IV/RV ratio sit within its own trailing
    3-month history" -- without needing 3 months of historical *implied* vol,
    which (as with IV Rank) no free source provides.

    The trick: ratio(t) = IV_now / RV(t) for a fixed IV_now is a strictly
    decreasing function of RV(t) alone, so ranking {ratio(t)} over the past
    `window` days is mathematically identical to *inverse*-ranking {RV(t)}
    over the same days -- today's IV cancels out of the comparison entirely,
    whatever it is. So this is really "today's RV percentile within its own
    trailing 3-month range, flipped" (high recent RV -> low percentile, since
    a rich RV regime makes the *ratio* look cheaper for a given IV), computed
    the moment RV's own history is available -- no live IV needed at all.

    This assumes IV has stayed roughly constant over the window, which is
    the same simplification IV Rank makes; both are proxies for a true
    historical-IV-based signal this project has no free data source for."""
    if len(rv_series) < window:
        return None
    recent = rv_series.tail(window)
    return 1.0 - float((recent <= rv_30).mean())


def compute_market_signals(symbols: list[str], rv_window: int = RV_WINDOW,
                           price_window: int = PRICE_WINDOW, chunk_size: int = CHUNK_SIZE,
                           chunk_gap: float = CHUNK_GAP) -> dict[str, dict]:
    """Per symbol: `{"iv_rank_pct": ..., "rv_30": ..., "iv_rv_pct": ...,
    "rsi_14": ..., "recent_closes": [...]}`, any key omitted if there isn't
    enough history for it. `recent_closes` is the trailing `price_window`
    daily closes -- the scanner combines this with each ticker's live spot to
    compute a price percentile (see scanner_job.py) rather than freezing it to
    yesterday's close here. `rv_30` is likewise combined with each ticker's
    live ATM option IV to compute the IV/RV column; `iv_rv_pct` is that
    ratio's own trailing-3-month percentile (see `_iv_rv_percentile` for why
    it's computable from rv_30's history alone, without live IV). `rsi_14` is
    plain 14-day RSI, unrelated to the IV/RV signals but computed here too
    since it's free from the same already-downloaded close series.

    Batched via yf.download (one/few HTTP calls per chunk) rather than one
    call per ticker -- this runs far less often than the option scan (see
    IvRankJob) so it can afford a modest amount of history per call, but
    still chunked+paced to stay well clear of Yahoo's rate limit.
    """
    import yfinance as yf

    out: dict[str, dict] = {}
    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i:i + chunk_size]
        try:
            data = yf.download(tickers=chunk, period=LOOKBACK_DAYS, interval="1d",
                               group_by="ticker", auto_adjust=True, progress=False,
                               threads=False)
        except Exception:
            data = None
        if data is not None and not data.empty:
            single = len(chunk) == 1 or not isinstance(data.columns, pd.MultiIndex)
            for sym in chunk:
                close = _extract_close(data, sym, single)
                if close is None or close.empty:
                    continue
                signals = {}
                rv_series, rv_30 = _realized_vol(close, rv_window)
                if rv_series is not None:
                    signals["iv_rank_pct"] = float(rv_series.rank(pct=True).iloc[-1])
                    signals["rv_30"] = rv_30
                    iv_rv_pct = _iv_rv_percentile(rv_series, rv_30, price_window)
                    if iv_rv_pct is not None:
                        signals["iv_rv_pct"] = iv_rv_pct
                if len(close) >= RSI_WINDOW * 2:
                    signals["rsi_14"] = float(_rsi(close).iloc[-1])
                if len(close) >= price_window:
                    signals["recent_closes"] = close.tail(price_window).round(4).tolist()
                if signals:
                    out[sym] = signals
        if i + chunk_size < len(symbols):
            time.sleep(chunk_gap)
    return out


def _is_rate_limited(exc: Exception) -> bool:
    """True if `exc` looks like Yahoo's HTTP 429 rather than some other failure."""
    msg = str(exc).lower()
    return "429" in msg or "too many requests" in msg or "rate limit" in msg


def _forward_pe(symbol: str) -> float | None:
    """Forward P/E from `Ticker.info`. Retries once on a 429 before giving up
    -- same defensive pattern as scanner._scan_one, duplicated rather than
    imported to keep this module's failure handling self-contained."""
    import yfinance as yf

    attempt = 0
    while True:
        try:
            pe = yf.Ticker(symbol).info.get("forwardPE")
            return float(pe) if pe is not None and pe > 0 else None
        except Exception as e:  # noqa: BLE001 -- one bad ticker must not sink the batch
            if _is_rate_limited(e) and attempt < 1:
                attempt += 1
                time.sleep(FORWARD_PE_RETRY_BACKOFF * attempt)
                continue
            return None


def compute_forward_pe(symbols: list[str], gap: float = FORWARD_PE_GAP) -> dict[str, float]:
    """{symbol: forward_pe}, one entry per symbol with a usable (positive)
    value. Sequential by design -- see FORWARD_PE_GAP's note above; there is
    no batched equivalent to `Ticker.info` the way there is for price history."""
    out: dict[str, float] = {}
    n = len(symbols)
    for i, sym in enumerate(symbols, 1):
        pe = _forward_pe(sym)
        if pe is not None:
            out[sym] = pe
        if gap and i < n:
            time.sleep(gap)
    return out
