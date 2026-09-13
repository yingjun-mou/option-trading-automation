"""Slow-moving, price-history-derived signals for the scanner, computed
together from one batched download per symbol (see `compute_market_signals`):
the realized-vol-percentile proxy for "IV Rank", the raw 30-day realized vol
(rv_30) the scanner divides into each option's implied vol for the IV/RV
column, and a trailing-closes window the scanner turns into a *live* price
percentile (today's actual spot ranked against recent history, not
yesterday's close -- see scanner_job.py). All are maintained on their own
slow cadence by `iv_rank_job.IvRankJob`, decoupled from the option scan."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

# 30 trading days, not 20 -- chosen (per explicit request) to roughly match the
# scanner's own 25-45 DTE option window, since IV/RV only means much when the
# two tenors are in the same ballpark (see the IV/RV column's docstring below).
RV_WINDOW = 30            # trading days for the realized-vol estimate itself
PRICE_WINDOW = 63         # ~3 trading months, for the price-percentile column
LOOKBACK_DAYS = "1y"      # history window both signals are measured against
CHUNK_SIZE = 50
CHUNK_GAP = 1.5

# Bump whenever compute_market_signals()'s return shape changes (a key renamed,
# added, or dropped) -- IvRankJob checks this against its on-disk cache to force
# an immediate recompute on a shape change, rather than treating a schema-stale
# but recent-timestamped cache as fresh and waiting up to REFRESH_SECONDS (20h)
# to notice. Bit twice already: rv_20->rv_30 (this constant's addition) sat
# silently stale for what would have been hours, since only the *content*
# changed, not the timestamp.
SCHEMA_VERSION = 2


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


def compute_market_signals(symbols: list[str], rv_window: int = RV_WINDOW,
                           price_window: int = PRICE_WINDOW, chunk_size: int = CHUNK_SIZE,
                           chunk_gap: float = CHUNK_GAP) -> dict[str, dict]:
    """Per symbol: `{"iv_rank_pct": ..., "rv_30": ..., "recent_closes": [...]}`,
    any key omitted if there isn't enough history for it. `recent_closes` is
    the trailing `price_window` daily closes -- the scanner combines this
    with each ticker's live spot to compute a price percentile (see
    scanner_job.py) rather than freezing it to yesterday's close here.
    `rv_30` is likewise combined with each ticker's live ATM option IV to
    compute the IV/RV column.

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
                if len(close) >= price_window:
                    signals["recent_closes"] = close.tail(price_window).round(4).tolist()
                if signals:
                    out[sym] = signals
        if i + chunk_size < len(symbols):
            time.sleep(chunk_gap)
    return out
