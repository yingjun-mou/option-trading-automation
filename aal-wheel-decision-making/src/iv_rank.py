"""Two slow-moving, price-history-derived signals for the scanner, computed
together from one batched download per symbol (see `compute_market_signals`):
the realized-vol-percentile proxy for "IV Rank", and a trailing-closes window
the scanner turns into a *live* price percentile (today's actual spot ranked
against recent history, not yesterday's close -- see scanner_job.py). Both
are maintained on their own slow cadence by `iv_rank_job.IvRankJob`,
decoupled from the option scan."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

RV_WINDOW = 20            # trading days for the realized-vol estimate itself
PRICE_WINDOW = 63         # ~3 trading months, for the price-percentile column
LOOKBACK_DAYS = "1y"      # history window both signals are measured against
CHUNK_SIZE = 50
CHUNK_GAP = 1.5


def _extract_close(data: pd.DataFrame, symbol: str, single: bool) -> pd.Series | None:
    """Pull one symbol's Close series out of a (possibly multi-ticker)
    `yf.download` result; None if `symbol` isn't in it."""
    try:
        if single:
            return data["Close"].dropna()
        return data[symbol]["Close"].dropna()
    except (KeyError, TypeError):
        return None


def _rv_rank(close: pd.Series, window: int) -> float | None:
    """Proxy for "IV Rank": where today's realized volatility sits (0-1
    percentile) within its own trailing-1Y realized-vol history.

    This is NOT true IV Rank (which needs a 1Y history of the options
    market's *implied* vol -- Yahoo's free feed has no such series, and nor
    does any other no-cost source for an 850-stock universe). Realized vol
    is a reasonable stand-in for "is this name in an elevated-volatility
    regime right now" but can diverge from true IV Rank, especially around
    known upcoming events (earnings, etc.) that are priced into IV ahead of
    time but obviously can't be "realized" yet."""
    if len(close) < window * 2:
        return None
    logret = np.log(close).diff()
    rv = (logret.rolling(window).std() * np.sqrt(252)).dropna()
    return float(rv.rank(pct=True).iloc[-1]) if len(rv) >= window else None


def compute_market_signals(symbols: list[str], rv_window: int = RV_WINDOW,
                           price_window: int = PRICE_WINDOW, chunk_size: int = CHUNK_SIZE,
                           chunk_gap: float = CHUNK_GAP) -> dict[str, dict]:
    """Per symbol: `{"iv_rank_pct": ..., "recent_closes": [...]}`, either key
    omitted if there isn't enough history for it. `recent_closes` is the
    trailing `price_window` daily closes -- the scanner combines this with
    each ticker's live spot to compute a price percentile (see
    scanner_job.py) rather than freezing it to yesterday's close here.

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
                rank = _rv_rank(close, rv_window)
                if rank is not None:
                    signals["iv_rank_pct"] = rank
                if len(close) >= price_window:
                    signals["recent_closes"] = close.tail(price_window).round(4).tolist()
                if signals:
                    out[sym] = signals
        if i + chunk_size < len(symbols):
            time.sleep(chunk_gap)
    return out
