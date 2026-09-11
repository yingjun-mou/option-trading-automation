from __future__ import annotations

import time

import numpy as np
import pandas as pd

RV_WINDOW = 20            # trading days for the realized-vol estimate itself
LOOKBACK_DAYS = "1y"      # history window the rank is measured against
CHUNK_SIZE = 50
CHUNK_GAP = 1.5


def _extract_close(data: pd.DataFrame, symbol: str, single: bool) -> pd.Series | None:
    try:
        if single:
            return data["Close"].dropna()
        return data[symbol]["Close"].dropna()
    except (KeyError, TypeError):
        return None


def compute_rv_rank(symbols: list[str], window: int = RV_WINDOW,
                    chunk_size: int = CHUNK_SIZE, chunk_gap: float = CHUNK_GAP) -> dict[str, float]:
    """Proxy for "IV Rank": where today's realized volatility sits (0-1
    percentile) within its own trailing-1Y realized-vol history.

    This is NOT true IV Rank (which needs a 1Y history of the options
    market's *implied* vol -- Yahoo's free feed has no such series, and nor
    does any other no-cost source for an 850-stock universe). Realized vol
    is a reasonable stand-in for "is this name in an elevated-volatility
    regime right now" but can diverge from true IV Rank, especially around
    known upcoming events (earnings, etc.) that are priced into IV ahead of
    time but obviously can't be "realized" yet.

    Batched via yf.download (one/few HTTP calls per chunk) rather than one
    call per ticker -- this runs far less often than the option scan (see
    IvRankJob) so it can afford a modest amount of history per call, but
    still chunked+paced to stay well clear of Yahoo's rate limit.
    """
    import yfinance as yf

    out: dict[str, float] = {}
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
                if close is None or len(close) < window * 2:
                    continue
                logret = np.log(close).diff()
                rv = logret.rolling(window).std() * np.sqrt(252)
                rv = rv.dropna()
                if len(rv) < window:
                    continue
                out[sym] = float(rv.rank(pct=True).iloc[-1])
        if i + chunk_size < len(symbols):
            time.sleep(chunk_gap)
    return out
