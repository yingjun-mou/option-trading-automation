"""Watchlist tab: a personal, static ticker list (realtime_data/watchlist.csv,
git-ignored -- see that file's header) rendered as one row per ticker grouped
by category, plus live price/return columns computed from one batched
yf.download per chunk -- same shape as iv_rank.py's compute_market_signals,
just a different set of derived columns (price + trailing-window % changes
instead of vol/RSI/IV-rank). watchlist_job.WatchlistJob owns the background
cadence and disk cache; app.py's /api/watchlist merges the two."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import pandas as pd

from .realtime import REALTIME_DIR

WATCHLIST_FILE = REALTIME_DIR / "watchlist.csv"

# Yahoo has no quote under the bare symbol for these -- the CSV keeps the
# ticker the user actually wants to see (BTC, BRK.B), this maps it to what
# Yahoo's API/chart URL actually needs (confirmed live: "BRK.B" 404s, Yahoo
# uses a dash for share classes). Extend as more such mismatches turn up.
YAHOO_SYMBOL_ALIASES = {"BTC": "BTC-USD", "BRK.B": "BRK-B"}

LOOKBACK = "10y"   # well past the 5Y window below, so that window always has a true start-of-range close
CHUNK_SIZE = 50
CHUNK_GAP = 1.5

# Trading-day offsets, same convention as iv_rank.py's RV_WINDOW/PRICE_WINDOW
# (~21 trading days/month, ~252/year). YTD is calendar-anchored instead (see
# _ytd_change) since "day of the year" has no fixed trading-day count.
WINDOWS = {
    "chg_1d": 1,
    "chg_1w": 5,
    "chg_1m": 21,
    "chg_3m": 63,
    "chg_1y": 252,
    "chg_5y": 252 * 5,
}


def load_watchlist_rows(path: Path = WATCHLIST_FILE) -> list[dict]:
    """Category,Note,Stock rows in file order. The table's category grouping
    relies on each category's rows being contiguous in the file -- true of
    how this file has always been maintained, and simplest to just preserve
    rather than re-sort (re-sorting would also undo the deliberate ordering
    of categories themselves, roughly most- to least-core)."""
    if not path.exists():
        return []
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            category = (row.get("Category") or "").strip()
            stock = (row.get("Stock") or "").strip()
            if not category or not stock:
                continue
            rows.append({
                "category": category,
                "note": (row.get("Note") or "").strip(),
                "stock": stock,
                "yahoo_symbol": YAHOO_SYMBOL_ALIASES.get(stock, stock),
            })
    return rows


def _extract_close(data: pd.DataFrame, symbol: str, single: bool) -> pd.Series | None:
    """Pull one symbol's Close series out of a (possibly multi-ticker)
    `yf.download` result; None if `symbol` isn't in it. Duplicated from
    iv_rank.py's identical helper rather than imported -- see that module's
    note on keeping this live-data tree independent of the research one."""
    try:
        if single:
            return data["Close"].dropna()
        return data[symbol]["Close"].dropna()
    except (KeyError, TypeError):
        return None


def _pct_change(close: pd.Series, periods: int) -> float | None:
    if len(close) <= periods:
        return None
    base = float(close.iloc[-1 - periods])
    return (float(close.iloc[-1]) / base - 1.0) if base else None


def _ytd_change(close: pd.Series) -> float | None:
    """vs. the first close of the current calendar year (by the series' own
    last timestamp, so this still works correctly if run right after a new
    year starts, before much history has accumulated for it)."""
    this_year = close[close.index.year == close.index[-1].year]
    if this_year.empty:
        return None
    base = float(this_year.iloc[0])
    return (float(close.iloc[-1]) / base - 1.0) if base else None


def compute_price_changes(symbols: list[str], chunk_size: int = CHUNK_SIZE,
                          chunk_gap: float = CHUNK_GAP) -> dict[str, dict]:
    """{symbol: {"price": ..., "chg_1d": ..., "chg_1w": ..., "chg_1m": ...,
    "chg_3m": ..., "chg_ytd": ..., "chg_1y": ..., "chg_5y": ...}}, any window
    key omitted if there isn't enough history for it yet -- normal for a
    recent IPO (a 5Y return is meaningless for a stock that listed last
    year), and for `symbols` Yahoo has no quote for at all (the whole entry
    is omitted then).

    Batched via yf.download (one/few HTTP calls per chunk), same pattern and
    pacing as iv_rank.compute_market_signals -- this is a far smaller
    universe (the watchlist, not 500+ tickers) so it can run on a much
    shorter cadence (see WatchlistJob) without extra rate-limit risk.
    """
    import yfinance as yf

    out: dict[str, dict] = {}
    uniq = list(dict.fromkeys(symbols))  # de-dupe (ASML appears under two categories), keep order
    for i in range(0, len(uniq), chunk_size):
        chunk = uniq[i:i + chunk_size]
        try:
            data = yf.download(tickers=chunk, period=LOOKBACK, interval="1d",
                               group_by="ticker", auto_adjust=True, progress=False,
                               threads=False)
        except Exception:
            data = None
        if data is not None and not data.empty:
            # NOT len(chunk) == 1 -- confirmed live, yf.download can still return
            # MultiIndex columns for a one-symbol request (seen with "BTC-USD"),
            # so that shortcut silently dropped the whole chunk (data["Close"]
            # raised KeyError, swallowed below). The column shape itself is the
            # only reliable signal.
            single = not isinstance(data.columns, pd.MultiIndex)
            for sym in chunk:
                close = _extract_close(data, sym, single)
                if close is None or close.empty:
                    continue
                signals = {"price": float(close.iloc[-1])}
                for key, periods in WINDOWS.items():
                    v = _pct_change(close, periods)
                    if v is not None:
                        signals[key] = v
                ytd = _ytd_change(close)
                if ytd is not None:
                    signals["chg_ytd"] = ytd
                out[sym] = signals
        if i + chunk_size < len(uniq):
            time.sleep(chunk_gap)
    return out
