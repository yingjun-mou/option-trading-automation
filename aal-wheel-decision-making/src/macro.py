"""Macro-regime signals for the dashboard's Macro tab: QQQ trend/momentum
plus the VIX term structure, refreshed on macro_job.MacroJob's own cadence
(see its docstring).

Per an explicit "fetch, don't compute" ask: the 50/200-day moving averages
are Yahoo's own already-computed `fiftyDayAverage`/`twoHundredDayAverage`
fields (read via `Ticker.info`, not recomputed from a rolling window here),
VIX and VIX3M are plain index closes, and the 6-month return and the two
DMA ratios are simple arithmetic on those already-fetched numbers. ADX(14)
and market breadth (% of Nasdaq-100 members above their own 200-day SMA)
are the two exceptions -- no free, no-signup provider exposes ADX (Alpha
Vantage/Twelve Data both gate it behind an API key the user declined to set
up), and a 200DMA-per-stock check across ~100 names has no free per-basket
source either (see _market_breadth's docstring for why that one isn't just
~100 Ticker.info calls). Both are computed here from freely-fetched OHLC
history -- ADX via the same Wilder's-smoothing convention already used by
this project's RSI implementation (see iv_rank._rsi), breadth via a plain
rolling-mean SMA."""

from __future__ import annotations

import io
import json
import time

import numpy as np
import pandas as pd

from .realtime import REALTIME_DIR

QQQ_SYMBOL = "QQQ"
VIX_SYMBOL = "^VIX"
VIX3M_SYMBOL = "^VIX3M"

ADX_WINDOW = 14
RETURN_LOOKBACK_DAYS = 183   # ~6 calendar months
HISTORY_PERIOD = "2y"        # trailing bars needed for a stable ADX(14)/200-day SMA + a clean 6-month return

# Breadth = % of Nasdaq-100 members trading above their own 200-day SMA.
# Wikipedia moved this table at some point from the "Nasdaq-100" article
# itself to this dedicated page -- note the capitalization, MediaWiki titles
# are case-sensitive past the first letter (https://.../Nasdaq-100 has no
# constituent table at all any more, just a hatnote pointing here).
NASDAQ100_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies"
NASDAQ100_CACHE_FILE = REALTIME_DIR / "nasdaq100_constituents.json"
NASDAQ100_MAX_AGE_DAYS = 7   # index membership changes only a few times a year
BREADTH_SMA_WINDOW = 200
BREADTH_CHUNK_SIZE = 50
BREADTH_CHUNK_GAP = 1.5

# Bump whenever compute_macro_signals()'s return shape changes -- same
# stale-cache guard as iv_rank.SCHEMA_VERSION, see that constant's note.
SCHEMA_VERSION = 2


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = ADX_WINDOW) -> float | None:
    """Wilder's-smoothing ADX(n), approximated the same way this project's
    existing RSI does (`.ewm(alpha=1/n, adjust=False)` in place of Wilder's
    classic simple-average seed) -- a standard, widely-used approximation,
    not a from-scratch indicator design."""
    if len(close) < n * 2:
        return None
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1 / n, adjust=False).mean().iloc[-1]
    return float(adx) if pd.notna(adx) else None


def _six_month_return(close: pd.Series) -> float | None:
    """QQQ's simple return over the trailing ~6 calendar months -- plain
    arithmetic on already-fetched closes, not a technical indicator."""
    if close.empty:
        return None
    cutoff = close.index[-1] - pd.Timedelta(days=RETURN_LOOKBACK_DAYS)
    past = close[close.index <= cutoff]
    if past.empty:
        return None
    start, end = float(past.iloc[-1]), float(close.iloc[-1])
    return (end / start) - 1.0 if start else None


def _last_close(symbol: str) -> float | None:
    """Most recent daily close for an index ticker (VIX/VIX3M have no
    options chain or Ticker.info moving averages -- just a price)."""
    import yfinance as yf
    try:
        hist = yf.Ticker(symbol).history(period="5d", interval="1d")
        closes = hist["Close"].dropna()
        return float(closes.iloc[-1]) if not closes.empty else None
    except Exception:
        return None


def _fetch_nasdaq100_from_wikipedia() -> list[str]:
    """Scrapes the current Nasdaq-100 roster -- same pattern as
    scripts/build_universe.py's S&P 500 fetch, just a different page/column."""
    import requests
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    r = requests.get(NASDAQ100_WIKI_URL, headers=headers, timeout=20)
    r.raise_for_status()
    table = pd.read_html(io.StringIO(r.text))[0]
    return sorted({str(s).strip().upper().replace(".", "-") for s in table["Ticker"]})


def _load_nasdaq100_tickers() -> list[str]:
    """Cached Nasdaq-100 constituents -- refetched from Wikipedia only once
    the cache is missing or older than NASDAQ100_MAX_AGE_DAYS, since index
    membership barely changes, unlike everything else on MacroJob's 30-minute
    cadence. Falls back to a stale cache (rather than an empty list) if a
    refetch fails -- a slightly outdated breadth number beats none at all."""
    cached_tickers, cached_at = None, None
    if NASDAQ100_CACHE_FILE.exists():
        try:
            raw = json.loads(NASDAQ100_CACHE_FILE.read_text())
            cached_tickers = raw.get("tickers") or None
            cached_at = pd.Timestamp(raw["fetched_at"])
        except Exception:
            cached_tickers, cached_at = None, None
    if cached_tickers and cached_at is not None:
        age_days = (pd.Timestamp.now(tz=cached_at.tzinfo) - cached_at).total_seconds() / 86400
        if age_days < NASDAQ100_MAX_AGE_DAYS:
            return cached_tickers
    try:
        tickers = _fetch_nasdaq100_from_wikipedia()
        NASDAQ100_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        NASDAQ100_CACHE_FILE.write_text(json.dumps(
            {"fetched_at": pd.Timestamp.now("UTC").isoformat(timespec="seconds"), "tickers": tickers}))
        return tickers
    except Exception:
        return cached_tickers or []


def _extract_close(data: pd.DataFrame, symbol: str, single: bool) -> pd.Series | None:
    """Pull one symbol's Close series out of a (possibly multi-ticker)
    `yf.download` result -- duplicated from iv_rank._extract_close rather
    than imported, per this project's live-scanner-module-tree convention
    (see iv_rank._rsi's docstring)."""
    try:
        if single:
            return data["Close"].dropna()
        return data[symbol]["Close"].dropna()
    except (KeyError, TypeError):
        return None


def _market_breadth(tickers: list[str], window: int = BREADTH_SMA_WINDOW,
                    chunk_size: int = BREADTH_CHUNK_SIZE,
                    chunk_gap: float = BREADTH_CHUNK_GAP) -> tuple[float | None, int]:
    """(fraction of `tickers` whose latest close is above its own trailing
    `window`-day SMA, count of tickers that had enough history to check).

    Unlike QQQ's own 50/200 DMA above, there's no free "already computed"
    per-basket source for this: Yahoo's Ticker.info moving-average fields are
    one HTTP round-trip *per stock*, which at ~100 names on a 30-minute
    cadence is a much heavier, more rate-limit-exposed request pattern than a
    couple of batched downloads. So, like ADX, this is computed here from
    freely-fetched OHLC history -- same chunked yf.download pattern as
    iv_rank.compute_market_signals, just with a plain rolling-mean SMA
    instead of that module's realized-vol/RSI math."""
    import yfinance as yf

    if not tickers:
        return None, 0
    above, total = 0, 0
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        try:
            data = yf.download(tickers=chunk, period=HISTORY_PERIOD, interval="1d",
                               group_by="ticker", auto_adjust=True, progress=False,
                               threads=False)
        except Exception:
            data = None
        if data is not None and not data.empty:
            single = len(chunk) == 1 or not isinstance(data.columns, pd.MultiIndex)
            for sym in chunk:
                close = _extract_close(data, sym, single)
                if close is None or len(close) < window:
                    continue
                sma = close.rolling(window).mean().iloc[-1]
                if pd.isna(sma):
                    continue
                total += 1
                if close.iloc[-1] > sma:
                    above += 1
        if i + chunk_size < len(tickers):
            time.sleep(chunk_gap)
    return (above / total if total else None), total


def compute_macro_signals() -> dict:
    """QQQ trend/momentum + VIX term-structure snapshot. Returns {} on total
    failure so MacroJob just keeps serving its last cached snapshot; any
    individual signal that can't be computed is simply omitted rather than
    failing the whole call."""
    import yfinance as yf

    out: dict = {}
    try:
        qqq = yf.Ticker(QQQ_SYMBOL)
        hist = qqq.history(period=HISTORY_PERIOD, interval="1d")
        if not hist.empty:
            close = hist["Close"].dropna()
            price = float(close.iloc[-1])
            out["qqq_price"] = price

            info = qqq.info or {}
            fifty = info.get("fiftyDayAverage")
            two_hundred = info.get("twoHundredDayAverage")
            if fifty:
                out["fifty_dma"] = float(fifty)
            if two_hundred:
                out["two_hundred_dma"] = float(two_hundred)
            if fifty and two_hundred:
                out["fifty_over_two_hundred"] = float(fifty) / float(two_hundred)
            if two_hundred:
                out["price_over_two_hundred"] = price / float(two_hundred)

            ret_6m = _six_month_return(close)
            if ret_6m is not None:
                out["qqq_return_6m"] = ret_6m

            adx = _adx(hist["High"], hist["Low"], close)
            if adx is not None:
                out["adx_14"] = adx
    except Exception:
        pass

    vix = _last_close(VIX_SYMBOL)
    vix3m = _last_close(VIX3M_SYMBOL)
    if vix is not None:
        out["vix"] = vix
    if vix3m is not None:
        out["vix3m"] = vix3m
    if vix is not None and vix3m:
        out["vix_over_vix3m"] = vix / vix3m

    try:
        tickers = _load_nasdaq100_tickers()
        breadth, breadth_n = _market_breadth(tickers)
        if breadth is not None:
            out["nasdaq100_breadth"] = breadth
            out["nasdaq100_breadth_n"] = breadth_n
    except Exception:
        pass

    return out
