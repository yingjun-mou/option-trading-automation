"""Macro-regime signals for the dashboard's Macro tab: QQQ trend/momentum
plus the VIX term structure, refreshed on macro_job.MacroJob's own cadence
(see its docstring).

Per an explicit "fetch, don't compute" ask: the 50/200-day moving averages
*shown in the snapshot table* are Yahoo's own already-computed
`fiftyDayAverage`/`twoHundredDayAverage` fields (read via `Ticker.info`, not
recomputed from a rolling window here), VIX and VIX3M are plain index
closes, and the 6-month return and the two DMA ratios are simple arithmetic
on those already-fetched numbers. ADX(14) and market breadth (% of
Nasdaq-100 members above their own 200-day SMA) are computed here instead --
no free, no-signup provider exposes ADX (Alpha Vantage/Twelve Data both gate
it behind an API key the user declined to set up), and a 200DMA-per-stock
check across ~100 names has no free per-basket source either (see
_market_breadth's docstring for why that one isn't just ~100 Ticker.info
calls). Both are computed from freely-fetched OHLC history -- ADX via the
same Wilder's-smoothing convention already used by this project's RSI
implementation (see iv_rank._rsi), breadth via a plain rolling-mean SMA.

The tier-1 "slow score" (see `_slow_score`) needs a second, independent
50/200-day SMA calculation of its own: its `Slope_200` component needs the
200-day SMA's value *20 trading days ago*, which Yahoo's Ticker.info field
only ever gives for *today* -- there is no getting a historical SMA series
without either computing it or paying for an indicator API. So the whole
slow score (price vs SMA200, SMA50 vs SMA200, slope, +DI/-DI) is computed
from one internally-consistent rolling SMA calculation over freely-fetched
OHLC history, deliberately kept separate from the snapshot table's
Yahoo-fetched fifty_dma/two_hundred_dma above (mixing a fetched "today"
value with a locally-computed 20-day-old one would make the slope's own
denominator inconsistent with its numerator)."""

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

# Tier-1 "slow score": +1/-1 per condition below (0 if in the neutral zone
# between the two thresholds, or the input is missing), summed to a -4..+4
# total. Thresholds given explicitly by the user.
SLOW_SCORE_SMA50_WINDOW = 50
SLOW_SCORE_SMA200_WINDOW = 200
SLOW_SCORE_SLOPE_LOOKBACK = 20   # trading days
PRICE_VS_SMA200_UP = 1.03
PRICE_VS_SMA200_DOWN = 0.97
SMA50_VS_SMA200_UP = 1.01
SMA50_VS_SMA200_DOWN = 0.99
SLOPE_200_UP = 0.005
SLOPE_200_DOWN = -0.005
REGIME_ADX_THRESHOLD = 25

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
SCHEMA_VERSION = 3


def _directional_movement(high: pd.Series, low: pd.Series, close: pd.Series,
                          n: int = ADX_WINDOW) -> tuple[float | None, float | None, float | None]:
    """Wilder's-smoothing (adx, plus_di, minus_di) -- ADX(n) plus its two
    directional components: +DI/-DI say whether rising momentum (+DM) or
    falling momentum (-DM) currently dominates, which is what the slow
    score's "+DI>-DI" condition below actually reads. Approximated the same
    way this project's existing RSI does (`.ewm(alpha=1/n, adjust=False)` in
    place of Wilder's classic simple-average seed) -- a standard,
    widely-used approximation, not a from-scratch indicator design.
    (None, None, None) with too little history."""
    if len(close) < n * 2:
        return None, None, None
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
    plus_di_series = 100 * pd.Series(plus_dm, index=high.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    minus_di_series = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (plus_di_series - minus_di_series).abs() / (plus_di_series + minus_di_series)
    adx = dx.ewm(alpha=1 / n, adjust=False).mean().iloc[-1]
    plus_di, minus_di = plus_di_series.iloc[-1], minus_di_series.iloc[-1]
    return (float(adx) if pd.notna(adx) else None,
            float(plus_di) if pd.notna(plus_di) else None,
            float(minus_di) if pd.notna(minus_di) else None)


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


def _slope_200(sma200: pd.Series, lookback: int = SLOW_SCORE_SLOPE_LOOKBACK) -> float | None:
    """SMA200(t)/SMA200(t-lookback) - 1 -- positive when the 200-day average
    has been rising over the last `lookback` trading days (bullish), negative
    when falling (bearish).

    NOTE on direction: the request that specified this score wrote the ratio
    the other way around (SMA200(t-lookback)/SMA200(t) - 1). Taken literally,
    that version is NEGATIVE while the average is rising and POSITIVE while
    it's falling -- which would flip the "+1 if Slope_200 > +0.5%" rule
    backwards relative to the other three bullish conditions in this same
    score (price above its 200-day average, 50-day above 200-day, +DI>-DI
    all read bullish-when-true; a slope that's positive during a *decline*
    would be the odd one out). Implemented here the way that keeps the whole
    scorecard internally consistent; flagged explicitly rather than silently
    guessed -- flip this fraction if the literal formula was actually
    intended."""
    if len(sma200) < lookback + 1:
        return None
    now, past = sma200.iloc[-1], sma200.iloc[-1 - lookback]
    if pd.isna(now) or pd.isna(past) or not past:
        return None
    return float(now / past - 1.0)


def _slow_score(price: float | None, sma50: float | None, sma200: float | None,
                slope_200: float | None, plus_di: float | None,
                minus_di: float | None) -> tuple[int, dict]:
    """Tier-1 "slow score": sums four +1/-1/0 conditions to a -4..+4 total.
    Returns (total, {condition: contribution}) so the UI can show each
    factor's own contribution, not just the sum. A condition contributes 0
    both in its stated neutral zone and when an input is missing."""
    def _bucket(ratio: float | None, up: float, down: float) -> int:
        if ratio is None:
            return 0
        if ratio > up:
            return 1
        if ratio < down:
            return -1
        return 0

    parts = {
        "price_vs_sma200": _bucket(price / sma200 if (price and sma200) else None,
                                   PRICE_VS_SMA200_UP, PRICE_VS_SMA200_DOWN),
        "sma50_vs_sma200": _bucket(sma50 / sma200 if (sma50 and sma200) else None,
                                   SMA50_VS_SMA200_UP, SMA50_VS_SMA200_DOWN),
        "slope_200": _bucket(slope_200, SLOPE_200_UP, SLOPE_200_DOWN),
        "directional": (0 if plus_di is None or minus_di is None
                        else (1 if plus_di > minus_di else (-1 if plus_di < minus_di else 0))),
    }
    return sum(parts.values()), parts


def _classify_regime(slow_score: int | None, adx: float | None) -> str | None:
    """Strong Bull/Bull/Sideways/Bear/Strong Bear from the slow score + ADX,
    per the user's explicit table. The four named bands don't overlap (score
    can't be both >=3 and <=-3; ADX can't be both >=25 and <25), so order
    doesn't matter among them -- "Sideways" is genuinely everything else,
    including e.g. a score of +2 alongside a >=25 ADX (a real gap in the
    literal rules as given: strong-trend-confirmed but the score itself
    isn't high enough to call it a Strong Bull), not just the "no score"
    middle ground."""
    if slow_score is None or adx is None:
        return None
    if slow_score >= 3 and adx >= REGIME_ADX_THRESHOLD:
        return "Strong Bull"
    if slow_score >= 2 and adx < REGIME_ADX_THRESHOLD:
        return "Bull"
    if slow_score <= -3 and adx >= REGIME_ADX_THRESHOLD:
        return "Strong Bear"
    if slow_score <= -2 and adx < REGIME_ADX_THRESHOLD:
        return "Bear"
    return "Sideways"


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

            adx, plus_di, minus_di = _directional_movement(hist["High"], hist["Low"], close)
            if adx is not None:
                out["adx_14"] = adx
            if plus_di is not None:
                out["plus_di_14"] = plus_di
            if minus_di is not None:
                out["minus_di_14"] = minus_di

            # Slow score's own SMA50/SMA200 -- deliberately separate from
            # fifty_dma/two_hundred_dma above, see this module's docstring.
            sma50_series = close.rolling(SLOW_SCORE_SMA50_WINDOW).mean()
            sma200_series = close.rolling(SLOW_SCORE_SMA200_WINDOW).mean()
            sma50_calc = sma50_series.iloc[-1] if not sma50_series.empty else None
            sma200_calc = sma200_series.iloc[-1] if not sma200_series.empty else None
            sma50_calc = float(sma50_calc) if pd.notna(sma50_calc) else None
            sma200_calc = float(sma200_calc) if pd.notna(sma200_calc) else None
            if sma50_calc is not None:
                out["sma50_calc"] = sma50_calc
            if sma200_calc is not None:
                out["sma200_calc"] = sma200_calc

            slope_200 = _slope_200(sma200_series)
            if slope_200 is not None:
                out["slope_200"] = slope_200

            slow_score, slow_score_parts = _slow_score(
                price, sma50_calc, sma200_calc, slope_200, plus_di, minus_di)
            out["slow_score"] = slow_score
            out["slow_score_parts"] = slow_score_parts
            regime = _classify_regime(slow_score, adx)
            if regime is not None:
                out["market_regime"] = regime
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
