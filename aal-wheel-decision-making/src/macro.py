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
denominator inconsistent with its numerator).

The tier-2 "reversal" counts (see `_reversal_parts`) reuse tier 1's SMA50
and need one more of their own (SMA20), plus ADX's own value *10 trading
days ago* for its momentum condition -- same "no fetched field gives a
historical value" situation as Slope_200, so `_directional_movement` now
returns full ADX/+DI/-DI series (not just today's value) precisely so this
module can read both today's and 10-days-ago's ADX off the one calculation.

`_regime_history` reuses that same "read an earlier point off an already-
computed Series" trick one step further: it re-runs the *entire*
classification (both tiers + `_classify_regime`) as of each of the last
`REGIME_HISTORY_DAYS` trading days, not just today, so the Macro tab can
show whether the regime has been stable or flip-flopping -- large swings in
either score can flip the labeled regime day to day even when the
underlying trend hasn't really changed, and that's exactly the kind of
noise a single "as of today" label can't reveal on its own."""

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

# Tier-2 "reversal" counts: 5 plain strict-inequality conditions (no
# percentage-band neutral zone, except the ADX-momentum one), each
# contributing to a bullish-tilt count OR a bearish-tilt count (never both)
# -- see _reversal_parts. A reversal regime only fires against an opposing
# tier-1 score (bullish reversal needs slow_score<=-2, bearish needs >=2),
# so these are kept as two separate 0-5 counts rather than netted into one
# signed score the way the slow score is.
REVERSAL_SMA20_WINDOW = 20
REVERSAL_ADX_MOMENTUM_LOOKBACK = 10   # trading days
REVERSAL_ADX_MOMENTUM_UP = 3.0
REVERSAL_ADX_MOMENTUM_DOWN = -3.0
REVERSAL_COUNT_THRESHOLD = 4          # out of 5, either direction

# How many extra days (before today) the Macro tab shows as small "regime
# history" tags, so a flip-flopping classification is visible at a glance
# rather than only ever showing today's possibly-noisy snapshot.
REGIME_HISTORY_DAYS = 3

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

# US 10-year Treasury yield chart: nominal + real (TIPS), overlaid.
# TradingView's free embed doesn't carry either series -- confirmed live:
# FRED:DFII10 (and the "USA10YRY" alternates some sites use) refuses to
# render in the Advanced Chart widget with "this symbol is only available
# on TradingView", a real restriction on their public embed product, not
# something fixable from this side. So instead both series are fetched
# straight from FRED's own public CSV endpoint (no API key needed, unlike
# FRED's REST API) and the dashboard draws them itself via Lightweight
# Charts, fed the fetched points (see index.html's drawTreasuryChart()) --
# still "fetch, don't compute": the Fed publishes both constant-maturity
# yields directly, this project only connects the already-fetched points.
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
REAL_YIELD_SERIES_ID = "DFII10"     # 10-Year Treasury Inflation-Indexed Security, Constant Maturity ("real" yield)
NOMINAL_YIELD_SERIES_ID = "DGS10"   # 10-Year Treasury Constant Maturity Rate ("nominal" yield)
YIELD_HISTORY_POINTS = 504          # ~2 trading years of daily observations, matching HISTORY_PERIOD

# US "the interest rate" chart -- the Fed's own overnight policy rate, not
# a market-priced Treasury yield like the two above. DFF (daily effective
# federal funds rate) rather than FEDFUNDS (the monthly-average version) to
# match the daily granularity of the yield series above. Same FRED CSV
# fetch, same reasoning for not using a TradingView embed.
FED_FUNDS_SERIES_ID = "DFF"         # Daily Effective Federal Funds Rate

# Sector charts: 9 sector-ETF price lines for the Macro tab's 3x3 grid.
# Plain close prices via yfinance (one batched call, same pattern as
# iv_rank.compute_market_signals/_market_breadth above), not a technical
# indicator -- the dashboard draws each as a simple Lightweight Charts line,
# same style as the interest-rate charts above, rather than the fuller
# TradingView candlestick embed the QQQ chart uses.
#
# Hourly, not daily, per an explicit "make it more granular" ask -- so the
# tab's 1D/1W/1M range buttons actually show intraday-ish detail instead of
# 1-2 daily points. Yahoo's free intraday feed allows 60m bars up to ~730
# days back; SECTOR_HISTORY_PERIOD only asks for 1y (~1750 bars/ticker,
# confirmed live), since that already covers the longest range button (1Y)
# exactly and roughly halves the fetch/payload size versus asking for the
# full 2y window this project uses for daily series elsewhere.
SECTOR_ETFS = [
    ("SPY", "Overall"),
    ("QQQ", "Tech"),
    ("SOXX", "Semiconductor"),
    ("IGV", "Software"),
    ("CIBR", "Cybersecurity"),
    ("XBI", "Biotech"),
    ("XLE", "Traditional Energy"),
    ("XLB", "Raw Material"),
    ("XLF", "Finance"),
]
SECTOR_HISTORY_PERIOD = "1y"
SECTOR_HISTORY_INTERVAL = "60m"

# Bump whenever compute_macro_signals()'s return shape changes -- same
# stale-cache guard as iv_rank.SCHEMA_VERSION, see that constant's note.
SCHEMA_VERSION = 10


def _directional_movement(high: pd.Series, low: pd.Series, close: pd.Series,
                          n: int = ADX_WINDOW) -> tuple[pd.Series | None, pd.Series | None, pd.Series | None]:
    """Wilder's-smoothing (adx_series, plus_di_series, minus_di_series) --
    ADX(n) plus its two directional components: +DI/-DI say whether rising
    momentum (+DM) or falling momentum (-DM) currently dominates, which is
    what both tiers' "+DI vs -DI" condition actually reads. Returns full
    Series (not just today's value) so callers can read off both today's ADX
    and its value N days ago (the tier-2 reversal score's ADX-momentum
    condition) from one calculation. Approximated the same way this
    project's existing RSI does (`.ewm(alpha=1/n, adjust=False)` in place of
    Wilder's classic simple-average seed) -- a standard, widely-used
    approximation, not a from-scratch indicator design. (None, None, None)
    with too little history."""
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
    adx_series = dx.ewm(alpha=1 / n, adjust=False).mean()
    return adx_series, plus_di_series, minus_di_series


def _at(series: pd.Series | None, offset: int = 0) -> float | None:
    """series.iloc[-1-offset] as a plain float -- offset=0 (default) is
    "today" (the most recent bar); offset=k reads the value as of k trading
    days before today, used to recompute past regimes for the Macro tab's
    "last N days" history strip (see _regime_history). None if `series` is
    None, too short, or the value itself is NaN."""
    if series is None or len(series) <= offset:
        return None
    v = series.iloc[-1 - offset]
    return float(v) if pd.notna(v) else None


def _change_over(series: pd.Series | None, lookback: int, offset: int = 0) -> float | None:
    """series[-1-offset] - series[-1-offset-lookback] (an absolute
    difference, not a ratio -- unlike _slope_200, ADX momentum is specified
    in ADX points, not percent). `offset` re-derives the same measure as of
    `offset` trading days before today, same as _at. None with too little
    history."""
    if series is None or len(series) < offset + lookback + 1:
        return None
    now, past = series.iloc[-1 - offset], series.iloc[-1 - offset - lookback]
    if pd.isna(now) or pd.isna(past):
        return None
    return float(now - past)


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


def _slope_200(sma200: pd.Series, lookback: int = SLOW_SCORE_SLOPE_LOOKBACK,
               offset: int = 0) -> float | None:
    """SMA200(t)/SMA200(t-lookback) - 1 -- positive when the 200-day average
    has been rising over the last `lookback` trading days (bullish), negative
    when falling (bearish). `offset` shifts "t" itself back `offset` trading
    days from today (see _at), used to recompute past regimes.

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
    if len(sma200) < offset + lookback + 1:
        return None
    now, past = sma200.iloc[-1 - offset], sma200.iloc[-1 - offset - lookback]
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


def _reversal_parts(price: float | None, sma20: float | None, sma50: float | None,
                    plus_di: float | None, minus_di: float | None,
                    adx_momentum: float | None) -> dict:
    """Tier-2 reversal: per-measure +1 (bullish-tilt) / -1 (bearish-tilt) / 0
    for the 5 measures behind both reversal counts -- the user's bullish and
    bearish condition lists are exact mirrors of the same 5 measures (Price
    vs SMA20, Price vs SMA50, SMA20 vs SMA50, +DI vs -DI, ADX momentum), so
    one bucket per measure serves both: _reversal_counts below just tallies
    which sign each bucket landed on. Unlike the slow score's ratio-based
    _bucket, these are plain strict inequalities with no percentage-band
    neutral zone -- except ADX momentum, which does have a real neutral band
    (-3..+3 ADX points) per the given thresholds."""
    def _cmp(a: float | None, b: float | None) -> int:
        if a is None or b is None:
            return 0
        if a > b:
            return 1
        if a < b:
            return -1
        return 0

    return {
        "price_vs_sma20": _cmp(price, sma20),
        "price_vs_sma50": _cmp(price, sma50),
        "sma20_vs_sma50": _cmp(sma20, sma50),
        "directional": _cmp(plus_di, minus_di),
        "adx_momentum": (0 if adx_momentum is None
                        else (1 if adx_momentum > REVERSAL_ADX_MOMENTUM_UP
                              else (-1 if adx_momentum < REVERSAL_ADX_MOMENTUM_DOWN else 0))),
    }


def _reversal_counts(parts: dict) -> tuple[int, int]:
    """(bullish-tilt count, bearish-tilt count), each 0-5 -- how many of the
    5 reversal measures landed on each side. A measure with no valid data
    (bucket 0) counts toward neither."""
    bullish = sum(1 for v in parts.values() if v > 0)
    bearish = sum(1 for v in parts.values() if v < 0)
    return bullish, bearish


def _classify_regime(slow_score: int | None, adx: float | None,
                     bullish_reversal_count: int | None = None,
                     bearish_reversal_count: int | None = None) -> str | None:
    """Strong Bull/Bull/Sideways/Bear/Strong Bear from the slow score + ADX
    (tier 1), or Bullish/Bearish Reversal (tier 2) when a reversal count hits
    its threshold against an opposing tier-1 score. Reversal is checked
    first and, when it fires, overrides the tier-1 label entirely -- so up
    to 7 distinct labels are possible in total, not just the 5 tier-1 bands.
    Within each tier, the named bands don't overlap (score can't be both
    >=3 and <=-3; ADX can't be both >=25 and <25; a bullish reversal needs
    slow_score<=-2 while a bearish one needs >=2, mutually exclusive), so
    order only matters *between* the two tiers -- "Sideways" is genuinely
    everything left over, including e.g. a score of +2 alongside a >=25 ADX
    (a real gap in the literal tier-1 rules: strong-trend-confirmed but the
    score itself isn't high enough for a Strong Bull), not just the "no
    score" middle ground."""
    if slow_score is None or adx is None:
        return None
    if (bullish_reversal_count is not None and bullish_reversal_count >= REVERSAL_COUNT_THRESHOLD
            and slow_score <= -2):
        return "Bullish Reversal"
    if (bearish_reversal_count is not None and bearish_reversal_count >= REVERSAL_COUNT_THRESHOLD
            and slow_score >= 2):
        return "Bearish Reversal"
    if slow_score >= 3 and adx >= REGIME_ADX_THRESHOLD:
        return "Strong Bull"
    if slow_score >= 2 and adx < REGIME_ADX_THRESHOLD:
        return "Bull"
    if slow_score <= -3 and adx >= REGIME_ADX_THRESHOLD:
        return "Strong Bear"
    if slow_score <= -2 and adx < REGIME_ADX_THRESHOLD:
        return "Bear"
    return "Sideways"


def _regime_at(offset: int, close: pd.Series, sma20_series: pd.Series, sma50_series: pd.Series,
               sma200_series: pd.Series, adx_series: pd.Series | None,
               plus_di_series: pd.Series | None, minus_di_series: pd.Series | None) -> str | None:
    """Recomputes the full tier-1 + tier-2 + regime classification as of
    `offset` trading days before today, by re-running the exact same
    functions compute_macro_signals uses for today against each series'
    value `offset` days back instead of always the latest bar. `offset=0` is
    today (identical to what compute_macro_signals computes inline)."""
    price = _at(close, offset)
    sma20 = _at(sma20_series, offset)
    sma50 = _at(sma50_series, offset)
    sma200 = _at(sma200_series, offset)
    adx = _at(adx_series, offset)
    plus_di = _at(plus_di_series, offset)
    minus_di = _at(minus_di_series, offset)

    slope_200 = _slope_200(sma200_series, offset=offset)
    adx_momentum = _change_over(adx_series, REVERSAL_ADX_MOMENTUM_LOOKBACK, offset=offset)

    slow_score, _ = _slow_score(price, sma50, sma200, slope_200, plus_di, minus_di)
    reversal_parts = _reversal_parts(price, sma20, sma50, plus_di, minus_di, adx_momentum)
    bullish_count, bearish_count = _reversal_counts(reversal_parts)
    return _classify_regime(slow_score, adx, bullish_count, bearish_count)


def _regime_history(close: pd.Series, sma20_series: pd.Series, sma50_series: pd.Series,
                    sma200_series: pd.Series, adx_series: pd.Series | None,
                    plus_di_series: pd.Series | None, minus_di_series: pd.Series | None,
                    days: int = REGIME_HISTORY_DAYS) -> list[dict]:
    """[{"date": ..., "regime": ...}, ...] for the `days` trading days before
    today, oldest first -- so a flip-flopping classification (e.g. Bull,
    Sideways, Bull, Bear across 4 days) is visible at a glance on the Macro
    tab rather than only ever showing today's possibly-noisy snapshot. Skips
    a day entirely (rather than emitting a None-regime entry) if there isn't
    enough history that far back yet."""
    out = []
    for offset in range(days, 0, -1):
        if len(close) <= offset:
            continue
        regime = _regime_at(offset, close, sma20_series, sma50_series, sma200_series,
                            adx_series, plus_di_series, minus_di_series)
        if regime is None:
            continue
        out.append({"date": str(close.index[-1 - offset].date()), "regime": regime})
    return out


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


def _sector_price_histories() -> list[dict]:
    """[{"ticker", "label", "history": [{"time", "value"}, ...], "latest"}, ...]
    for SECTOR_ETFS, in that fixed order -- hourly close prices (see this
    module's SECTOR_ETFS comment for why hourly, not daily), one batched
    `yf.download` call (same pattern as `_market_breadth` above), not a
    technical indicator. `time` is a Unix timestamp in seconds, not a
    'YYYY-MM-DD' string like every other history in this module -- Lightweight
    Charts' BusinessDay string format can only express whole days, and these
    bars need sub-day precision. A ticker with no usable history is simply
    omitted rather than failing the whole call."""
    import yfinance as yf

    tickers = [t for t, _ in SECTOR_ETFS]
    try:
        data = yf.download(tickers=tickers, period=SECTOR_HISTORY_PERIOD,
                           interval=SECTOR_HISTORY_INTERVAL, group_by="ticker",
                           auto_adjust=True, progress=False, threads=False)
    except Exception:
        data = None
    if data is None or data.empty:
        return []

    single = len(tickers) == 1
    out = []
    for ticker, label in SECTOR_ETFS:
        close = _extract_close(data, ticker, single)
        if close is None or close.empty:
            continue
        history = [{"time": int(d.timestamp()), "value": float(v)} for d, v in close.items()]
        out.append({"ticker": ticker, "label": label, "history": history, "latest": float(close.iloc[-1])})
    return out


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


def _fetch_fred_series(series_id: str) -> pd.Series | None:
    """A FRED series' full daily history as a date-indexed Series, via
    FRED's public `fredgraph.csv` endpoint -- unlike FRED's REST API, this
    needs no API key/signup, just a plain CSV download. Missing
    observations (FRED marks data gaps as ".") are dropped rather than
    interpolated."""
    import requests
    try:
        r = requests.get(FRED_CSV_URL.format(series_id=series_id), timeout=20)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = ["date", "value"]
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")["value"]
    except Exception:
        return None


def _fred_yield_history(series_id: str,
                        points: int = YIELD_HISTORY_POINTS) -> tuple[list[dict], float | None]:
    """(trailing `points` daily [{"date": ..., "value": ...}, ...], latest
    value) for a FRED yield series (DFII10 or DGS10 here) -- see this
    module's docstring and FRED_CSV_URL's comment for why these are fetched
    straight from FRED rather than embedded via TradingView. ([], None) if
    the fetch fails."""
    series = _fetch_fred_series(series_id)
    if series is None or series.empty:
        return [], None
    recent = series.tail(points)
    history = [{"date": str(d.date()), "value": float(v)} for d, v in recent.items()]
    return history, float(series.iloc[-1])


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

            adx_series, plus_di_series, minus_di_series = _directional_movement(
                hist["High"], hist["Low"], close)
            adx = _at(adx_series)
            plus_di = _at(plus_di_series)
            minus_di = _at(minus_di_series)
            if adx is not None:
                out["adx_14"] = adx
            if plus_di is not None:
                out["plus_di_14"] = plus_di
            if minus_di is not None:
                out["minus_di_14"] = minus_di

            # Slow score's (tier 1) own SMA50/SMA200, and the reversal
            # score's (tier 2) own SMA20 -- all deliberately separate from
            # fifty_dma/two_hundred_dma above, see this module's docstring.
            sma20_series = close.rolling(REVERSAL_SMA20_WINDOW).mean()
            sma50_series = close.rolling(SLOW_SCORE_SMA50_WINDOW).mean()
            sma200_series = close.rolling(SLOW_SCORE_SMA200_WINDOW).mean()
            sma20_calc = _at(sma20_series)
            sma50_calc = _at(sma50_series)
            sma200_calc = _at(sma200_series)
            if sma20_calc is not None:
                out["sma20_calc"] = sma20_calc
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

            adx_momentum_10 = _change_over(adx_series, REVERSAL_ADX_MOMENTUM_LOOKBACK)
            if adx_momentum_10 is not None:
                out["adx_momentum_10"] = adx_momentum_10

            reversal_parts = _reversal_parts(
                price, sma20_calc, sma50_calc, plus_di, minus_di, adx_momentum_10)
            bullish_reversal_count, bearish_reversal_count = _reversal_counts(reversal_parts)
            out["reversal_parts"] = reversal_parts
            out["bullish_reversal_count"] = bullish_reversal_count
            out["bearish_reversal_count"] = bearish_reversal_count

            regime = _classify_regime(slow_score, adx, bullish_reversal_count, bearish_reversal_count)
            if regime is not None:
                out["market_regime"] = regime

            out["regime_history"] = _regime_history(
                close, sma20_series, sma50_series, sma200_series,
                adx_series, plus_di_series, minus_di_series)
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

    try:
        real_yield_history, real_yield_latest = _fred_yield_history(REAL_YIELD_SERIES_ID)
        if real_yield_history:
            out["real_yield_history"] = real_yield_history
        if real_yield_latest is not None:
            out["real_yield_latest"] = real_yield_latest
    except Exception:
        real_yield_latest = None

    try:
        nominal_yield_history, nominal_yield_latest = _fred_yield_history(NOMINAL_YIELD_SERIES_ID)
        if nominal_yield_history:
            out["nominal_yield_history"] = nominal_yield_history
        if nominal_yield_latest is not None:
            out["nominal_yield_latest"] = nominal_yield_latest
    except Exception:
        nominal_yield_latest = None

    # Breakeven inflation = nominal - real -- the bond market's own implied
    # inflation expectation over the next 10 years. Plain arithmetic on the
    # two already-fetched latest values, not a technical indicator.
    if real_yield_latest is not None and nominal_yield_latest is not None:
        out["breakeven_inflation"] = nominal_yield_latest - real_yield_latest

    try:
        fed_funds_history, fed_funds_latest = _fred_yield_history(FED_FUNDS_SERIES_ID)
        if fed_funds_history:
            out["fed_funds_history"] = fed_funds_history
        if fed_funds_latest is not None:
            out["fed_funds_latest"] = fed_funds_latest
    except Exception:
        pass

    try:
        sector_charts = _sector_price_histories()
        if sector_charts:
            out["sector_charts"] = sector_charts
    except Exception:
        pass

    return out
