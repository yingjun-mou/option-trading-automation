"""Live cross-sectional ATM option-premium scan (the dashboard's Premium
Scanner tab). `scan_universe()` fetches one row per ticker -- spot, the
nearest-ATM strike at a near-monthly expiry, and the covered-call/cash-
secured-put "reward %" (premium as a fraction of the capital each leg ties
up). See the README's "Premium scanner" section for the full picture;
`scanner_job.ScannerJob` owns the background cadence and disk cache."""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

import pandas as pd

from .realtime import REALTIME_DIR

UNIVERSE_FILE = REALTIME_DIR / "universe.json"
DTE_RANGE = (25, 45)          # target expiry window: near-monthly, matches the wheel's own tenor
# yfinance zeroes bid/ask outside regular trading hours (confirmed: AAPL/TSLA show
# bid=ask=0 overnight even though they traded seconds before the close). A recent
# lastPrice is a reasonable stand-in then -- but only if it is actually recent;
# INIO's lastPrice was a real trade, just from 9 days earlier at a different spot.
# So: use lastPrice as a "closing price" fallback within this window, otherwise none.
MAX_CLOSE_STALENESS_DAYS = 4

# Yahoo's unofficial quote/options endpoint rate-limits (HTTP 429) hard and fast --
# testing this module with even modest concurrency (4 workers) got an IP-wide block
# within ~150 tickers. There is no auth/paid tier to raise the limit, so the scanner
# must pace itself gently rather than race to finish: fully sequential, one ticker
# at a time, with a fixed gap between requests. ~850 tickers * 0.6s ~= 8-9 minutes,
# comfortably inside the 15-minute refresh cycle.
REQUEST_GAP = 0.6
RETRY_ON_429 = 1
RETRY_BACKOFF = 15.0

RESULT_COLUMNS = [
    "symbol", "spot", "expiration", "dte", "atm_strike",
    "call_bid", "call_ask", "call_oi", "call_volume",
    "put_bid", "put_ask", "put_oi", "put_volume",
    "cc_premium", "csp_premium",
    "cc_reward_pct", "cc_reward_pct_annualized",
    "csp_reward_pct", "csp_reward_pct_annualized",
    "cc_live", "csp_live",
    "iv_rank_pct", "price_pct",
    "error",
]


def load_universe(path: Path = UNIVERSE_FILE) -> list[str]:
    """Ticker list to scan, as built by `scripts/build_universe.py`."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python scripts/build_universe.py` first")
    return json.loads(path.read_text())["tickers"]


def _pick_expiration(expirations: tuple[str, ...], today: date,
                     dte_range: tuple[int, int] = DTE_RANGE) -> tuple[str, int] | None:
    """(expiration, dte) closest to the midpoint of `dte_range`, preferring
    an expiration actually inside that range; None if none are in the future."""
    cands = [(e, (date.fromisoformat(e) - today).days) for e in expirations]
    cands = [c for c in cands if c[1] > 0]
    if not cands:
        return None
    mid = sum(dte_range) / 2
    in_range = [c for c in cands if dte_range[0] <= c[1] <= dte_range[1]]
    return min(in_range or cands, key=lambda c: abs(c[1] - mid))


def _safe_num(x, cast, default=0):
    """cast(x) but treating NaN/None as `default` -- yfinance leaves some
    columns (e.g. `volume`) as NaN rather than 0, and `nan or 0` is still
    `nan` in Python since NaN is truthy."""
    try:
        return default if x is None or pd.isna(x) else cast(x)
    except (TypeError, ValueError):
        return default


def _safe_int(x) -> int:
    return _safe_num(x, int, 0)


def _safe_float(x) -> float:
    return _safe_num(x, float, 0.0)


def _premium(row: pd.Series, now: pd.Timestamp) -> tuple[float | None, bool]:
    """Returns (premium, is_live_bid).

    Prefers the live bid -- what a seller would actually receive right now.
    With no bid (common outside market hours, or a genuinely thin contract),
    falls back to the last trade price, but ONLY if that trade is recent; a
    stale lastPrice can sit at a wildly different underlying price (confirmed
    on INIO: lastPrice=$7 from a trade 9 days earlier -- bid=ask=0 -- while
    the real market was ~$0.35) and reporting that as today's premium is
    simply wrong, not just optimistic."""
    bid = _safe_float(row.get("bid"))
    if bid > 0:
        return bid, True
    last_price = _safe_float(row.get("lastPrice"))
    last_trade = row.get("lastTradeDate")
    if last_price > 0 and last_trade is not None and not pd.isna(last_trade):
        ts = pd.Timestamp(last_trade)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        age_days = (now - ts).total_seconds() / 86400
        if 0 <= age_days <= MAX_CLOSE_STALENESS_DAYS:
            return last_price, False
    return None, False


def _is_rate_limited(exc: Exception) -> bool:
    """True if `exc` looks like Yahoo's HTTP 429 rather than some other failure."""
    msg = str(exc).lower()
    return "429" in msg or "too many requests" in msg or "rate limit" in msg


def _leg_quote(row: pd.Series) -> dict:
    """NaN-safe bid/ask/open-interest/volume for one option leg."""
    return dict(bid=_safe_float(row.get("bid")), ask=_safe_float(row.get("ask")),
                oi=_safe_int(row.get("openInterest")), volume=_safe_int(row.get("volume")))


def _reward_pct(premium: float | None, denominator: float,
                annualize: float) -> tuple[float | None, float | None]:
    """(reward_pct, reward_pct_annualized) = premium / denominator, or (None,
    None) with no premium. `denominator` is spot for CC, strike for CSP --
    the 100-share/collateral contract multiplier cancels out of the ratio
    either way, so callers pass per-share values throughout."""
    if premium is None:
        return None, None
    pct = premium / denominator
    return round(pct, 5), round(pct * annualize, 4)


def _scan_one(symbol: str, dte_range: tuple[int, int] = DTE_RANGE) -> dict:
    """One ticker's scan row: `{"symbol": ..., "error": None, ...}` on
    success, or `{"symbol": ..., "error": "<reason>"}` on failure -- a bad
    ticker returns an error row rather than raising, so `scan_universe` can
    keep going. Retries once on a 429 (see REQUEST_GAP note above) before
    giving up."""
    import yfinance as yf

    attempt = 0
    while True:
        try:
            t = yf.Ticker(symbol)
            spot = float(t.fast_info.get("lastPrice") or 0)
            if spot <= 0:
                return dict(symbol=symbol, error="no price")

            picked = _pick_expiration(t.options, date.today(), dte_range)
            if picked is None:
                return dict(symbol=symbol, error="no listed options")
            expiration, dte = picked

            chain = t.option_chain(expiration)
            calls, puts = chain.calls, chain.puts
            if calls.empty or puts.empty:
                return dict(symbol=symbol, error="empty chain")

            atm_strike = float(calls.loc[(calls["strike"] - spot).abs().idxmin(), "strike"])
            crow = calls.loc[(calls["strike"] - atm_strike).abs().idxmin()]
            prow = puts.loc[(puts["strike"] - atm_strike).abs().idxmin()]
            cq, pq = _leg_quote(crow), _leg_quote(prow)

            now = pd.Timestamp.now("UTC")
            cc_premium, cc_live = _premium(crow, now)
            csp_premium, csp_live = _premium(prow, now)
            annualize = 365 / dte
            cc_reward_pct, cc_reward_pct_ann = _reward_pct(cc_premium, spot, annualize)
            csp_reward_pct, csp_reward_pct_ann = _reward_pct(csp_premium, atm_strike, annualize)

            return dict(
                symbol=symbol, spot=round(spot, 2), expiration=expiration, dte=dte,
                atm_strike=atm_strike,
                call_bid=cq["bid"], call_ask=cq["ask"], call_oi=cq["oi"], call_volume=cq["volume"],
                put_bid=pq["bid"], put_ask=pq["ask"], put_oi=pq["oi"], put_volume=pq["volume"],
                cc_premium=round(cc_premium, 3) if cc_premium is not None else None,
                csp_premium=round(csp_premium, 3) if csp_premium is not None else None,
                cc_reward_pct=cc_reward_pct, cc_reward_pct_annualized=cc_reward_pct_ann,
                csp_reward_pct=csp_reward_pct, csp_reward_pct_annualized=csp_reward_pct_ann,
                cc_live=cc_live, csp_live=csp_live,
                # both filled in by ScannerJob from the (separately cached) IvRankJob signals:
                iv_rank_pct=None,
                price_pct=None,  # live spot (above) ranked against recent daily closes
                error=None,
            )
        except Exception as e:  # noqa: BLE001 -- one bad ticker must not sink the scan
            if _is_rate_limited(e) and attempt < RETRY_ON_429:
                attempt += 1
                time.sleep(RETRY_BACKOFF * attempt)
                continue
            return dict(symbol=symbol, error=str(e))


def scan_universe(symbols: list[str], dte_range: tuple[int, int] = DTE_RANGE,
                  request_gap: float = REQUEST_GAP, on_progress=None) -> pd.DataFrame:
    """Sequential by design -- see the REQUEST_GAP note above. Do not add
    concurrency back without re-verifying against Yahoo's rate limit."""
    rows = []
    n = len(symbols)
    for i, s in enumerate(symbols, 1):
        rows.append(_scan_one(s, dte_range))
        if on_progress:
            on_progress(i, n)
        if request_gap and i < n:
            time.sleep(request_gap)
    df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    return df.sort_values("symbol").reset_index(drop=True)
