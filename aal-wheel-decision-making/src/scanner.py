from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

import pandas as pd

from .realtime import REALTIME_DIR

UNIVERSE_FILE = REALTIME_DIR / "universe.json"
DTE_RANGE = (25, 45)          # target expiry window: near-monthly, matches the wheel's own tenor

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
    "call_bid", "call_ask", "put_bid", "put_ask",
    "cc_premium", "csp_premium",
    "cc_reward_pct", "cc_reward_pct_annualized",
    "csp_reward_pct", "csp_reward_pct_annualized",
    "cc_thin", "csp_thin",
    "error",
]


def load_universe(path: Path = UNIVERSE_FILE) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python scripts/build_universe.py` first")
    return json.loads(path.read_text())["tickers"]


def _pick_expiration(expirations: tuple[str, ...], today: date,
                     dte_range: tuple[int, int] = DTE_RANGE) -> tuple[str, int] | None:
    cands = [(e, (date.fromisoformat(e) - today).days) for e in expirations]
    cands = [c for c in cands if c[1] > 0]
    if not cands:
        return None
    mid = sum(dte_range) / 2
    in_range = [c for c in cands if dte_range[0] <= c[1] <= dte_range[1]]
    return min(in_range or cands, key=lambda c: abs(c[1] - mid))


def _premium(row: pd.Series) -> float:
    """What a seller would actually receive: the bid, falling back to a
    conservative half-spread estimate or last trade when no bid is quoted."""
    bid, ask = float(row.get("bid") or 0), float(row.get("ask") or 0)
    if bid > 0:
        return bid
    if ask > 0:
        return ask / 2
    return float(row.get("lastPrice") or 0)


def _is_rate_limited(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "too many requests" in msg or "rate limit" in msg


def _scan_one(symbol: str, dte_range: tuple[int, int] = DTE_RANGE) -> dict:
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

            cc_premium, csp_premium = _premium(crow), _premium(prow)
            # Contract multiplier (100 shares/collateral) cancels in both ratios.
            cc_reward = cc_premium / spot
            csp_reward = csp_premium / atm_strike
            annualize = 365 / dte
            call_oi = float(crow.get("openInterest") or 0) + float(crow.get("volume") or 0)
            put_oi = float(prow.get("openInterest") or 0) + float(prow.get("volume") or 0)

            return dict(
                symbol=symbol, spot=round(spot, 2), expiration=expiration, dte=dte,
                atm_strike=atm_strike,
                call_bid=float(crow.get("bid") or 0), call_ask=float(crow.get("ask") or 0),
                put_bid=float(prow.get("bid") or 0), put_ask=float(prow.get("ask") or 0),
                cc_premium=round(cc_premium, 3), csp_premium=round(csp_premium, 3),
                cc_reward_pct=round(cc_reward, 5),
                cc_reward_pct_annualized=round(cc_reward * annualize, 4),
                csp_reward_pct=round(csp_reward, 5),
                csp_reward_pct_annualized=round(csp_reward * annualize, 4),
                # No open interest AND no trades today -- the quote is likely stale/wide
                # and the reward% for that leg should not be trusted at face value.
                cc_thin=call_oi <= 0, csp_thin=put_oi <= 0,
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
