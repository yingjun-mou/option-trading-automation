from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from .datasource import STOCK_CSV, _read_stock_csv
from .pricing import bs_delta, bs_price

REALTIME_DIR = Path(__file__).resolve().parent.parent / "realtime_data"
MOCK_DIR = REALTIME_DIR / "mock"
# ROBINHOOD_DIR = REALTIME_DIR / "robinhood_api"   # TODO: real feed

CHAIN_COLUMNS = ["date", "expiration", "dte", "strike", "opt_type",
                 "bid", "ask", "mid", "iv", "delta", "underlying"]


@dataclass
class Quote:
    symbol: str
    price: float
    bid: float
    ask: float
    prev_close: float
    timestamp: datetime
    source: str


class RealtimeSource(Protocol):
    def quote(self) -> Quote: ...
    def option_chain(self) -> pd.DataFrame: ...


def _synthetic_chain(spot: float, atm_iv: float, as_of: datetime, r: float = 0.04,
                     skew: float = 0.45, smile: float = 0.6) -> pd.DataFrame:
    today = pd.Timestamp(as_of.date())
    fridays = pd.date_range(today + pd.Timedelta(days=1), today + pd.Timedelta(days=75), freq="W-FRI")
    rows = []
    for exp in fridays:
        dte = (exp - today).days
        if dte < 3:
            continue
        t = dte / 365.0
        strikes = np.arange(np.floor(spot * 0.70), np.ceil(spot * 1.30) + 0.5, 0.5)
        m = np.log(strikes / spot)
        iv_k = np.clip(atm_iv - skew * m + smile * m ** 2, 0.10, 3.0) * (1 + 0.05 * math.sqrt(t))
        for side in ("C", "P"):
            px = bs_price(spot, strikes, t, r, iv_k, side)
            dl = np.abs(bs_delta(spot, strikes, t, r, iv_k, side))
            half = np.maximum(0.02, 0.05 * px + 0.02)
            for j in range(len(strikes)):
                if not (0.03 < dl[j] < 0.97):
                    continue
                rows.append((today, exp, dte, float(strikes[j]), side,
                             round(max(0.01, float(px[j] - half[j])), 2),
                             round(float(px[j] + half[j]), 2), round(float(px[j]), 3),
                             round(float(iv_k[j]), 4), round(float(dl[j]), 4), round(spot, 2)))
    return pd.DataFrame(rows, columns=CHAIN_COLUMNS)


class MockRealtime:
    """Deterministic-per-timestamp fake feed built from the cached historical close.

    Price wanders within a few percent of the last historical bar so the dashboard
    visibly ticks. Swap this class for a RobinhoodRealtime later; nothing else changes.
    """

    def __init__(self, symbol: str = "AAL", history_csv: Path = STOCK_CSV, seed: int = 7):
        close = _read_stock_csv(history_csv)["close"]
        self.symbol = symbol
        self._base = float(close.iloc[-1])
        self._prev_close = float(close.iloc[-2])
        rv20 = float(np.log(close).diff().tail(20).std() * np.sqrt(252))
        self.atm_iv = float(np.clip(rv20 * 1.15, 0.20, 1.2))
        self._seed = seed

    def _price_at(self, ts: datetime) -> float:
        t = ts.timestamp()
        drift = 0.015 * math.sin(t / 3600.0)
        wiggle = 0.004 * math.sin(t / 45.0)
        noise = 0.002 * ((hash((int(t // 15), self._seed)) % 1000) / 500.0 - 1.0)
        return round(self._base * (1 + drift + wiggle + noise), 2)

    def quote(self) -> Quote:
        now = datetime.now(timezone.utc)
        px = self._price_at(now)
        spread = max(0.01, round(px * 0.0005, 2))
        return Quote(self.symbol, px, round(px - spread, 2), round(px + spread, 2),
                     self._prev_close, now, "mock")

    def option_chain(self) -> pd.DataFrame:
        q = self.quote()
        return _synthetic_chain(q.price, self.atm_iv, q.timestamp)

    def recent_closes(self, n: int = 120) -> pd.Series:
        return _read_stock_csv(STOCK_CSV)["close"].tail(n)


def default_realtime_source() -> RealtimeSource:
    # TODO: if realtime_data/robinhood_api credentials are present, return RobinhoodRealtime()
    return MockRealtime()
