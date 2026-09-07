from __future__ import annotations

import numpy as np
import pandas as pd

from .datasource import MarketData

YEAR = 252


def _rolling_percentile(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    return s.rolling(window, min_periods=min_periods).apply(lambda x: (x[-1] >= x).mean(), raw=True)


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(50)


def _regime(idx: pd.DatetimeIndex) -> pd.Series:
    out = pd.Series("post_covid", index=idx, dtype=object)
    out[idx < pd.Timestamp("2020-02-20")] = "pre_covid"
    out[(idx >= pd.Timestamp("2020-02-20")) & (idx < pd.Timestamp("2021-04-01"))] = "covid"
    return out


def _atm_iv(options: pd.DataFrame) -> pd.DataFrame:
    o = options[options["dte"].between(20, 45)].copy()
    o["dte_gap"] = (o["dte"] - 30).abs()
    o["mny"] = (o["strike"] / o["underlying"] - 1.0).abs()
    o = o.sort_values(["date", "dte_gap", "mny"]).groupby("date").first()
    return o[["iv"]].rename(columns={"iv": "iv_30"})


def build_features(data: MarketData) -> pd.DataFrame:
    stock, options = data.stock, data.options
    c = stock["close"]
    f = pd.DataFrame(index=stock.index)
    f["close"] = c
    f["volume"] = stock["volume"]

    f["pct_1y"] = _rolling_percentile(c, YEAR, 60)
    f["pct_3y"] = _rolling_percentile(c, 3 * YEAR, 120)
    f["roll_low_3y"] = c.rolling(3 * YEAR, min_periods=120).min()
    f["roll_high_3y"] = c.rolling(3 * YEAR, min_periods=120).max()
    f["dist_from_low_3y"] = c / f["roll_low_3y"] - 1.0
    f["dist_from_high_3y"] = c / f["roll_high_3y"] - 1.0

    f["ret_20"] = c.pct_change(20)
    f["ret_60"] = c.pct_change(60)
    f["rsi_14"] = _rsi(c)
    f["dist_ma50"] = c / c.rolling(50).mean() - 1.0
    f["dist_ma200"] = c / c.rolling(200).mean() - 1.0

    logret = np.log(c).diff()
    f["rv_20"] = logret.rolling(20).std() * np.sqrt(YEAR)
    f["rv_30"] = logret.rolling(30).std() * np.sqrt(YEAR)

    if len(options):
        f = f.join(_atm_iv(options))
        f["iv_30"] = f["iv_30"].ffill()
        f["iv_pct_1y"] = _rolling_percentile(f["iv_30"], YEAR, 60)
        f["iv_minus_rv20"] = f["iv_30"] - f["rv_20"]
    else:
        for col in ("iv_30", "iv_pct_1y", "iv_minus_rv20"):
            f[col] = np.nan

    f["regime"] = _regime(f.index)
    return f
