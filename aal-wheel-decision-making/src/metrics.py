from __future__ import annotations

import numpy as np
import pandas as pd

YEAR = 252


def _cagr(equity: pd.Series) -> float:
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0 or equity.iloc[0] <= 0:
        return np.nan
    return (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1


def _drawdown(equity: pd.Series):
    peak = equity.cummax()
    dd = equity / peak - 1.0
    underwater = equity < peak
    longest = current = 0
    start = None
    for date, wet in underwater.items():
        if wet:
            start = start or date
            current = (date - start).days
            longest = max(longest, current)
        else:
            start, current = None, 0
    trough = dd.idxmin()
    return float(dd.min()), str(trough.date()) if isinstance(trough, pd.Timestamp) else None, longest


def compute_metrics(equity: pd.Series, rf_annual: float = 0.02) -> dict:
    equity = equity.dropna()
    if len(equity) < 5:
        return {"error": "equity curve too short", "cagr": np.nan, "calmar": np.nan}

    rets = equity.pct_change().dropna()
    excess = rets - rf_annual / YEAR
    downside = rets[rets < 0].std() * np.sqrt(YEAR)
    cagr = _cagr(equity)
    max_dd, dd_date, longest_dd = _drawdown(equity)

    yearly = equity.resample("YE").last()
    yearly_ret = pd.concat([pd.Series({yearly.index[0]: yearly.iloc[0] / equity.iloc[0] - 1}),
                            yearly.pct_change().dropna()]).sort_index()

    return dict(
        start=str(equity.index[0].date()), end=str(equity.index[-1].date()),
        start_value=float(equity.iloc[0]), end_value=float(equity.iloc[-1]),
        total_return=float(equity.iloc[-1] / equity.iloc[0] - 1),
        cagr=float(cagr),
        ann_vol=float(rets.std() * np.sqrt(YEAR)),
        sharpe=float(excess.mean() * YEAR / (rets.std() * np.sqrt(YEAR))) if rets.std() else np.nan,
        sortino=float(excess.mean() * YEAR / downside) if downside else np.nan,
        max_drawdown=max_dd, max_drawdown_date=dd_date, longest_drawdown_days=int(longest_dd),
        calmar=float(cagr / abs(max_dd)) if max_dd < 0 else np.nan,
        best_year=float(yearly_ret.max()), worst_year=float(yearly_ret.min()),
        yearly_returns={str(k.year): float(v) for k, v in yearly_ret.items()},
    )
