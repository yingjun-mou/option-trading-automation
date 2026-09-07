from __future__ import annotations

import pandas as pd

from .wheel import WheelConfig


def buy_and_hold_curve(features: pd.DataFrame, portfolio_size=100_000.0,
                       start=None, end=None) -> pd.Series:
    px = features["close"]
    if start:
        px = px.loc[pd.Timestamp(start):]
    if end:
        px = px.loc[:pd.Timestamp(end)]
    return pd.Series(portfolio_size / float(px.iloc[0]) * px.values, index=px.index, name="equity")


def mechanical_wheel(dte: int, delta: float = 0.20, portfolio_size=100_000.0) -> WheelConfig:
    return WheelConfig(
        portfolio_size=portfolio_size, max_alloc_pct=1.0,
        csp_signal="pct_3y", csp_edges=(1.01,), csp_tranches=(1.0,), csp_reentry_edge=1.01,
        csp_dte=dte, csp_delta=delta, csp_profit_take=1.0,
        cc_signal="pct_3y", cc_edges=(0.0,), cc_tranches=(1.0,),
        cc_dte=dte, cc_delta=delta, cc_callaway_delta=delta,
        label=f"mechanical_wheel_{dte}dte_{int(delta * 100)}delta",
    )


def price_wheel_variants(optimized: WheelConfig) -> dict[str, WheelConfig]:
    no_iv = optimized.copy(iv_pct_min=None, iv_minus_rv_min=None, label="price_wheel_no_iv")
    with_iv = optimized.copy(iv_pct_min=optimized.iv_pct_min or 0.50, label="price_wheel_with_iv")
    return {"price_wheel_no_iv": no_iv, "price_wheel_with_iv": with_iv}
