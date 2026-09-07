from __future__ import annotations

import numpy as np
import pandas as pd

from ..engine import MIN_TICKET, Chain, WheelAccount


def csp_signal(cfg, frow: pd.Series, spot: float) -> float:
    if cfg.csp_signal == "abs":
        return spot
    return float(frow["pct_1y" if cfg.csp_signal == "pct_1y" else "pct_3y"])


def csp_ladder_target(signal_value: float, edges, tranches) -> float:
    if signal_value is None or np.isnan(signal_value):
        return 0.0
    target = 0.0
    for edge, tranche in zip(edges, tranches):
        if signal_value < edge:
            target = tranche
    return target


def _iv_ok(cfg, frow: pd.Series) -> bool:
    if cfg.iv_pct_min is not None:
        v = frow.get("iv_pct_1y", np.nan)
        if np.isnan(v) or v < cfg.iv_pct_min:
            return False
    if cfg.iv_minus_rv_min is not None:
        v = frow.get("iv_minus_rv20", np.nan)
        if np.isnan(v) or v < cfg.iv_minus_rv_min:
            return False
    return True


class CashSecuredPutLeg:
    def __init__(self, cfg):
        self.cfg = cfg

    def rebalance(self, acct: WheelAccount, chain: Chain, date, spot: float, frow: pd.Series) -> None:
        cfg = self.cfg
        signal = csp_signal(cfg, frow, spot)

        reentry = cfg.csp_edges[0] if cfg.csp_signal == "abs" else cfg.csp_reentry_edge
        if acct.waiting_for_dip and signal < reentry:
            acct.waiting_for_dip = False
        if acct.waiting_for_dip or not chain.has(date) or not _iv_ok(cfg, frow):
            return

        target_cap = (csp_ladder_target(signal, cfg.csp_edges, cfg.csp_tranches)
                      * cfg.max_alloc_pct * cfg.portfolio_size)
        gap = target_cap - (acct.put_collateral + acct.stock_cost)
        free_cash = acct.cash - acct.put_collateral
        dtes = cfg.csp_dte_ladder or (cfg.csp_dte,)
        per_leg = gap / len(dtes)
        for dte in dtes:
            if per_leg < MIN_TICKET or free_cash < MIN_TICKET:
                break
            q = chain.select(date, "P", dte, cfg.csp_delta, cfg.dte_tol, max_strike=spot * 1.05)
            if q is None:
                continue
            collateral = q["strike"] * 100
            n = int(min(per_leg // collateral, free_cash // collateral))
            if n >= 1:
                acct.sell(q, n, date, spot, signal, profit_take=cfg.csp_profit_take)
                free_cash -= collateral * n
