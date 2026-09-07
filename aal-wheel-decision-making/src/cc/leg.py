from __future__ import annotations

import numpy as np
import pandas as pd

from ..engine import Chain, WheelAccount


def cc_signal(cfg, frow: pd.Series, spot: float) -> float:
    return spot if cfg.cc_signal == "abs" else float(frow["pct_3y"])


def cc_ladder_target(value: float, edges, tranches) -> float:
    if value is None or np.isnan(value):
        return 0.0
    target = 0.0
    for edge, tranche in zip(edges, tranches):
        if value >= edge:
            target = tranche
    return target


class CoveredCallLeg:
    def __init__(self, cfg):
        self.cfg = cfg

    def rebalance(self, acct: WheelAccount, chain: Chain, date, spot: float, frow: pd.Series) -> None:
        cfg = self.cfg
        if acct.shares == 0 and cfg.cc_rebuy_when_flat:
            qty = int(cfg.max_alloc_pct * cfg.portfolio_size // (spot * 100)) * 100
            if qty and acct.cash >= qty * spot:
                acct.buy_shares(qty, spot)
        if acct.shares < 100 or not chain.has(date):
            return
        signal = cc_signal(cfg, frow, spot)
        callaway = signal >= cfg.cc_edges[-1]

        want = int(cc_ladder_target(signal, cfg.cc_edges, cfg.cc_tranches) * acct.shares // 100) * 100
        add = (want - acct.covered_shares) // 100
        if add < 1:
            return

        min_strike = acct.effective_basis if (cfg.cc_min_strike_vs_basis and not callaway) else None
        delta = cfg.cc_callaway_delta if callaway else cfg.cc_delta
        q = chain.select(date, "C", cfg.cc_dte, delta, cfg.dte_tol, min_strike=min_strike)
        if q is not None:
            acct.sell(q, add, date, spot, signal, profit_take=cfg.cc_profit_take)
