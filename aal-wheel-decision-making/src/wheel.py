from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from .cc import CoveredCallLeg
from .csp import CashSecuredPutLeg
from .engine import BacktestResult, Chain, Leg, run_strategy


@dataclass
class WheelConfig:
    portfolio_size: float = 100_000.0
    max_alloc_pct: float = 1.0

    enable_csp: bool = True
    enable_cc: bool = True
    initial_alloc: float = 0.0            # shares bought on day 1 (for a CC-only study)

    csp_signal: str = "pct_3y"            # "pct_3y" | "pct_1y" | "abs"
    csp_edges: tuple = (0.50, 0.30, 0.20, 0.10)
    csp_tranches: tuple = (0.25, 0.50, 0.75, 1.00)
    csp_reentry_edge: float = 0.50
    csp_dte: int = 21
    csp_delta: float = 0.20
    csp_profit_take: float = 1.00
    csp_dte_ladder: tuple = ()

    iv_pct_min: float | None = None
    iv_minus_rv_min: float | None = None

    cc_signal: str = "abs"               # "abs" | "pct_3y"
    cc_edges: tuple = (12.0, 14.0, 16.0, 18.0)
    cc_tranches: tuple = (0.0, 0.34, 0.67, 1.00)
    cc_dte: int = 30
    cc_delta: float = 0.25
    cc_callaway_delta: float = 0.40
    cc_min_strike_vs_basis: bool = True
    cc_profit_take: float = 1.00
    cc_rebuy_when_flat: bool = False      # standalone CC overlay: re-buy shares after call-away

    sell_fill: str = "mid"
    buy_fill: str = "mid"
    slippage: float = 0.02
    commission_per_contract: float = 0.65
    dte_tol: int = 6
    label: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def copy(self, **changes) -> "WheelConfig":
        return WheelConfig(**{**self.to_dict(), **changes})


def build_legs(cfg: WheelConfig) -> list[Leg]:
    legs: list[Leg] = []
    if cfg.enable_csp:
        legs.append(CashSecuredPutLeg(cfg))
    if cfg.enable_cc:
        legs.append(CoveredCallLeg(cfg))
    return legs


def run_backtest(features: pd.DataFrame, options: pd.DataFrame, cfg: WheelConfig,
                 start=None, end=None, chain: Chain | None = None) -> BacktestResult:
    return run_strategy(build_legs(cfg), features, options, cfg, start, end, chain,
                        initial_alloc=cfg.initial_alloc)
