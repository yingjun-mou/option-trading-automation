from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .datasource import MarketData
from .engine import Chain
from .metrics import compute_metrics
from .wheel import WheelConfig, run_backtest

_SUMMARY_KEYS = ("premium_csp", "premium_cc", "realized_stock_pnl", "n_csp", "n_cc",
                 "n_assign", "n_callaway", "pct_days_holding_shares", "pct_days_idle",
                 "avg_capital_utilization", "worst_single_cashflow")


@dataclass(frozen=True)
class Window:
    name: str
    start: str | None
    end: str | None


TRAIN = Window("train", "2016-01-01", "2021-12-31")
VALID = Window("valid", "2022-01-01", "2023-12-31")
TEST = Window("test", "2024-01-01", None)
FULL = Window("full", "2016-01-01", None)

REGIMES = [
    Window("pre_covid", "2016-01-01", "2020-02-19"),
    Window("covid", "2020-02-20", "2021-03-31"),
    Window("post_covid", "2021-04-01", None),
]


class Objective:
    def __init__(self, data: MarketData, features: pd.DataFrame, chain: Chain | None = None):
        self.options = data.options
        self.features = features
        self.chain = chain or Chain(data.options)

    def backtest(self, cfg: WheelConfig, window: Window = FULL):
        return run_backtest(self.features, self.options, cfg,
                            window.start, window.end, self.chain)

    def __call__(self, cfg: WheelConfig, window: Window = FULL) -> dict:
        result = self.backtest(cfg, window)
        metrics = compute_metrics(result.equity)
        metrics.update({k: result.summary[k] for k in _SUMMARY_KEYS})
        metrics["label"] = cfg.label
        metrics["window"] = window.name
        return metrics


def walk_forward(objective: Objective, configs: list[WheelConfig],
                 windows=(TRAIN, VALID, TEST)) -> pd.DataFrame:
    rows = []
    for cfg in configs:
        row = {"label": cfg.label}
        for w in windows:
            m = objective(cfg, w)
            row[f"{w.name}_cagr"] = m["cagr"]
            row[f"{w.name}_calmar"] = m["calmar"]
            row[f"{w.name}_maxdd"] = m["max_drawdown"]
        rows.append(row)
    return pd.DataFrame(rows)


def regime_report(objective: Objective, cfg: WheelConfig) -> pd.DataFrame:
    keys = ("cagr", "total_return", "max_drawdown", "sharpe", "calmar", "n_csp", "n_cc",
            "n_assign", "n_callaway", "pct_days_holding_shares", "pct_days_idle")
    rows = [dict(regime=w.name, **{k: objective(cfg, w)[k] for k in keys}) for w in REGIMES]
    return pd.DataFrame(rows)
