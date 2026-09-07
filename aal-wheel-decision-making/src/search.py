from __future__ import annotations

import time
from typing import Protocol, Sequence

import pandas as pd

from .evaluation import FULL, VALID, Objective, Window
from .wheel import WheelConfig

_METRIC_KEYS = ("cagr", "total_return", "max_drawdown", "sharpe", "sortino", "calmar",
                "worst_year", "best_year", "end_value", "premium_csp", "premium_cc",
                "realized_stock_pnl", "n_csp", "n_cc", "n_assign", "n_callaway",
                "pct_days_holding_shares", "pct_days_idle", "avg_capital_utilization")

_TUPLE_FIELDS = ("csp_edges", "csp_tranches", "cc_edges", "cc_tranches", "csp_dte_ladder")


def result_row(cfg: WheelConfig, metrics: dict) -> dict:
    row = cfg.to_dict()
    for field in _TUPLE_FIELDS:
        row[field] = str(row[field])
    row["tranche_shape"] = "increasing" if "increasing" in cfg.label else "equal"
    row["iv_filter"] = "on" if cfg.iv_pct_min else "off"
    row.update({f"m_{k}": metrics.get(k) for k in _METRIC_KEYS})
    return row


class Searcher(Protocol):
    def propose(self) -> list[WheelConfig]: ...
    def observe(self, results: pd.DataFrame) -> None: ...
    @property
    def finished(self) -> bool: ...


class GridSearcher:
    def __init__(self, space: Sequence[WheelConfig]):
        self._space = list(space)
        self._finished = False

    def propose(self) -> list[WheelConfig]:
        return self._space

    def observe(self, results: pd.DataFrame) -> None:
        self._finished = True

    @property
    def finished(self) -> bool:
        return self._finished


def run_search(objective: Objective, searcher: Searcher, window: Window = FULL,
               verbose: bool = True) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    started = time.time()
    while not searcher.finished:
        batch = searcher.propose()
        rows = []
        for i, cfg in enumerate(batch):
            rows.append(result_row(cfg, objective(cfg, window)))
            if verbose and (i % 100 == 0 or i == len(batch) - 1):
                best = max((r["m_cagr"] for r in rows), default=float("nan"))
                print(f"  {i + 1}/{len(batch)}  {time.time() - started:5.0f}s  best CAGR {best:.1%}")
        frame = pd.DataFrame(rows)
        frames.append(frame)
        searcher.observe(frame)
    return pd.concat(frames, ignore_index=True)


def grid_search(objective: Objective, space: Sequence[WheelConfig],
                window: Window = FULL, verbose: bool = True) -> pd.DataFrame:
    return run_search(objective, GridSearcher(space), window, verbose)


def rank(results: pd.DataFrame, by: str = "m_cagr", n: int = 20,
         catastrophe_dd: float = -0.60) -> pd.DataFrame:
    out = results.copy()
    out["catastrophic_dd"] = out["m_max_drawdown"] < catastrophe_dd
    return out.sort_values(by, ascending=False).head(n)


def walk_forward_select(results: pd.DataFrame, space: Sequence[WheelConfig],
                        objective: Objective, dd_limit: float = -0.45,
                        shortlist: int = 15) -> tuple[WheelConfig, pd.DataFrame]:
    ranked = results[results["m_max_drawdown"] > dd_limit].sort_values("m_cagr", ascending=False)
    if not len(ranked):
        ranked = results.sort_values("m_cagr", ascending=False)
    ranked = ranked.head(shortlist)

    by_label = {c.label: c for c in space}
    rows = []
    for label in ranked["label"]:
        v = objective(by_label[label], VALID)
        rows.append(dict(label=label,
                         train_cagr=float(ranked.loc[ranked["label"] == label, "m_cagr"].iloc[0]),
                         valid_cagr=v["cagr"], valid_calmar=v["calmar"],
                         valid_maxdd=v["max_drawdown"]))
    table = pd.DataFrame(rows)
    table["score"] = table["valid_cagr"] * table["valid_calmar"].clip(lower=0) ** 0.3
    table = table.sort_values("score", ascending=False).reset_index(drop=True)
    return by_label[table.iloc[0]["label"]], table
