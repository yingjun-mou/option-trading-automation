from __future__ import annotations

from itertools import product

from .wheel import WheelConfig

CSP_PCT_LADDERS = {
    "pct_50_30_20_10": ((0.50, 0.30, 0.20, 0.10), (0.25, 0.50, 0.75, 1.00)),
    "pct_40_25_15_08": ((0.40, 0.25, 0.15, 0.08), (0.25, 0.50, 0.75, 1.00)),
    "pct_35_20_10": ((0.35, 0.20, 0.10), (0.33, 0.66, 1.00)),
    "pct_25_15_08": ((0.25, 0.15, 0.08), (0.20, 0.55, 1.00)),
}
CSP_ABS_LADDERS = {
    "abs_14_13_12_11_10": ((14, 13, 12, 11, 10), (0.20, 0.40, 0.60, 0.80, 1.00)),
    "abs_13_12_11_10": ((13, 12, 11, 10), (0.25, 0.50, 0.75, 1.00)),
    "abs_12_11_10_9": ((12, 11, 10, 9), (0.25, 0.50, 0.75, 1.00)),
}
CC_ABS_LADDERS = {
    "cc_12_14_16_18": ((12, 14, 16, 18), (0.0, 0.34, 0.67, 1.00)),
    "cc_14_16_18": ((14, 16, 18), (0.34, 0.67, 1.00)),
    "cc_13_15_17_19": ((13, 15, 17, 19), (0.0, 0.34, 0.67, 1.00)),
}


def shape_tranches(cumulative: tuple, shape: str) -> tuple:
    if shape == "equal":
        return cumulative
    weights = [(i + 1) ** 1.6 for i in range(len(cumulative))]
    total = sum(weights)
    out, running = [], 0.0
    for w in weights:
        running += w / total
        out.append(round(running, 3))
    return tuple(out)


def stage1_space(base: WheelConfig | None = None) -> list[WheelConfig]:
    base = base or WheelConfig()
    ladders = ([("pct_3y", edges, tr, name) for name, (edges, tr) in CSP_PCT_LADDERS.items()]
               + [("abs", edges, tr, name) for name, (edges, tr) in CSP_ABS_LADDERS.items()])
    out = []
    for (signal, edges, tranches, name), dte, delta, pt, alloc, iv_min, shape in product(
        ladders,
        [7, 14, 21, 30, 45],
        [0.15, 0.20, 0.30],
        [1.00, 0.50],
        [0.60, 1.00],
        [None, 0.50],
        ["equal", "increasing"],
    ):
        out.append(base.copy(
            csp_signal=signal,
            csp_edges=edges,
            csp_tranches=shape_tranches(tranches, shape),
            csp_reentry_edge=edges[0] if signal != "abs" else 0.50,
            csp_dte=dte, csp_delta=delta, csp_profit_take=pt,
            max_alloc_pct=alloc, iv_pct_min=iv_min,
            label=f"{name}|dte{dte}|d{delta}|pt{pt}|a{alloc}|iv{iv_min}|{shape}",
        ))
    return out


def stage2_space(base: WheelConfig) -> list[WheelConfig]:
    out = []
    for (name, (edges, tranches)), dte, delta, callaway, min_basis in product(
        CC_ABS_LADDERS.items(),
        [14, 30, 45],
        [0.20, 0.30],
        [0.35, 0.50],
        [True, False],
    ):
        out.append(base.copy(
            cc_signal="abs", cc_edges=edges, cc_tranches=tranches,
            cc_dte=dte, cc_delta=delta, cc_callaway_delta=callaway,
            cc_min_strike_vs_basis=min_basis,
            label=base.label + f" || {name}|ccdte{dte}|ccd{delta}|caw{callaway}|minb{min_basis}",
        ))
    return out


def diversification_space(base: WheelConfig) -> list[WheelConfig]:
    return [
        base.copy(csp_dte_ladder=(), label=base.label + " || concentrated"),
        base.copy(csp_dte_ladder=(14, 21, 30), label=base.label + " || exp_ladder"),
        base.copy(csp_dte_ladder=(7, 14, 21, 30, 45), label=base.label + " || exp_ladder_wide"),
    ]
