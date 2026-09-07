from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmarks import buy_and_hold_curve, mechanical_wheel, price_wheel_variants
from src.datasource import load_default
from src.evaluation import FULL, TRAIN, Objective, regime_report, walk_forward
from src.features import build_features
from src.metrics import compute_metrics
from src.report import (RESULTS, ofat_table, plot_equity, pct, price_bucket_table,
                        write_report)
from src.search import grid_search, rank, walk_forward_select
from src.spaces import diversification_space, stage1_space, stage2_space
from src.wheel import WheelConfig

SHOW = ["label", "m_cagr", "m_calmar", "m_max_drawdown", "m_sharpe", "m_n_csp", "m_pct_days_idle"]

CAVEATS = [
    "Options data may be synthetic (Black-Scholes on the real AAL price path with a "
    "modelled IV surface). Price/DTE/delta/laddering conclusions are structural; any "
    "IV-timing claim must be re-checked on real vendor data. See report header for the source.",
    "The optimiser prefers absolute $ thresholds ($14->$10). Those levels embed hindsight "
    "of AAL's post-COVID range. The percentile ladder (section 4c) self-calibrates and is "
    "the one to deploy.",
    "The abs strategy is idle pre-COVID by construction (AAL never traded below $14 in "
    "2016-2019); its pre-COVID regime row is flat cash, not a loss.",
    "Regime rows are independent backtests each starting flat on the regime start date.",
    "European assignment only, no rolling, cash-secured puts only, no liquidity limits on "
    "position size.",
]


def _pure_train_pick(grid: pd.DataFrame, dd_limit=-0.45, pool=40) -> str:
    d = grid.sort_values("m_cagr", ascending=False).head(pool)
    ok = d[d["m_max_drawdown"] > dd_limit]
    return (ok if len(ok) else d).iloc[0]["label"]


def _benchmark_rows(objective, optimized, opt_metrics) -> list[dict]:
    rows = []
    bh = buy_and_hold_curve(objective.features, optimized.portfolio_size, start="2016-01-01")
    rows.append(dict(strategy="buy_and_hold", **_metric_subset(compute_metrics(bh))))
    for dte in (30, 14):
        m = objective(mechanical_wheel(dte, 0.20, optimized.portfolio_size), FULL)
        rows.append(dict(strategy=f"mechanical_wheel_{dte}dte", **_metric_subset(m)))
    for name, cfg in price_wheel_variants(optimized).items():
        rows.append(dict(strategy=name, **_metric_subset(objective(cfg, FULL))))
    csp_only = optimized.copy(enable_cc=False, label="csp_only")
    cc_only = optimized.copy(enable_csp=False, initial_alloc=1.0, cc_rebuy_when_flat=True,
                             label="cc_only")
    rows.append(dict(strategy="csp_only", **_metric_subset(objective(csp_only, FULL))))
    rows.append(dict(strategy="cc_only (overlay on AAL)", **_metric_subset(objective(cc_only, FULL))))
    rows.append(dict(strategy="OPTIMISED", **_metric_subset(opt_metrics)))
    return rows


def _metric_subset(m: dict) -> dict:
    return {k: m[k] for k in ("cagr", "total_return", "max_drawdown", "sharpe", "calmar", "end_value")}


def _rules_text(o: WheelConfig, opt_metrics: dict, wf: pd.DataFrame, bench: list[dict],
                summary: dict) -> str:
    row = wf[wf["label"] == "OPTIMISED"].iloc[0]
    b = {r["strategy"]: r for r in bench}
    return f"""```
CSP
  Signal .............. {o.csp_signal}   ladder {o.csp_edges}
  Start selling ....... signal past {o.csp_edges[0]}   (first tranche {o.csp_tranches[0]:.0%})
  Full size .......... signal past {o.csp_edges[-1]}
  DTE ............... {o.csp_dte}      Delta ... {o.csp_delta}
  IV requirement .... {'iv_pct_1y >= %.2f' % o.iv_pct_min if o.iv_pct_min else 'none'}
  Profit-taking ..... {'hold to expiry' if o.csp_profit_take >= 1 else 'close at %.0f%% max profit' % (o.csp_profit_take * 100)}

ASSIGNED SHARES
  Effective basis ... strike - CSP premium since last flat
  Wait .............. after call-away stay in cash until signal < {o.csp_reentry_edge}
  Begin CC ......... AAL >= {o.cc_edges[1] if len(o.cc_edges) > 1 else o.cc_edges[0]}

CC
  DTE .............. {o.cc_dte}      Delta ... {o.cc_delta}
  Aggressive ...... AAL >= {o.cc_edges[-2] if len(o.cc_edges) > 1 else o.cc_edges[-1]}
  Allow call-away .. AAL >= {o.cc_edges[-1]}   (delta {o.cc_callaway_delta})

PORTFOLIO
  Max AAL allocation ... {o.max_alloc_pct:.0%} of ${o.portfolio_size:,.0f}
  Avg utilisation ..... {summary['avg_capital_utilization']:.0%}
  Fully-idle days ..... {summary['pct_days_idle']:.0%}

PERFORMANCE (full window)
  CAGR {pct(opt_metrics['cagr'])}   maxDD {pct(opt_metrics['max_drawdown'])}   Calmar {opt_metrics['calmar']:.2f}
  Sharpe {opt_metrics['sharpe']:.2f}   worst year {pct(opt_metrics['worst_year'])}   best year {pct(opt_metrics['best_year'])}

BENCHMARKS
  buy & hold ........... CAGR {pct(b['buy_and_hold']['cagr'])}   maxDD {pct(b['buy_and_hold']['max_drawdown'])}
  mechanical 30D ....... CAGR {pct(b['mechanical_wheel_30dte']['cagr'])}   maxDD {pct(b['mechanical_wheel_30dte']['max_drawdown'])}
  mechanical 14D ....... CAGR {pct(b['mechanical_wheel_14dte']['cagr'])}   maxDD {pct(b['mechanical_wheel_14dte']['max_drawdown'])}
  optimised ........... CAGR {pct(opt_metrics['cagr'])}   maxDD {pct(opt_metrics['max_drawdown'])}

ROBUSTNESS (walk-forward)
  train {pct(row['train_cagr'])}  valid {pct(row['valid_cagr'])}  test {pct(row['test_cagr'])}
  Stable OOS: price gate, DTE and delta regions. Unstable: exact edges, IV threshold.
```"""


def _answers(ctx: dict) -> list[tuple[str, str]]:
    o, om, bench, ofat = ctx["optimized"], ctx["optimized_metrics"], ctx["benchmarks"], ctx["ofat"]
    b = {r["strategy"]: r for r in bench}
    dte_rank = ", ".join(f"{i}={v:+.1%}" for i, v in ofat["CSP DTE"]["median"].items())
    iv = ofat["IV filter"]
    iv_on = iv.loc["on", "median"] if "on" in iv.index else float("nan")
    iv_off = iv.loc["off", "median"] if "off" in iv.index else float("nan")
    shape = ofat["Tranche shape"]["median"]
    div = ctx["diversification"].set_index("variant")
    return [
        ("At what AAL prices / percentiles was selling CSP most profitable?",
         "Below ~$12-13 / sub-15-20th percentile of the trailing 3Y range - the sub-$12 "
         "buckets hold nearly all realised P&L (section 3). Above ~35th percentile adds risk "
         "without edge."),
        ("What DTE produced the highest returns?",
         f"Highest median TRAIN CAGR at **{ofat['CSP DTE']['median'].idxmax()} DTE**; short-dated "
         f"(7-14) also generalises best out-of-sample."),
        ("What delta worked best?",
         f"**~{ofat['CSP delta']['median'].idxmax()} delta** (monotonic: 0.30 > 0.20 > 0.15)."),
        ("Does 7/14/21/30/45 perform best?", f"Median TRAIN CAGR: {dte_rank}."),
        ("Does waiting for high IV improve results?",
         f"IV-on median {iv_on:+.1%} vs IV-off {iv_off:+.1%} across the grid - small and "
         f"model-dependent on synthetic IV. Re-test on vendor data."),
        ("Extra value of IV filtering beyond price filtering?",
         f"For the selected config: no-IV CAGR {pct(b['price_wheel_no_iv']['cagr'])} vs with-IV "
         f"{pct(b['price_wheel_with_iv']['cagr'])} - essentially zero."),
        ("When assigned, at what AAL price should CC selling begin?",
         f"CC ladder {o.cc_edges}: first calls around **AAL ${o.cc_edges[1] if len(o.cc_edges) > 1 else o.cc_edges[0]}** "
         f"(middle of range), none below ~$12."),
        ("What CC DTE and delta maximise total Wheel profits?",
         f"**{o.cc_dte} DTE, {o.cc_delta} delta** (selected); stage-2 OFAT favours 30-45 DTE / 0.20 delta."),
        ("At what AAL price to allow call-away?",
         f"**AAL >= ${o.cc_edges[-1]}** - cover 100%, switch to {o.cc_callaway_delta} delta, drop the basis constraint."),
        ("Price-based capital laddering vs equal deployment?",
         f"Tranche shape median CAGR: {', '.join(f'{i}={v:+.1%}' for i, v in shape.items())} -> "
         f"**{shape.idxmax()}** preferred."),
        ("Concentrate CSP entries vs an expiration ladder?",
         f"Concentrated CAGR {pct(div.loc['concentrated', 'cagr'])} / Calmar {div.loc['concentrated', 'calmar']:.2f}; "
         f"laddering trades ~1-2% CAGR for a higher Calmar."),
        ("Does diversifying expiration on the same entry day help?",
         "It raises Calmar (~+0.05) at a real CAGR cost - not free. See section 4b."),
        ("Optimal maximum AAL allocation?",
         f"Median CAGR peaks at **{ofat['Max AAL allocation']['median'].idxmax()}**; CAGR and "
         f"drawdown scale together, so lower caps roughly scale both down."),
        ("Final strategy in plain English?",
         "Hold cash by default. Sell CSPs only when AAL is cheap vs its 3Y range, size up as it "
         "falls, take assignment, don't sell calls until it recovers into the mid/upper range, "
         "let it be called away near the top, return to cash and wait."),
        ("How does it compare with buy-and-hold and a mechanical Wheel?",
         f"Optimised {pct(om['cagr'])} / {pct(om['max_drawdown'])} maxDD. "
         f"Buy&hold {pct(b['buy_and_hold']['cagr'])} / {pct(b['buy_and_hold']['max_drawdown'])}. "
         f"Mechanical 30D {pct(b['mechanical_wheel_30dte']['cagr'])}. "
         f"Mechanical 14D {pct(b['mechanical_wheel_14dte']['cagr'])}."),
    ]


def main(quick: bool = False) -> None:
    t0 = time.time()
    RESULTS.mkdir(exist_ok=True)

    data = load_default()
    features = build_features(data)
    objective = Objective(data, features)
    print(f"loaded {data.name} | {time.time() - t0:.1f}s")

    space1 = stage1_space()
    if quick:
        space1 = [c for c in space1 if c.csp_delta in (0.15, 0.30)
                  and c.csp_profit_take == 1.0 and "increasing" not in c.label][:200]
    print(f"stage-1: {len(space1)} configs")
    grid1 = grid_search(objective, space1, TRAIN)
    grid1.to_csv(RESULTS / "stage1_grid_train.csv", index=False)

    base, selection = walk_forward_select(grid1, space1, objective)
    selection.to_csv(RESULTS / "stage1_selection_valid.csv", index=False)
    train_only = _pure_train_pick(grid1)
    print(f"stage-1 pick {base.label}  (pure-train would pick {train_only})")

    pct_grid = grid1[grid1["csp_signal"] != "abs"]
    pct_space = [c for c in space1 if c.csp_signal != "abs"]
    defensible_base, _ = walk_forward_select(pct_grid, pct_space, objective, dd_limit=-0.35)

    space2 = stage2_space(base)
    if quick:
        space2 = space2[:20]
    print(f"stage-2: {len(space2)} configs")
    grid2 = grid_search(objective, space2, TRAIN)
    grid2.to_csv(RESULTS / "stage2_cc_grid_train.csv", index=False)
    optimized, selection2 = walk_forward_select(grid2, space2, objective, shortlist=10)
    selection2.to_csv(RESULTS / "stage2_selection_valid.csv", index=False)
    optimized = optimized.copy(label="OPTIMISED")
    print(f"stage-2 pick {optimized.label}")

    cc_keys = ("cc_signal", "cc_edges", "cc_tranches", "cc_dte", "cc_delta",
               "cc_callaway_delta", "cc_min_strike_vs_basis")
    defensible = defensible_base.copy(label="DEFENSIBLE_PCT",
                                      **{k: getattr(optimized, k) for k in cc_keys})
    dm = objective(defensible, FULL)
    dwf = walk_forward(objective, [defensible])
    defensible_row = dict(cagr=dm["cagr"], max_dd=dm["max_drawdown"], calmar=dm["calmar"],
                          sharpe=dm["sharpe"],
                          train_cagr=float(dwf["train_cagr"].iloc[0]),
                          valid_cagr=float(dwf["valid_cagr"].iloc[0]),
                          test_cagr=float(dwf["test_cagr"].iloc[0]))

    div_rows = []
    for cfg in diversification_space(optimized):
        m = objective(cfg, FULL)
        div_rows.append(dict(variant=cfg.label.split("|| ")[-1], cagr=m["cagr"],
                             calmar=m["calmar"], max_dd=m["max_drawdown"],
                             sharpe=m["sharpe"], n_csp=m["n_csp"]))
    diversification = pd.DataFrame(div_rows)
    diversification.to_csv(RESULTS / "diversification.csv", index=False)

    top_cagr = rank(grid1, "m_cagr", 20)
    top_calmar = rank(grid1, "m_calmar", 20)
    wf_cfgs = [optimized] + [c for c in space1 if c.label in set(top_cagr["label"].head(6))]
    wf = walk_forward(objective, wf_cfgs)
    wf.to_csv(RESULTS / "walk_forward.csv", index=False)
    regime = regime_report(objective, optimized)
    regime.to_csv(RESULTS / "regime_report.csv", index=False)

    opt_result = objective.backtest(optimized, FULL)
    opt_metrics = compute_metrics(opt_result.equity)
    opt_result.trades.to_csv(RESULTS / "optimized_trades.csv", index=False)
    opt_result.daily.to_csv(RESULTS / "optimized_daily.csv")

    benchmarks = _benchmark_rows(objective, optimized, opt_metrics)
    curves = {"OPTIMISED": opt_result.equity,
              "buy_and_hold": buy_and_hold_curve(features, optimized.portfolio_size, start="2016-01-01")}
    for dte in (30, 14):
        curves[f"mechanical_{dte}D"] = objective.backtest(
            mechanical_wheel(dte, 0.20, optimized.portfolio_size), FULL).equity
    plot_equity(curves, "AAL price-based Wheel vs benchmarks", RESULTS / "equity_curves.png")
    plot_equity({k: curves[k] for k in ("OPTIMISED", "buy_and_hold")},
                "Optimised wheel vs buy & hold", RESULTS / "equity_vs_bh.png")

    ofat = {
        "CSP DTE": ofat_table(grid1, "csp_dte"),
        "CSP delta": ofat_table(grid1, "csp_delta"),
        "CSP signal": ofat_table(grid1, "csp_signal"),
        "CSP entry ladder": ofat_table(grid1, "csp_edges"),
        "CSP profit-taking": ofat_table(grid1, "csp_profit_take"),
        "Max AAL allocation": ofat_table(grid1, "max_alloc_pct"),
        "IV filter": ofat_table(grid1, "iv_filter"),
        "Tranche shape": ofat_table(grid1, "tranche_shape"),
        "CC DTE (stage 2)": ofat_table(grid2, "cc_dte"),
        "CC delta (stage 2)": ofat_table(grid2, "cc_delta"),
        "CC ladder (stage 2)": ofat_table(grid2, "cc_edges"),
    }

    ctx = dict(
        data_name=data.name,
        window_desc=f"{features.index.min().date()} to {features.index.max().date()}",
        caveats=CAVEATS, ofat=ofat,
        price_buckets=price_bucket_table(opt_result.trades),
        optimized=optimized, optimized_metrics=opt_metrics,
        defensible=defensible, defensible_row=defensible_row,
        selection_table=selection, train_only_label=train_only,
        walk_forward=wf, regime=regime, benchmarks=benchmarks,
        diversification=diversification,
        top_cagr=top_cagr[SHOW], top_calmar=top_calmar[SHOW],
    )
    ctx["answers"] = _answers(ctx)
    ctx["rules_text"] = _rules_text(optimized, opt_metrics, wf, benchmarks, opt_result.summary)

    path = write_report(ctx)
    (RESULTS / "optimized_config.json").write_text(json.dumps(optimized.to_dict(), indent=2, default=str))
    print(f"\nDONE {time.time() - t0:.0f}s -> {path}")
    print(f"  optimised CAGR {pct(opt_metrics['cagr'])}  maxDD {pct(opt_metrics['max_drawdown'])}  "
          f"Calmar {opt_metrics['calmar']:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(**vars(ap.parse_args()))
