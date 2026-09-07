from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"


def pct(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:+.1%}"


def ofat_table(grid: pd.DataFrame, param: str, metric: str = "m_cagr") -> pd.DataFrame:
    g = grid.copy()
    g[param] = g[param].astype(str)
    return (g.groupby(param)[metric].agg(["median", "mean", "max", "count"])
            .sort_values("median", ascending=False))


def price_bucket_table(trades: pd.DataFrame) -> pd.DataFrame:
    if trades is None or not len(trades):
        return pd.DataFrame()
    puts = trades[trades["action"] == "SELL_PUT"]
    if not len(puts):
        return pd.DataFrame()
    df = puts[["underlying", "cash_flow"]].copy()
    df["bucket"] = pd.cut(df["underlying"], [0, 10, 11, 12, 13, 14, 16, 18, 100])
    return df.groupby("bucket", observed=True).agg(
        n_entries=("cash_flow", "count"),
        total_credit=("cash_flow", "sum"),
        avg_credit_per_entry=("cash_flow", "mean"))


def plot_equity(curves: dict[str, pd.Series], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, eq in curves.items():
        ax.plot(eq.index, eq.values, label=name, linewidth=1.3)
    ax.set_title(title)
    ax.set_ylabel("Portfolio value ($)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _config_block(cfg) -> str:
    return "```\n" + "\n".join(f"{k:26s} {v}" for k, v in cfg.to_dict().items()) + "\n```"


def build_report(ctx: dict) -> str:
    out: list[str] = []
    w = out.append

    w("# AAL Price-Based Wheel - Research Report\n")
    w(f"*Generated {pd.Timestamp.today().date()}. Data: **{ctx['data_name']}**. "
      f"Window {ctx['window_desc']}.*\n")

    w("## 0. Caveats\n")
    for c in ctx["caveats"]:
        w(f"- {c}")
    w("")

    w("## 1. Executive answers\n")
    for i, (q, a) in enumerate(ctx["answers"], 1):
        w(f"**Q{i}. {q}**  \n{a}\n")

    w("## 2. One-factor sensitivity (stage-1 grid, TRAIN)\n")
    for name, table in ctx["ofat"].items():
        w(f"### {name}\n")
        w(table.round(3).to_markdown())
        w("")

    if len(ctx.get("price_buckets", [])):
        w("## 3. CSP credit capture by AAL entry price\n")
        w(ctx["price_buckets"].round(1).to_markdown())
        w("")

    w("## 4. Optimised strategy (validation-selected)\n")
    w(_config_block(ctx["optimized"]))
    w("")

    w("## 4b. Same-day expiration diversification\n")
    w(ctx["diversification"].round(3).to_markdown(index=False))
    w("")

    w("## 4c. Defensible percentile-only variant (no absolute-price hindsight)\n")
    w(_config_block(ctx["defensible"]))
    w(pd.DataFrame([ctx["defensible_row"]]).round(3).to_markdown(index=False))
    w("")

    w("## 4d. Anti-overfit selection (TRAIN shortlist, VALID chooses)\n")
    w(f"Pure-TRAIN CAGR would pick `{ctx['train_only_label']}`. Re-ranked on 2022-2023:\n")
    w(ctx["selection_table"].round(3).to_markdown(index=False))
    w("")

    w("## 5. Walk-forward (TEST never optimised on)\n")
    w(ctx["walk_forward"].round(3).to_markdown(index=False))
    w("")

    w("## 6. Regime breakdown (optimised config)\n")
    w(ctx["regime"].round(3).to_markdown(index=False))
    w("")

    w("## 7. Benchmarks\n")
    w(pd.DataFrame(ctx["benchmarks"]).round(3).to_markdown(index=False))
    w("")

    w("## 8. Top 20 by CAGR (TRAIN)\n")
    w(ctx["top_cagr"].round(3).to_markdown(index=False))
    w("\n## 9. Top 20 by Calmar (TRAIN)\n")
    w(ctx["top_calmar"].round(3).to_markdown(index=False))
    w("")

    w("## 10. AAL OPTIMIZED WHEEL RULES\n")
    w(ctx["rules_text"])

    return "\n".join(str(x) for x in out)


def write_report(ctx: dict, name: str = "RESEARCH_REPORT.md") -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / name
    path.write_text(build_report(ctx), encoding="utf-8")
    return path
