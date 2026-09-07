from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.advisor import PortfolioState, advise, compute_live_features
from src.datasource import load_default
from src.features import build_features
from src.realtime import MOCK_DIR, default_realtime_source
from src.wheel import WheelConfig


def _config() -> WheelConfig:
    path = ROOT / "results" / "optimized_config.json"
    if path.exists():
        raw = json.loads(path.read_text())
        raw.pop("label", None)
        return WheelConfig(**{**WheelConfig().to_dict(), **raw, "label": "OPTIMISED"})
    return WheelConfig(label="DEFAULT")


def main():
    market = load_default()
    feats_hist = build_features(market)
    source = default_realtime_source()
    quote = source.quote()
    chain = source.option_chain()
    feats = compute_live_features(feats_hist["close"], feats_hist["iv_30"],
                                  quote.price, source.atm_iv, quote.timestamp)
    portfolio = PortfolioState.load(MOCK_DIR / "portfolio.json")
    a = advise(_config(), portfolio, feats, chain, quote)

    print(f"\n{a.symbol}  ${a.spot:.2f}  ({a.change_pct:+.2f}%)   {a.as_of}   [{a.data_source}]")
    print(f"portfolio: ${portfolio.cash:,.0f} cash, {portfolio.shares} shares, "
          f"{len(portfolio.positions)} open option(s)\n")
    print("signals:", {k: round(v, 3) for k, v in a.signals.items()
                       if isinstance(v, float)})
    print()
    for i, r in enumerate(a.recommendations):
        head = ">>" if i == 0 else "  "
        print(f"{head} [{r['action']}] {r['headline']}")
        print(f"     {r['rationale']}")
        for k, v in (r["params"] or {}).items():
            if k not in ("kind", "opt_type"):
                print(f"       - {k}: {v}")
        print()


if __name__ == "__main__":
    main()
