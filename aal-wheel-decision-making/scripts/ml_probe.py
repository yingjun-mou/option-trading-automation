from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasource import load_default
from src.engine import Chain
from src.features import build_features

FEATURES = ["pct_3y", "iv_pct_1y", "iv_minus_rv20", "ret_20", "ret_60", "rsi_14", "dte", "delta"]


def build_csp_dataset(features, options, dtes=(7, 14, 21, 30, 45),
                      deltas=(0.15, 0.20, 0.30)) -> pd.DataFrame:
    chain = Chain(options)
    close = features["close"]
    rows = []
    for date in features.index:
        if not chain.has(date):
            continue
        fx = features.loc[date]
        for dte in dtes:
            for delta in deltas:
                q = chain.select(date, "P", dte, delta, 6, max_strike=float(fx["close"]) * 1.05)
                if q is None:
                    continue
                path = close.loc[date:q["expiration"]]
                if len(path) < 2:
                    continue
                s_exp = float(path.iloc[-1])
                payoff = q["mid"] - max(0.0, q["strike"] - s_exp)
                rows.append(dict(date=date, dte=q["dte"], delta=q["delta"],
                                 pct_3y=fx.get("pct_3y"), iv_pct_1y=fx.get("iv_pct_1y"),
                                 iv_minus_rv20=fx.get("iv_minus_rv20"), ret_20=fx.get("ret_20"),
                                 ret_60=fx.get("ret_60"), rsi_14=fx.get("rsi_14"),
                                 realised_ret=payoff / q["strike"]))
    return pd.DataFrame(rows).dropna().set_index("date").sort_index()


def main(train_end="2021-12-31", valid_end="2023-12-31"):
    data = load_default()
    features = build_features(data)
    df = build_csp_dataset(features, data.options)
    train, valid, test = df.loc[:train_end], df.loc[train_end:valid_end], df.loc[valid_end:]
    print(f"rows  train={len(train)}  valid={len(valid)}  test={len(test)}")

    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.linear_model import LinearRegression
    except ImportError:
        print("scikit-learn not installed; skipping model fit.")
        return

    for name, model in [("linear", LinearRegression()),
                        ("gbr", GradientBoostingRegressor(max_depth=2, n_estimators=200,
                                                          learning_rate=0.03))]:
        model.fit(train[FEATURES], train["realised_ret"])
        for split_name, split in [("valid", valid), ("test", test)]:
            pred = model.predict(split[FEATURES])
            take = pred > np.quantile(pred, 0.5)
            print(f"{name:7s} {split_name}: all={split['realised_ret'].mean():+.4f}  "
                  f"model-gated={split['realised_ret'][take].mean():+.4f}  n={int(take.sum())}")

    print("\nunivariate corr with realised CSP return (train):")
    print(train[FEATURES + ["realised_ret"]].corr()["realised_ret"].drop("realised_ret")
          .sort_values(key=abs, ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()
