from __future__ import annotations

import json
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.advisor import PortfolioState, advise, compute_live_features  # noqa: E402
from src.datasource import load_default  # noqa: E402
from src.features import build_features  # noqa: E402
from src.realtime import MOCK_DIR, default_realtime_source  # noqa: E402
from src.wheel import WheelConfig  # noqa: E402

OPTIMIZED_CONFIG = ROOT / "results" / "optimized_config.json"
PORTFOLIO_FILE = MOCK_DIR / "portfolio.json"

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True


@app.after_request
def _no_cache(resp):
    resp.headers["Cache-Control"] = "no-store"
    return resp


def _load_config() -> WheelConfig:
    if OPTIMIZED_CONFIG.exists():
        raw = json.loads(OPTIMIZED_CONFIG.read_text())
        raw.pop("label", None)
        return WheelConfig(**{**WheelConfig().to_dict(), **raw, "label": "OPTIMISED"})
    return WheelConfig(label="DEFAULT")


print("dashboard: loading historical feature context ...")
_market = load_default()
_hist_features = build_features(_market)
HIST_CLOSE = _hist_features["close"]
HIST_IV30 = _hist_features["iv_30"]
CONFIG = _load_config()
SOURCE = default_realtime_source()
print(f"dashboard: config={CONFIG.label}  feed={type(SOURCE).__name__}  ready")


def _current_advice():
    quote = SOURCE.quote()
    chain_df = SOURCE.option_chain()
    feats = compute_live_features(HIST_CLOSE, HIST_IV30, quote.price, SOURCE.atm_iv,
                                  quote.timestamp)
    portfolio = PortfolioState.load(PORTFOLIO_FILE)
    return advise(CONFIG, portfolio, feats, chain_df, quote)


@app.route("/")
def index():
    return render_template("index.html", config_label=CONFIG.label)


@app.route("/api/advice")
def api_advice():
    from dataclasses import asdict
    return jsonify(asdict(_current_advice()))


@app.route("/api/history")
def api_history():
    closes = SOURCE.recent_closes(120)
    return jsonify(dates=[str(d.date()) for d in closes.index], close=[float(c) for c in closes])


if __name__ == "__main__":
    print("dashboard: open http://127.0.0.1:5000 in a browser")
    app.run(host="127.0.0.1", port=5000, debug=False)
