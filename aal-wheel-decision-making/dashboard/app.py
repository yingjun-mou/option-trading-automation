"""Flask app for the advisor dashboard: two tabs, two JSON APIs. AAL Wheel
(`/api/advice`, `/api/history`) polls a live quote through the same rules as
the research backtest. Premium Scanner (`/api/scan`, `/api/scan/refresh`)
serves the background ScannerJob's cached cross-sectional scan."""

from __future__ import annotations

import hmac
import json
import os
import sys
import threading
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.advisor import PortfolioState, advise, compute_live_features  # noqa: E402
from src.datasource import load_default  # noqa: E402
from src.features import build_features  # noqa: E402
from src.iv_rank_job import IvRankJob  # noqa: E402
from src.macro_job import MacroJob  # noqa: E402
from src.realtime import MOCK_DIR, default_realtime_source  # noqa: E402
from src.scanner_job import ScannerJob  # noqa: E402
from src.wheel import WheelConfig  # noqa: E402

OPTIMIZED_CONFIG = ROOT / "results" / "optimized_config.json"
PORTFOLIO_FILE = MOCK_DIR / "portfolio.json"

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

# HTTP Basic Auth, active only when both env vars are set -- local dev (nothing
# set) stays completely open, exactly like before. Set these when deploying
# anywhere reachable from the internet (this dashboard has no other access
# control, and the Premium Scanner's activity is otherwise visible to anyone
# with the URL). Compared with hmac.compare_digest to avoid a timing attack.
DASHBOARD_USER = os.environ.get("DASHBOARD_USER")
DASHBOARD_PASSWORD = os.environ.get("DASHBOARD_PASSWORD")

if os.environ.get("RENDER") and not (DASHBOARD_USER and DASHBOARD_PASSWORD):
    print("dashboard: WARNING -- running on Render with DASHBOARD_USER/DASHBOARD_PASSWORD "
          "unset. This instance is reachable from the public internet with NO login.")


@app.before_request
def _require_auth():
    """Prompt for the shared username/password if auth is configured (see above)."""
    if not (DASHBOARD_USER and DASHBOARD_PASSWORD):
        return None
    auth = request.authorization
    valid = (auth and hmac.compare_digest(auth.username or "", DASHBOARD_USER)
             and hmac.compare_digest(auth.password or "", DASHBOARD_PASSWORD))
    if not valid:
        return Response("Authentication required.", 401,
                        {"WWW-Authenticate": 'Basic realm="AAL Wheel Advisor"'})
    return None


@app.after_request
def _no_cache(resp):
    """Every response is a live snapshot -- never let the browser cache one."""
    resp.headers["Cache-Control"] = "no-store"
    return resp


def _load_config() -> WheelConfig:
    """The research pipeline's optimised config if it's been run, else defaults."""
    if OPTIMIZED_CONFIG.exists():
        raw = json.loads(OPTIMIZED_CONFIG.read_text())
        raw.pop("label", None)
        return WheelConfig(**{**WheelConfig().to_dict(), **raw, "label": "OPTIMISED"})
    return WheelConfig(label="DEFAULT")


# Yahoo Finance blocks/challenges requests from cloud-provider IP ranges
# (Render's free tier included) with a 401 "Invalid Crumb" error even though
# the exact same yfinance calls work fine from a residential IP -- this is
# unrelated to the DASHBOARD_USER/PASSWORD login above, it's Yahoo rejecting
# the *outbound* request before it ever reaches this app. Routing through a
# proxy is the fix; this is the one place that sets it, since yf.config is
# global and every module (scanner/iv_rank/macro) imports yfinance lazily
# per-function rather than sharing one session. Unset locally, so local runs
# are unaffected.
YFINANCE_PROXY_URL = os.environ.get("YFINANCE_PROXY_URL")
if YFINANCE_PROXY_URL:
    import yfinance as yf
    from urllib.parse import urlsplit
    yf.config.network.proxy = YFINANCE_PROXY_URL
    # Confirms (in Render's logs) that the env var was actually read and
    # parses as a URL -- without this, "still getting 401s" is ambiguous
    # between "proxy never got configured" and "proxy configured but that
    # IP is also blocked/misconfigured", which need very different fixes.
    parsed = urlsplit(YFINANCE_PROXY_URL)
    print(f"dashboard: yfinance proxy configured -- scheme={parsed.scheme!r} "
          f"host={parsed.hostname!r} port={parsed.port!r} "
          f"user_set={bool(parsed.username)} password_set={bool(parsed.password)}")

IV_RANK = IvRankJob()
IV_RANK.start()
SCANNER = ScannerJob(iv_rank_provider=lambda: IV_RANK.signals)
SCANNER.start()
MACRO = MacroJob()
MACRO.start()

# The AAL Wheel tab's context (historical features + live-quote source) is
# built lazily, on first use, NOT at import time like the two jobs above.
# Building it eagerly used to block the module import itself on
# MockOptions.fetch() regenerating its ~124MB synthetic-options file from
# scratch (a per-day Black-Scholes loop over ~13 years) whenever
# historical_data/ is empty -- true on any fresh clone/deploy, since it's
# gitignored. Confirmed live: this made gunicorn's own app-import validation
# (which runs before it binds the port) blow past Render's port-scan
# timeout, failing the deploy outright with "no open ports detected" even
# though the app itself was fine. The Premium Scanner tab has no dependency
# on any of this and is unaffected either way.
_aal_lock = threading.Lock()
_aal_context = None  # (hist_close, hist_iv30, config, source) once built


def _get_aal_context():
    """Build (and cache) the AAL Wheel tab's context on first call."""
    global _aal_context
    with _aal_lock:
        if _aal_context is None:
            print("dashboard: loading historical feature context ...")
            market = load_default()
            hist_features = build_features(market)
            config = _load_config()
            source = default_realtime_source()
            _aal_context = (hist_features["close"], hist_features["iv_30"], config, source)
            print(f"dashboard: config={config.label}  feed={type(source).__name__}  ready")
        return _aal_context


def _current_advice():
    """One fresh Advice: live quote + chain -> features -> `advisor.advise()`."""
    hist_close, hist_iv30, config, source = _get_aal_context()
    quote = source.quote()
    chain_df = source.option_chain()
    feats = compute_live_features(hist_close, hist_iv30, quote.price, source.atm_iv,
                                  quote.timestamp)
    portfolio = PortfolioState.load(PORTFOLIO_FILE)
    return advise(config, portfolio, feats, chain_df, quote)


@app.route("/")
def index():
    """The single-page dashboard (both tabs; JS fetches the APIs below).
    Doesn't force the AAL context to build -- shows a placeholder label
    until whichever request gets there first (see _get_aal_context)."""
    label = _aal_context[2].label if _aal_context is not None else "loading..."
    return render_template("index.html", config_label=label)


@app.route("/api/advice")
def api_advice():
    """Polled every 6s by the AAL Wheel tab."""
    from dataclasses import asdict
    return jsonify(asdict(_current_advice()))


@app.route("/api/history")
def api_history():
    """Recent closes for the AAL Wheel tab's sparkline."""
    _, _, _, source = _get_aal_context()
    closes = source.recent_closes(120)
    return jsonify(dates=[str(d.date()) for d in closes.index], close=[float(c) for c in closes])


@app.route("/api/scan")
def api_scan():
    """Polled every 15s by the Premium Scanner tab -- the cached scan, not a live one."""
    return jsonify(SCANNER.snapshot())


@app.route("/api/scan/refresh", methods=["POST"])
def api_scan_refresh():
    """The Premium Scanner tab's "Refresh now" button."""
    SCANNER.trigger_refresh()
    return jsonify(triggered=True)


@app.route("/api/macro")
def api_macro():
    """Polled every 60s by the Macro tab -- MacroJob's cached QQQ/VIX snapshot."""
    return jsonify(as_of=MACRO.as_of, signals=MACRO.signals)


if __name__ == "__main__":
    print("dashboard: open http://127.0.0.1:5000 in a browser")
    app.run(host="127.0.0.1", port=5000, debug=False)
