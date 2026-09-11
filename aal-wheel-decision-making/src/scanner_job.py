"""Background loop that runs `scanner.scan_universe()` on a cadence, caches
the result to disk, and serves it to the dashboard (`ScannerJob.snapshot()`
-> `/api/scan`). See scanner.py's module docstring for the feature overview."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .realtime import REALTIME_DIR
from .scanner import load_universe, scan_universe

CACHE_FILE = REALTIME_DIR / "scanner_cache.json"
# A full scan takes ~8-9 min on its own (scanner.REQUEST_GAP paces requests to avoid
# Yahoo's rate limit) -- wait only ~6 min after that so cycles land near 15 min apart.
REFRESH_SECONDS = 6 * 60


@dataclass
class ScannerState:
    status: str = "idle"          # idle | scanning | error
    as_of: str | None = None
    scanned: int = 0
    failed: int = 0
    progress: tuple[int, int] = (0, 0)
    error: str | None = None
    rows: list[dict] = field(default_factory=list)


def _price_percentile(spot: float, recent_closes: list[float] | None) -> float | None:
    """Where `spot` -- today's LIVE quote, not a stale daily close -- sits
    (0-1 percentile) against a trailing window of historical closes. Same
    definition as advisor.py's pct_1y/pct_3y signals: the fraction of the
    window at or below the current price. None with no window to compare
    against yet (IvRankJob hasn't completed its first cycle)."""
    if not recent_closes:
        return None
    closes = pd.Series(recent_closes)
    return float((closes <= spot).mean())


def _json_safe(obj):
    """Recursively replace float NaN with None -- pandas leaves genuinely
    missing numeric fields (e.g. cc_premium with no live/recent quote) as
    NaN, and `json.dumps` emits that as the bare token NaN, which is not
    valid JSON and JS's JSON.parse rejects outright."""
    if isinstance(obj, float) and pd.isna(obj):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


class ScannerJob:
    """Owns the background scan loop: runs on startup, then every
    REFRESH_SECONDS, writing results to CACHE_FILE. `trigger_refresh()` wakes
    it immediately (used by the dashboard's manual refresh button).

    `iv_rank_provider`, if given, is called once per scan to get the current
    {symbol: {"iv_rank_pct": ..., "recent_closes": [...]}} map (see
    iv_rank_job.IvRankJob) and merge iv_rank_pct + a live-computed price_pct
    (today's scanned spot ranked against recent_closes) into each row --
    kept as an injected callable so this module doesn't need to know that
    job's refresh cadence or cache format."""

    def __init__(self, cache_file: Path = CACHE_FILE, interval: int = REFRESH_SECONDS,
                iv_rank_provider=None):
        self.cache_file = cache_file
        self.interval = interval
        self.iv_rank_provider = iv_rank_provider
        self.state = ScannerState()
        self._wake = threading.Event()
        self._lock = threading.Lock()
        self._load_cache()

    def _load_cache(self) -> None:
        """Seed `self.state` from CACHE_FILE if present, so the dashboard has
        something to show immediately on startup, before the first scan runs."""
        if not self.cache_file.exists():
            return
        try:
            raw = json.loads(self.cache_file.read_text())
            self.state.as_of = raw.get("as_of")
            self.state.rows = raw.get("rows", [])
            self.state.scanned = raw.get("scanned", len(self.state.rows))
            self.state.failed = raw.get("failed", 0)
        except Exception:
            pass

    def _write_cache(self) -> None:
        """Persist the current results so a restart can reuse them (see `_load_cache`)."""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_text(json.dumps({
            "as_of": self.state.as_of,
            "scanned": self.state.scanned,
            "failed": self.state.failed,
            "rows": self.state.rows,
        }))

    def _run_scan(self) -> None:
        """Scan the full universe once, merge in IV Rank + price percentile,
        and cache the result. No-ops if a scan is already running."""
        with self._lock:
            if self.state.status == "scanning":
                return
            self.state.status = "scanning"
            self.state.error = None
        try:
            symbols = load_universe()
            self.state.progress = (0, len(symbols))
            df = scan_universe(symbols, on_progress=lambda i, n: setattr(self.state, "progress", (i, n)))
            ok = df[df["error"].isna()].drop(columns=["error"])
            if self.iv_rank_provider is not None:
                signals = self.iv_rank_provider()
                if signals:
                    ok = ok.copy()
                    ok["iv_rank_pct"] = ok["symbol"].map(lambda s: signals.get(s, {}).get("iv_rank_pct"))
                    ok["price_pct"] = ok.apply(
                        lambda r: _price_percentile(r["spot"], signals.get(r["symbol"], {}).get("recent_closes")),
                        axis=1)
            self.state.rows = _json_safe(ok.to_dict(orient="records"))
            self.state.scanned = len(ok)
            self.state.failed = int(df["error"].notna().sum())
            self.state.as_of = pd.Timestamp.now("UTC").isoformat(timespec="seconds")
            self._write_cache()
        except Exception as e:  # noqa: BLE001
            self.state.error = str(e)
        finally:
            self.state.status = "idle"

    def _cache_age_seconds(self) -> float | None:
        """Seconds since the cached scan, or None if there isn't one yet."""
        if not self.state.as_of:
            return None
        try:
            ts = pd.Timestamp(self.state.as_of)
            now = pd.Timestamp.now(tz=ts.tzinfo)
            return (now - ts).total_seconds()
        except Exception:
            return None

    def _loop(self) -> None:
        """Runs forever on a background thread: scan, wait `interval`, repeat.
        `trigger_refresh()` interrupts the wait to scan sooner."""
        # A restart (dev reload, crash recovery) shouldn't blow away a still-fresh
        # cache by immediately re-scanning 800+ tickers against a rate-limited API --
        # only fire early if the cache is missing or already stale.
        age = self._cache_age_seconds()
        if age is not None and age < self.interval:
            self._wake.wait(self.interval - age)
            self._wake.clear()
        while True:
            self._run_scan()
            self._wake.wait(self.interval)
            self._wake.clear()

    def start(self) -> None:
        """Launch the background loop (daemon thread -- doesn't block process exit)."""
        threading.Thread(target=self._loop, daemon=True).start()

    def trigger_refresh(self) -> None:
        """Wake the loop early -- the dashboard's manual "Refresh now" button."""
        self._wake.set()

    def snapshot(self) -> dict:
        """Current state as a plain dict, ready for `jsonify()` -- what `/api/scan` serves."""
        s = self.state
        return dict(status=s.status, as_of=s.as_of, scanned=s.scanned, failed=s.failed,
                    progress=list(s.progress), error=s.error, rows=s.rows)
