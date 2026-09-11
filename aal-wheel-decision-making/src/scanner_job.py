from __future__ import annotations

import json
import threading
import time
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


class ScannerJob:
    """Owns the background scan loop: runs on startup, then every
    REFRESH_SECONDS, writing results to CACHE_FILE. `trigger_refresh()` wakes
    it immediately (used by the dashboard's manual refresh button)."""

    def __init__(self, cache_file: Path = CACHE_FILE, interval: int = REFRESH_SECONDS):
        self.cache_file = cache_file
        self.interval = interval
        self.state = ScannerState()
        self._wake = threading.Event()
        self._lock = threading.Lock()
        self._load_cache()

    def _load_cache(self) -> None:
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
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_text(json.dumps({
            "as_of": self.state.as_of,
            "scanned": self.state.scanned,
            "failed": self.state.failed,
            "rows": self.state.rows,
        }))

    def _run_scan(self) -> None:
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
            self.state.rows = ok.to_dict(orient="records")
            self.state.scanned = len(ok)
            self.state.failed = int(df["error"].notna().sum())
            self.state.as_of = pd.Timestamp.now("UTC").isoformat(timespec="seconds")
            self._write_cache()
        except Exception as e:  # noqa: BLE001
            self.state.error = str(e)
        finally:
            self.state.status = "idle"

    def _cache_age_seconds(self) -> float | None:
        if not self.state.as_of:
            return None
        try:
            ts = pd.Timestamp(self.state.as_of)
            now = pd.Timestamp.now(tz=ts.tzinfo)
            return (now - ts).total_seconds()
        except Exception:
            return None

    def _loop(self) -> None:
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
        threading.Thread(target=self._loop, daemon=True).start()

    def trigger_refresh(self) -> None:
        self._wake.set()

    def snapshot(self) -> dict:
        s = self.state
        return dict(status=s.status, as_of=s.as_of, scanned=s.scanned, failed=s.failed,
                    progress=list(s.progress), error=s.error, rows=s.rows)
