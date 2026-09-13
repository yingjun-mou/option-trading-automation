"""Background loop that keeps `macro.compute_macro_signals()` fresh on its
own cadence and caches it to disk -- see MacroJob's docstring. Modeled on
IvRankJob (src/iv_rank_job.py) but for a single, tiny signal set (QQQ + VIX,
not an 850-symbol universe), so it runs far more often and needs no
startup-priority delay for the option scan."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pandas as pd

from .macro import SCHEMA_VERSION, compute_macro_signals
from .realtime import REALTIME_DIR

CACHE_FILE = REALTIME_DIR / "macro_cache.json"
REFRESH_SECONDS = 30 * 60   # QQQ/VIX move during the trading day, unlike IvRankJob's ~daily signals
STARTUP_DELAY = 10          # a handful of yfinance calls -- cheap, no need to wait for the scan


class MacroJob:
    """Background loop maintaining the Macro tab's snapshot (QQQ price,
    Yahoo's own 50/200-day averages, the 6-month return, ADX(14), and
    VIX/VIX3M -- see macro.compute_macro_signals). Independent of
    ScannerJob/IvRankJob's universes and cadence."""

    def __init__(self, cache_file: Path = CACHE_FILE, interval: int = REFRESH_SECONDS,
                startup_delay: int = STARTUP_DELAY):
        self.cache_file = cache_file
        self.interval = interval
        self.startup_delay = startup_delay
        self.as_of: str | None = None
        self.signals: dict = {}
        self._wake = threading.Event()
        self._load_cache()

    def _load_cache(self) -> None:
        """Seed `self.signals` from CACHE_FILE if present -- see
        IvRankJob._load_cache's docstring for the schema-mismatch handling,
        duplicated here identically."""
        if not self.cache_file.exists():
            return
        try:
            raw = json.loads(self.cache_file.read_text())
            self.signals = raw.get("signals", {})
            if raw.get("schema_version") == SCHEMA_VERSION:
                self.as_of = raw.get("as_of")
        except Exception:
            pass

    def _write_cache(self) -> None:
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_text(json.dumps(
            {"schema_version": SCHEMA_VERSION, "as_of": self.as_of, "signals": self.signals}))

    def _cache_age_seconds(self) -> float | None:
        if not self.as_of:
            return None
        try:
            ts = pd.Timestamp(self.as_of)
            return (pd.Timestamp.now(tz=ts.tzinfo) - ts).total_seconds()
        except Exception:
            return None

    def _run(self) -> None:
        try:
            signals = compute_macro_signals()
            if signals:
                self.signals = signals
                self.as_of = pd.Timestamp.now("UTC").isoformat(timespec="seconds")
                self._write_cache()
        except Exception:
            pass  # keep serving the last good snapshot; next cycle tries again

    def _loop(self) -> None:
        age = self._cache_age_seconds()
        wait_s = (self.interval - age) if (age is not None and age < self.interval) else self.startup_delay
        self._wake.wait(wait_s)
        self._wake.clear()
        while True:
            self._run()
            self._wake.wait(self.interval)
            self._wake.clear()

    def start(self) -> None:
        """Launch the background loop (daemon thread -- doesn't block process exit,
        and doesn't block gunicorn's startup import either)."""
        threading.Thread(target=self._loop, daemon=True).start()
