from __future__ import annotations

import json
import threading
from pathlib import Path

import pandas as pd

from .iv_rank import compute_rv_rank
from .realtime import REALTIME_DIR
from .scanner import load_universe

CACHE_FILE = REALTIME_DIR / "iv_rank_cache.json"
REFRESH_SECONDS = 20 * 3600   # RV-rank barely moves within a day; no need to chase it
STARTUP_DELAY = 5 * 60        # let the options scan claim Yahoo's attention first on a cold start


class IvRankJob:
    """Background loop maintaining the realized-vol-rank proxy used for the
    scanner's "IV Rank" filter (see iv_rank.compute_rv_rank for why it's a
    proxy). Decoupled from ScannerJob's ~15-min cadence on purpose -- this
    changes slowly and every extra Yahoo call is extra rate-limit risk."""

    def __init__(self, cache_file: Path = CACHE_FILE, interval: int = REFRESH_SECONDS,
                startup_delay: int = STARTUP_DELAY):
        self.cache_file = cache_file
        self.interval = interval
        self.startup_delay = startup_delay
        self.as_of: str | None = None
        self.ranks: dict[str, float] = {}
        self._wake = threading.Event()
        self._load_cache()

    def _load_cache(self) -> None:
        if not self.cache_file.exists():
            return
        try:
            raw = json.loads(self.cache_file.read_text())
            self.as_of = raw.get("as_of")
            self.ranks = raw.get("ranks", {})
        except Exception:
            pass

    def _write_cache(self) -> None:
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_text(json.dumps({"as_of": self.as_of, "ranks": self.ranks}))

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
            self.ranks = compute_rv_rank(load_universe())
            self.as_of = pd.Timestamp.now("UTC").isoformat(timespec="seconds")
            self._write_cache()
        except Exception:
            pass  # keep serving the last good ranks; next cycle tries again

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
        threading.Thread(target=self._loop, daemon=True).start()
