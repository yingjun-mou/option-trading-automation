"""Background loop that keeps `watchlist.compute_price_changes()` fresh for
every ticker in realtime_data/watchlist.csv and caches it to disk -- see
WatchlistJob's docstring. Same shape as iv_rank_job.IvRankJob, just a much
smaller universe on a much shorter cadence."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pandas as pd

from .realtime import REALTIME_DIR
from .watchlist import compute_price_changes, load_watchlist_rows

CACHE_FILE = REALTIME_DIR / "watchlist_cache.json"
REFRESH_SECONDS = 15 * 60   # a couple of batched yf.download calls for ~85 tickers -- cheap, can run often
STARTUP_DELAY = 45          # let the heavier scanner/IV-rank jobs claim Yahoo's attention first


class WatchlistJob:
    """Background loop maintaining {yahoo_symbol: price/return signals} for
    every ticker in the watchlist CSV. `WatchlistJob.signals` is read by
    app.py's /api/watchlist and merged there with the static
    category/note/stock rows from watchlist.load_watchlist_rows()."""

    def __init__(self, cache_file: Path = CACHE_FILE, interval: int = REFRESH_SECONDS,
                startup_delay: int = STARTUP_DELAY):
        self.cache_file = cache_file
        self.interval = interval
        self.startup_delay = startup_delay
        self.as_of: str | None = None
        self.signals: dict[str, dict] = {}
        self._wake = threading.Event()
        self._load_cache()

    def _load_cache(self) -> None:
        """Seed `self.signals` from CACHE_FILE if present, so there's
        something to serve immediately on startup, before the first
        computation runs."""
        if not self.cache_file.exists():
            return
        try:
            raw = json.loads(self.cache_file.read_text())
            self.signals = raw.get("signals", {})
            self.as_of = raw.get("as_of")
        except Exception:
            pass

    def _write_cache(self) -> None:
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_text(json.dumps({"as_of": self.as_of, "signals": self.signals}))

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
            symbols = [r["yahoo_symbol"] for r in load_watchlist_rows()]
            if not symbols:
                return
            self.signals = compute_price_changes(symbols)
            self.as_of = pd.Timestamp.now("UTC").isoformat(timespec="seconds")
            self._write_cache()
        except Exception:
            pass  # keep serving the last good signals; next cycle tries again

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
        """Launch the background loop (daemon thread -- doesn't block process exit)."""
        threading.Thread(target=self._loop, daemon=True).start()
