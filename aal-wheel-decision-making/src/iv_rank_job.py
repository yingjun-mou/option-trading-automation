"""Background loop that keeps `iv_rank.compute_market_signals()` +
`compute_forward_pe()` fresh on its own slow (~daily) cadence and caches them
to disk -- see IvRankJob's docstring."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pandas as pd

from .iv_rank import SCHEMA_VERSION, compute_forward_pe, compute_market_signals
from .realtime import REALTIME_DIR
from .scanner import load_universe

CACHE_FILE = REALTIME_DIR / "iv_rank_cache.json"
REFRESH_SECONDS = 20 * 3600   # these signals barely move within a day; no need to chase them
STARTUP_DELAY = 5 * 60        # let the options scan claim Yahoo's attention first on a cold start


class IvRankJob:
    """Background loop maintaining the per-symbol signals from
    iv_rank.compute_market_signals() (IV Rank proxy, rv_30, IV/RV %ile, RSI,
    the trailing-closes window the scanner turns into a live price
    percentile) plus compute_forward_pe(). Decoupled from ScannerJob's
    manual-refresh cadence on purpose -- these change slowly and every extra
    Yahoo call is extra rate-limit risk."""

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
        """Seed `self.signals` from CACHE_FILE if present, so there's something
        to serve immediately on startup, before the first computation runs.

        A schema-version mismatch (compute_market_signals()'s return shape
        changed since this cache was written) still loads `self.signals` as a
        stale-but-better-than-nothing fallback, but leaves `self.as_of` unset
        so `_loop()` treats it as needing an immediate recompute rather than
        waiting up to REFRESH_SECONDS on a cache that merely *looks* recent."""
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
        """Persist the current signals so a restart can reuse them (see `_load_cache`)."""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_text(json.dumps(
            {"schema_version": SCHEMA_VERSION, "as_of": self.as_of, "signals": self.signals}))

    def _cache_age_seconds(self) -> float | None:
        """Seconds since the cached ranks were computed, or None if there aren't any yet."""
        if not self.as_of:
            return None
        try:
            ts = pd.Timestamp(self.as_of)
            return (pd.Timestamp.now(tz=ts.tzinfo) - ts).total_seconds()
        except Exception:
            return None

    def _run(self) -> None:
        """Recompute `self.signals` for the full universe and cache it.
        forward_pe is merged in as an extra key per symbol -- it has no
        batched fetch (see compute_forward_pe's note), so it's the slower
        half of this cycle; still fine given this job's own slow cadence."""
        try:
            symbols = load_universe()
            signals = compute_market_signals(symbols)
            for sym, pe in compute_forward_pe(symbols).items():
                signals.setdefault(sym, {})["forward_pe"] = pe
            self.signals = signals
            self.as_of = pd.Timestamp.now("UTC").isoformat(timespec="seconds")
            self._write_cache()
        except Exception:
            pass  # keep serving the last good signals; next cycle tries again

    def _loop(self) -> None:
        """Runs forever on a background thread: wait until the cache would go
        stale (or `startup_delay` on a cold start), recompute, repeat every
        `interval`. The startup delay -- rather than computing immediately --
        lets the option scan claim Yahoo's attention first."""
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
