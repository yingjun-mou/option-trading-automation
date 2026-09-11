"""Builds the scanner's stock universe: S&P 500 (Wikipedia) union the 500
largest Nasdaq-listed stocks by market cap (Nasdaq screener API).

Not run automatically -- composition of "500 largest" and S&P 500 membership
change slowly, unlike option premiums. Re-run this occasionally (monthly is
plenty); the live scanner (src/scanner.py) reads whatever list it wrote last.

    python scripts/build_universe.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.realtime import REALTIME_DIR  # noqa: E402

OUT_FILE = REALTIME_DIR / "universe.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
NON_COMMON_MARKERS = ("Warrant", "Right", "Unit", "Preferred", "Note", "Debenture")


def fetch_sp500() -> set[str]:
    r = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                      headers=HEADERS, timeout=20)
    r.raise_for_status()
    table = pd.read_html(io.StringIO(r.text))[0]
    return {s.strip().upper().replace(".", "-") for s in table["Symbol"]}


def fetch_top_nasdaq(n: int = 500) -> set[str]:
    url = ("https://api.nasdaq.com/api/screener/stocks"
           "?tableonly=true&limit=8000&exchange=NASDAQ&download=true")
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    rows = r.json()["data"]["rows"]
    df = pd.DataFrame(rows)
    df["marketCap"] = pd.to_numeric(df["marketCap"], errors="coerce").fillna(0.0)
    is_common = ~df["name"].str.contains("|".join(NON_COMMON_MARKERS), case=False, na=False)
    is_common &= df["symbol"].str.match(r"^[A-Z]{1,5}$", na=False)
    df = df[is_common & (df["marketCap"] > 0)]
    top = df.sort_values("marketCap", ascending=False).head(n)
    return set(top["symbol"].str.upper())


def main() -> None:
    sp500 = fetch_sp500()
    print(f"S&P 500: {len(sp500)} tickers")
    nasdaq_top = fetch_top_nasdaq(500)
    print(f"Top 500 Nasdaq by market cap: {len(nasdaq_top)} tickers")
    universe = sorted(sp500 | nasdaq_top)
    print(f"Union (deduped): {len(universe)} tickers")

    REALTIME_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps({
        "built_at": pd.Timestamp.now("UTC").isoformat(timespec="seconds"),
        "sources": ["S&P 500 (Wikipedia)", "Top 500 Nasdaq-listed by market cap (Nasdaq screener)"],
        "count": len(universe),
        "tickers": universe,
    }, indent=2))
    print(f"Wrote {OUT_FILE}")


if __name__ == "__main__":
    main()
