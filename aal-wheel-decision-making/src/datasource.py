from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from .pricing import bs_delta, bs_greeks, bs_price

DATA_DIR = Path(__file__).resolve().parent.parent / "historical_data"
MOCK_DIR = DATA_DIR / "mock"
ORATS_DIR = DATA_DIR / "orats"
STOCK_CSV = DATA_DIR / "stock_aal.csv"

RISK_FREE = 0.02
DIV_YIELD = 0.0

STOCK_COLS = ["open", "high", "low", "close", "adj_close", "volume"]
OPTION_COLS = ["date", "expiration", "dte", "strike", "opt_type",
               "bid", "ask", "mid", "iv", "delta", "underlying"]

# ORATS EOD "strikes" schema -- one row per (tradeDate, expirDate, strike) with the
# call and put legs side by side. `delta` is the CALL delta (put delta = delta - 1).
ORATS_STRIKES_COLUMNS = [
    "ticker", "tradeDate", "expirDate", "dte", "strike", "stockPrice",
    "callVolume", "callOpenInterest", "callBidSize", "callAskSize",
    "putVolume", "putOpenInterest", "putBidSize", "putAskSize",
    "callBidPrice", "callValue", "callAskPrice", "putBidPrice", "putValue", "putAskPrice",
    "callBidIv", "callMidIv", "callAskIv", "smvVol", "putBidIv", "putMidIv", "putAskIv",
    "residualRate", "delta", "gamma", "theta", "vega", "rho", "phi", "driftlessTheta",
    "extSmvVol", "extCallValue", "extPutValue", "spotPrice", "quoteDate", "updatedAt",
    "snapShotEstTime", "snapShotDate", "expiryTod",
]


@dataclass
class MarketData:
    stock: pd.DataFrame
    options: pd.DataFrame
    name: str


class StockSource(Protocol):
    def fetch(self, start: str | None, end: str | None) -> pd.DataFrame: ...


class OptionSource(Protocol):
    def fetch(self, stock: pd.DataFrame) -> pd.DataFrame: ...


# --------------------------------------------------------------------------- #
# stock
# --------------------------------------------------------------------------- #

_STOCK_ALIASES = {
    "date": "date", "trade_date": "date", "open": "open", "adjopen": "open",
    "high": "high", "adjhigh": "high", "low": "low", "adjlow": "low",
    "close": "close", "adj_close": "adj_close", "adjclose": "adj_close",
    "adj close": "adj_close", "volume": "volume",
}


def _read_stock_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw.columns = [c.strip().lower() for c in raw.columns]
    df = raw.rename(columns={c: _STOCK_ALIASES[c] for c in raw.columns if c in _STOCK_ALIASES})
    df = df[[c for c in ["date", *STOCK_COLS] if c in df.columns]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["date"]).sort_values("date").set_index("date")
    if "adj_close" not in df:
        df["adj_close"] = df["close"]
    return df


class CsvStock:
    def __init__(self, path: str | Path = STOCK_CSV):
        self.path = Path(path)

    def fetch(self, start=None, end=None) -> pd.DataFrame:
        return _read_stock_csv(self.path)


class YahooStock:
    def __init__(self, ticker="AAL", cache: Path | None = STOCK_CSV):
        self.ticker = ticker
        self.cache = cache

    def fetch(self, start="2013-01-02", end=None) -> pd.DataFrame:
        if self.cache and self.cache.exists():
            return _read_stock_csv(self.cache)
        import yfinance as yf

        df = yf.download(self.ticker, start=start, end=end, progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.lower).rename(columns={"adj close": "adj_close"})
        df.index.name = "date"
        df = df[STOCK_COLS].dropna()
        if self.cache:
            self.cache.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.cache)
        return df


# --------------------------------------------------------------------------- #
# options -- one reader for the ORATS strikes schema (mock and real share it)
# --------------------------------------------------------------------------- #

def read_orats_strikes(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
              for p in sorted(paths)]
    raw = pd.concat(frames, ignore_index=True)
    raw.columns = [c.strip().lower() for c in raw.columns]

    date = pd.to_datetime(raw["tradedate"])
    expiration = pd.to_datetime(raw["expirdate"])
    dte = raw["dte"] if "dte" in raw else (expiration - date).dt.days
    underlying = raw["stockprice"] if "stockprice" in raw else raw.get("spotprice")
    call_delta = pd.to_numeric(raw["delta"], errors="coerce")
    greeks = {g: pd.to_numeric(raw[g], errors="coerce") for g in ("gamma", "theta", "vega")
              if g in raw}

    def leg(side: str) -> pd.DataFrame:
        p = "call" if side == "C" else "put"
        out = pd.DataFrame({
            "date": date, "expiration": expiration, "dte": dte, "strike": raw["strike"],
            "opt_type": side,
            "bid": pd.to_numeric(raw[f"{p}bidprice"], errors="coerce"),
            "ask": pd.to_numeric(raw[f"{p}askprice"], errors="coerce"),
            "mid": pd.to_numeric(raw[f"{p}value"], errors="coerce"),
            "iv": pd.to_numeric(raw[f"{p}midiv"], errors="coerce"),
            "delta": (call_delta if side == "C" else call_delta - 1.0).abs(),
            "underlying": pd.to_numeric(underlying, errors="coerce"),
        })
        for g, v in greeks.items():
            out[g] = v
        return out

    df = pd.concat([leg("C"), leg("P")], ignore_index=True)
    df = df.dropna(subset=["bid", "ask", "strike", "delta"])
    df = df[(df["dte"].between(1, 90)) & (df["bid"] >= 0)]
    return df.sort_values(["date", "expiration", "strike"]).reset_index(drop=True)


def _folder_files(directory: Path) -> list[Path]:
    return [p for p in directory.glob("*")
            if p.suffix in (".csv", ".parquet") and not p.name.startswith("_")]


def _load_folder(directory: Path) -> pd.DataFrame:
    files = _folder_files(directory)
    if not files:
        raise FileNotFoundError(f"no ORATS strikes files in {directory}")
    cache = directory / "_normalized.parquet"
    newest = max(p.stat().st_mtime for p in files)
    if cache.exists() and cache.stat().st_mtime >= newest:
        return pd.read_parquet(cache)
    df = read_orats_strikes(files)
    try:
        df.to_parquet(cache)
    except Exception:
        pass
    return df


class OratsFolder:
    def __init__(self, directory: Path = ORATS_DIR):
        self.directory = Path(directory)

    def fetch(self, stock: pd.DataFrame) -> pd.DataFrame:
        return _load_folder(self.directory)


class MockOptions:
    def __init__(self, directory: Path = MOCK_DIR, seed=7, r=RISK_FREE, q=DIV_YIELD):
        self.directory = Path(directory)
        self.seed, self.r, self.q = seed, r, q

    def fetch(self, stock: pd.DataFrame) -> pd.DataFrame:
        if not _folder_files(self.directory):
            self.directory.mkdir(parents=True, exist_ok=True)
            frame = generate_mock_orats(stock, self.r, self.q, self.seed)
            frame.to_csv(self.directory / "ORATS_strikes_AAL_synthetic.csv", index=False)
        return _load_folder(self.directory)


# --------------------------------------------------------------------------- #
# mock generator -- writes rows in the ORATS strikes schema
# --------------------------------------------------------------------------- #

def _iv_path(stock: pd.DataFrame, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ret = np.log(stock["close"]).diff().fillna(0.0).to_numpy()
    rv20 = pd.Series(ret).rolling(20).std().bfill().to_numpy() * np.sqrt(252)
    ret20 = pd.Series(ret).rolling(20).sum().bfill().to_numpy()
    iv = np.empty(len(ret))
    iv[0] = 0.35
    for i in range(1, len(ret)):
        iv[i] = iv[i - 1] + 0.06 * (0.34 - iv[i - 1]) + 0.03 * (rv20[i] - iv[i - 1])
        iv[i] += 0.15 * (0.9 * max(0.0, -ret20[i])) + 0.012 * rng.standard_normal()
        iv[i] = min(max(iv[i], 0.16), 1.8)
    return iv


def generate_mock_orats(stock: pd.DataFrame, r=RISK_FREE, q=DIV_YIELD, seed=7,
                        ticker="AAL") -> pd.DataFrame:
    atm_iv = _iv_path(stock, seed)
    close = stock["close"].to_numpy()
    dates = stock.index
    fridays = pd.date_range(dates[0], dates[-1] + pd.Timedelta(days=100), freq="W-FRI")
    skew, smile = 0.45, 0.6
    rows = []
    for di, dt in enumerate(dates):
        s, base_iv = float(close[di]), float(atm_iv[di])
        for exp in fridays[(fridays > dt) & (fridays <= dt + pd.Timedelta(days=95))]:
            dte = (exp - dt).days
            t = dte / 365.0
            strikes = np.arange(np.floor(s * 0.65), np.ceil(s * 1.35) + 0.5, 1.0)
            m = np.log(strikes / s)
            iv_k = np.clip(base_iv - skew * m + smile * m ** 2, 0.10, 3.0) * (1 + 0.05 * np.sqrt(t))
            call = bs_price(s, strikes, t, r, iv_k, "C", q)
            put = bs_price(s, strikes, t, r, iv_k, "P", q)
            cdelta = bs_delta(s, strikes, t, r, iv_k, "C", q)
            gk = bs_greeks(s, strikes, t, r, iv_k, "C", q)
            half = np.maximum(0.02, 0.06 * np.maximum(call, put) + 0.015)
            for j in range(len(strikes)):
                if not (0.02 < abs(cdelta[j]) < 0.98):
                    continue
                rows.append({
                    "ticker": ticker, "tradeDate": dt.date(), "expirDate": exp.date(),
                    "dte": dte, "strike": float(strikes[j]), "stockPrice": round(s, 4),
                    "callVolume": 0, "callOpenInterest": 0, "callBidSize": 0, "callAskSize": 0,
                    "putVolume": 0, "putOpenInterest": 0, "putBidSize": 0, "putAskSize": 0,
                    "callBidPrice": round(max(0.01, call[j] - half[j]), 3),
                    "callValue": round(float(call[j]), 3),
                    "callAskPrice": round(call[j] + half[j], 3),
                    "putBidPrice": round(max(0.01, put[j] - half[j]), 3),
                    "putValue": round(float(put[j]), 3),
                    "putAskPrice": round(put[j] + half[j], 3),
                    "callBidIv": round(iv_k[j], 4), "callMidIv": round(iv_k[j], 4),
                    "callAskIv": round(iv_k[j], 4), "smvVol": round(base_iv, 4),
                    "putBidIv": round(iv_k[j], 4), "putMidIv": round(iv_k[j], 4),
                    "putAskIv": round(iv_k[j], 4), "residualRate": 0.0,
                    "delta": round(float(cdelta[j]), 4), "gamma": round(float(gk["gamma"][j]), 5),
                    "theta": round(float(gk["theta"][j]), 4), "vega": round(float(gk["vega"][j]), 4),
                    "rho": 0.0, "phi": 0.0, "driftlessTheta": round(float(gk["theta"][j]), 4),
                    "extSmvVol": round(base_iv, 4), "extCallValue": round(float(call[j]), 3),
                    "extPutValue": round(float(put[j]), 3), "spotPrice": round(s, 4),
                    "quoteDate": dt.date(), "updatedAt": "", "snapShotEstTime": "",
                    "snapShotDate": dt.date(), "expiryTod": "PM",
                })
    return pd.DataFrame(rows, columns=ORATS_STRIKES_COLUMNS)


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #

def load_market_data(stock_source: StockSource, option_source: OptionSource,
                     start="2013-01-02", option_start="2016-01-01", end=None) -> MarketData:
    stock = stock_source.fetch(start, end)
    stock = stock.loc[start:end] if (start or end) else stock
    options = option_source.fetch(stock.loc[option_start:] if option_start else stock)
    lo, hi = stock.index.min(), stock.index.max()
    options = options[(options["date"] >= lo) & (options["date"] <= hi)].reset_index(drop=True)
    name = f"{type(stock_source).__name__}+{type(option_source).__name__}"
    return MarketData(stock=stock, options=options, name=name)


def default_sources() -> tuple[StockSource, OptionSource]:
    stock = CsvStock() if STOCK_CSV.exists() else YahooStock()
    options = OratsFolder() if _folder_files(ORATS_DIR) else MockOptions()
    return stock, options


def load_default(**kw) -> MarketData:
    stock_source, option_source = default_sources()
    return load_market_data(stock_source, option_source, **kw)
