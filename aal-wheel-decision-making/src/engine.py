from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

MIN_TICKET = 1_000.0


class Chain:
    def __init__(self, options: pd.DataFrame):
        opt = options.sort_values("date")
        typ = opt["opt_type"].to_numpy().astype("U1")
        cols = {
            "exp": opt["expiration"].to_numpy().astype("datetime64[ns]"),
            "exp_i": opt["expiration"].to_numpy().astype("datetime64[ns]").astype("int64"),
            "dte": opt["dte"].to_numpy().astype(float),
            "strike": opt["strike"].to_numpy().astype(float),
            "typ": typ,
            "bid": opt["bid"].to_numpy().astype(float),
            "ask": opt["ask"].to_numpy().astype(float),
            "mid": opt["mid"].to_numpy().astype(float),
            "iv": opt["iv"].to_numpy().astype(float),
            "delta": opt["delta"].to_numpy().astype(float),
        }
        dates = opt["date"].to_numpy().astype("datetime64[ns]")
        uniq, starts = np.unique(dates, return_index=True)
        bounds = list(starts) + [len(dates)]
        self._by_date = {pd.Timestamp(d): {c: v[bounds[i]:bounds[i + 1]] for c, v in cols.items()}
                         for i, d in enumerate(uniq)}

    def has(self, date) -> bool:
        return pd.Timestamp(date) in self._by_date

    def select(self, date, opt_type, target_dte, target_delta, dte_tol,
               min_strike=None, max_strike=None):
        d = self._by_date.get(pd.Timestamp(date))
        if d is None:
            return None
        m = d["typ"] == opt_type
        if min_strike is not None:
            m &= d["strike"] >= min_strike
        if max_strike is not None:
            m &= d["strike"] <= max_strike
        if not m.any():
            return None
        gap = np.abs(d["dte"] - target_dte)
        near = m & (gap <= dte_tol)
        pool = near if near.any() else (m & (gap == gap[m].min()))
        idx = np.nonzero(pool)[0]
        return self._row(date, d, idx[np.argmin(np.abs(d["delta"][idx] - target_delta))])

    def quote(self, date, expiration, strike, opt_type):
        d = self._by_date.get(pd.Timestamp(date))
        if d is None:
            return None
        m = ((d["exp_i"] == np.int64(pd.Timestamp(expiration).value))
             & (d["strike"] == float(strike)) & (d["typ"] == opt_type))
        idx = np.nonzero(m)[0]
        return self._row(date, d, int(idx[0])) if len(idx) else None

    @staticmethod
    def _row(date, d, j):
        return dict(date=pd.Timestamp(date), expiration=pd.Timestamp(d["exp"][j]),
                    dte=float(d["dte"][j]), strike=float(d["strike"][j]),
                    opt_type=str(d["typ"][j]), bid=float(d["bid"][j]), ask=float(d["ask"][j]),
                    mid=float(d["mid"][j]), iv=float(d["iv"][j]), delta=float(d["delta"][j]))


@dataclass
class Position:
    opt_type: str
    expiration: pd.Timestamp
    strike: float
    contracts: int
    entry_date: pd.Timestamp
    entry_gross_ps: float
    entry_net: float
    entry_underlying: float
    entry_delta: float
    entry_iv: float
    profit_take: float = 1.0


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    daily: pd.DataFrame
    summary: dict
    config: object


class Leg(Protocol):
    def rebalance(self, account: "WheelAccount", chain: Chain, date, spot: float,
                  frow: pd.Series) -> None: ...


class WheelAccount:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cash = cfg.portfolio_size
        self.shares = 0
        self.stock_cost = 0.0
        self.csp_premium_since_flat = 0.0
        self.realized_stock_pnl = 0.0
        self.premium = {"P": 0.0, "C": 0.0}
        self.positions: list[Position] = []
        self.waiting_for_dip = False
        self.counts: Counter = Counter()
        self.trades: list[dict] = []

    @property
    def put_collateral(self) -> float:
        return sum(p.strike * 100 * p.contracts for p in self.positions if p.opt_type == "P")

    @property
    def covered_shares(self) -> int:
        return sum(p.contracts * 100 for p in self.positions if p.opt_type == "C")

    @property
    def effective_basis(self) -> float:
        if self.shares <= 0:
            return np.nan
        return (self.stock_cost - self.csp_premium_since_flat) / self.shares

    def buy_shares(self, quantity: int, price: float):
        self.cash -= quantity * price
        self.shares += quantity
        self.stock_cost += quantity * price

    def _fill_sell(self, q) -> float:
        base = q["bid"] if self.cfg.sell_fill == "bid" else q["mid"]
        return max(0.01, base - self.cfg.slippage)

    def _fill_buy(self, q) -> float:
        base = q["ask"] if self.cfg.buy_fill == "ask" else q["mid"]
        return max(0.01, base + self.cfg.slippage)

    def _log(self, **row):
        self.trades.append(row)

    def sell(self, q, contracts, date, spot, signal, profit_take=1.0):
        comm = self.cfg.commission_per_contract * contracts
        net = self._fill_sell(q) * 100 * contracts - comm
        self.cash += net
        self.premium[q["opt_type"]] += net
        if q["opt_type"] == "P":
            self.csp_premium_since_flat += net
            self.counts["csp"] += 1
        else:
            self.counts["cc"] += 1
        self.positions.append(Position(q["opt_type"], q["expiration"], q["strike"], contracts,
                                       date, q["mid"], net, spot, q["delta"], q["iv"], profit_take))
        self._log(date=date, action="SELL_PUT" if q["opt_type"] == "P" else "SELL_CALL",
                  opt_type=q["opt_type"], strike=q["strike"], contracts=contracts,
                  underlying=spot, cash_flow=net, expiration=q["expiration"],
                  dte=int(q["dte"]), delta=round(q["delta"], 3), entry_mid=q["mid"],
                  iv=round(q["iv"], 4), signal=round(signal, 3),
                  note=f"dte{int(q['dte'])} d{q['delta']:.2f} sig{signal:.2f}")

    def close(self, pos, q, date, spot):
        cost = self._fill_buy(q) * 100 * pos.contracts + self.cfg.commission_per_contract * pos.contracts
        self.cash -= cost
        self.premium[pos.opt_type] -= cost
        if pos.opt_type == "P":
            self.csp_premium_since_flat -= cost
            self.counts["csp_closed"] += 1
        else:
            self.counts["cc_closed"] += 1
        self.positions.remove(pos)
        self._log(date=date, action=f"CLOSE_{'PUT' if pos.opt_type == 'P' else 'CALL'}",
                  opt_type=pos.opt_type, strike=pos.strike, contracts=pos.contracts,
                  underlying=spot, cash_flow=-cost, note="profit_take")

    def _assign_put(self, pos, date, spot):
        cost = pos.strike * pos.contracts * 100
        self.cash -= cost
        self.shares += pos.contracts * 100
        self.stock_cost += cost
        self.counts["assign"] += 1
        self.positions.remove(pos)
        self._log(date=date, action="ASSIGN_PUT", opt_type="P", strike=pos.strike,
                  contracts=pos.contracts, underlying=spot, cash_flow=-cost,
                  note=f"basis~{self.effective_basis:.2f}")

    def _exercise_call(self, pos, date, spot):
        qty = min(pos.contracts * 100, self.shares)
        avg = self.stock_cost / self.shares
        self.realized_stock_pnl += (pos.strike - avg) * qty
        self.cash += pos.strike * qty
        self.stock_cost -= avg * qty
        self.shares -= qty
        self.counts["callaway"] += 1
        self.positions.remove(pos)
        self._log(date=date, action="CALL_AWAY", opt_type="C", strike=pos.strike,
                  contracts=pos.contracts, underlying=spot, cash_flow=pos.strike * qty,
                  note=f"stk_pnl~{(pos.strike - avg) * qty:.0f}")
        if self.shares == 0:
            self.waiting_for_dip = True
            self.csp_premium_since_flat = 0.0
            self.stock_cost = 0.0

    def _expire(self, pos, date, spot):
        self.positions.remove(pos)
        self._log(date=date, action=f"EXPIRE_{'PUT' if pos.opt_type == 'P' else 'CALL'}",
                  opt_type=pos.opt_type, strike=pos.strike, contracts=pos.contracts,
                  underlying=spot, cash_flow=0.0, note="worthless")

    def settle_expirations(self, date, spot):
        for pos in list(self.positions):
            if pos.expiration > date:
                continue
            if pos.opt_type == "P":
                (self._assign_put if pos.strike > spot else self._expire)(pos, date, spot)
            elif spot > pos.strike and self.shares > 0:
                self._exercise_call(pos, date, spot)
            else:
                self._expire(pos, date, spot)

    def take_profits(self, chain, date, spot):
        for pos in list(self.positions):
            if pos.profit_take >= 1.0:
                continue
            q = chain.quote(date, pos.expiration, pos.strike, pos.opt_type)
            if q is not None and self._fill_buy(q) <= (1.0 - pos.profit_take) * pos.entry_gross_ps:
                self.close(pos, q, date, spot)

    def option_liability(self, chain, date, spot) -> float:
        total = 0.0
        for pos in self.positions:
            q = chain.quote(date, pos.expiration, pos.strike, pos.opt_type)
            if q is not None:
                total += q["mid"] * 100 * pos.contracts
            else:
                intr = (pos.strike - spot) if pos.opt_type == "P" else (spot - pos.strike)
                total += max(0.0, intr) * 100 * pos.contracts
        return total


def run_strategy(legs: list[Leg], features: pd.DataFrame, options: pd.DataFrame, cfg,
                 start=None, end=None, chain: Chain | None = None,
                 initial_alloc: float = 0.0) -> BacktestResult:
    feat = features
    if start:
        feat = feat.loc[pd.Timestamp(start):]
    if end:
        feat = feat.loc[:pd.Timestamp(end)]
    chain = chain or Chain(options)
    acct = WheelAccount(cfg)

    if initial_alloc > 0:
        first_spot = float(feat.iloc[0]["close"])
        qty = int(initial_alloc * cfg.portfolio_size // (first_spot * 100)) * 100
        if qty:
            acct.buy_shares(qty, first_spot)

    snaps = []
    for date in feat.index:
        frow = feat.loc[date]
        spot = float(frow["close"])
        acct.settle_expirations(date, spot)
        acct.take_profits(chain, date, spot)
        for leg in legs:
            leg.rebalance(acct, chain, date, spot, frow)

        liability = acct.option_liability(chain, date, spot)
        snaps.append(dict(date=date, price=spot, cash=acct.cash, shares=acct.shares,
                          opt_liability=liability, put_collateral=acct.put_collateral,
                          n_positions=len(acct.positions),
                          equity=acct.cash + acct.shares * spot - liability,
                          deployed=acct.shares * spot + acct.put_collateral))

    daily = pd.DataFrame(snaps).set_index("date")
    equity = daily["equity"].rename("equity")
    trades = pd.DataFrame(acct.trades)
    c = acct.counts
    summary = dict(
        label=cfg.label, end_value=float(equity.iloc[-1]),
        premium_csp=acct.premium["P"], premium_cc=acct.premium["C"],
        realized_stock_pnl=acct.realized_stock_pnl,
        n_csp=c["csp"], n_cc=c["cc"], n_assign=c["assign"], n_callaway=c["callaway"],
        n_csp_closed_early=c["csp_closed"], n_cc_closed_early=c["cc_closed"],
        pct_days_holding_shares=float((daily["shares"] > 0).mean()),
        pct_days_idle=float(((daily["shares"] == 0) & (daily["n_positions"] == 0)).mean()),
        avg_capital_utilization=float((daily["deployed"] / cfg.portfolio_size).clip(upper=2).mean()),
        worst_single_cashflow=float(trades["cash_flow"].min()) if len(trades) else np.nan,
        final_shares=int(acct.shares),
    )
    return BacktestResult(equity=equity, trades=trades, daily=daily, summary=summary, config=cfg)
