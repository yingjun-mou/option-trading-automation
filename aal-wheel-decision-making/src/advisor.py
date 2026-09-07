from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .csp.leg import csp_ladder_target, csp_signal
from .cc.leg import cc_ladder_target, cc_signal
from .engine import Chain, Position, WheelAccount
from .features import _rsi
from .realtime import Quote
from .wheel import WheelConfig, build_legs

ROLL_DTE = 7                 # inside this many days to expiry -> roll candidate
ROLL_ITM_DTE = 21           # ITM and getting close -> roll candidate
CONSIDER_CLOSE_AT = 0.50    # % of max profit that flags "consider closing" when the rule holds to expiry


# --------------------------------------------------------------------------- #
# portfolio state (mock now; Robinhood positions later)
# --------------------------------------------------------------------------- #

@dataclass
class OpenOption:
    opt_type: str            # "P" | "C"
    expiration: str
    strike: float
    contracts: int
    entry_price: float       # credit received per share
    entry_date: str = ""
    delta: float = 0.0
    iv: float = 0.0


@dataclass
class PortfolioState:
    cash: float = 100_000.0
    shares: int = 0
    avg_share_cost: float = 0.0
    csp_premium_since_flat: float = 0.0
    waiting_for_dip: bool = False
    positions: list[OpenOption] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "PortfolioState":
        if not Path(path).exists():
            return cls()
        raw = json.loads(Path(path).read_text())
        raw["positions"] = [OpenOption(**p) for p in raw.get("positions", [])]
        return cls(**raw)


# --------------------------------------------------------------------------- #
# recommendation model
# --------------------------------------------------------------------------- #

@dataclass
class Recommendation:
    action: str              # SELL_CSP | SELL_CC | WAIT | CLOSE_CSP | CLOSE_CC | ROLL_CSP | ROLL_CC | CONSIDER_CLOSE_*
    priority: int            # lower = more urgent
    headline: str
    rationale: str
    params: dict = field(default_factory=dict)


@dataclass
class Advice:
    as_of: str
    symbol: str
    spot: float
    prev_close: float
    change_pct: float
    data_source: str
    signals: dict
    portfolio: dict
    primary: dict
    recommendations: list
    position_reviews: list


# --------------------------------------------------------------------------- #
# live features (same definitions as features.build_features, single latest row)
# --------------------------------------------------------------------------- #

def compute_live_features(hist_close: pd.Series, hist_iv30: pd.Series, spot: float,
                          atm_iv: float, as_of) -> dict:
    close = pd.concat([hist_close, pd.Series({pd.Timestamp(as_of.date()): spot})])
    logret = np.log(close).diff()
    w1, w3 = close.tail(252), close.tail(3 * 252)
    rv20 = float(logret.tail(20).std() * np.sqrt(252))
    iv_hist_1y = hist_iv30.dropna().tail(252)
    return dict(
        close=float(spot),
        pct_1y=float((spot >= w1).mean()),
        pct_3y=float((spot >= w3).mean()),
        roll_low_3y=float(w3.min()), roll_high_3y=float(w3.max()),
        dist_from_low_3y=float(spot / w3.min() - 1.0),
        dist_from_high_3y=float(spot / w3.max() - 1.0),
        ret_20=float(spot / close.iloc[-21] - 1.0) if len(close) > 21 else np.nan,
        ret_60=float(spot / close.iloc[-61] - 1.0) if len(close) > 61 else np.nan,
        rsi_14=float(_rsi(close).iloc[-1]),
        rv_20=rv20,
        rv_30=float(logret.tail(30).std() * np.sqrt(252)),
        iv_30=float(atm_iv),
        iv_pct_1y=float((atm_iv >= iv_hist_1y).mean()) if len(iv_hist_1y) else np.nan,
        iv_minus_rv20=float(atm_iv - rv20),
    )


# --------------------------------------------------------------------------- #
# advice
# --------------------------------------------------------------------------- #

def _seed_account(cfg: WheelConfig, pf: PortfolioState, spot: float, today: pd.Timestamp) -> WheelAccount:
    acct = WheelAccount(cfg)
    acct.cash = pf.cash
    acct.shares = pf.shares
    acct.stock_cost = pf.shares * pf.avg_share_cost
    acct.csp_premium_since_flat = pf.csp_premium_since_flat
    acct.waiting_for_dip = pf.waiting_for_dip
    for p in pf.positions:
        net = p.entry_price * 100 * p.contracts
        pt = cfg.csp_profit_take if p.opt_type == "P" else cfg.cc_profit_take
        acct.positions.append(Position(
            p.opt_type, pd.Timestamp(p.expiration), float(p.strike), p.contracts,
            pd.Timestamp(p.entry_date or today), p.entry_price, net, spot,
            p.delta, p.iv, pt))
    return acct


def _mark(pos: Position, chain: Chain, spot: float, today: pd.Timestamp) -> float:
    q = chain.quote(today, pos.expiration, pos.strike, pos.opt_type)
    if q is not None:
        return q["mid"]
    intr = (pos.strike - spot) if pos.opt_type == "P" else (spot - pos.strike)
    return max(0.0, intr)


def _review(pos: Position, chain: Chain, spot: float, today: pd.Timestamp) -> dict:
    mark = _mark(pos, chain, spot, today)
    qty = pos.contracts * 100
    dte = max(0, (pos.expiration - today).days)
    itm = pos.strike > spot if pos.opt_type == "P" else spot > pos.strike
    max_profit = pos.entry_gross_ps * qty
    cost_to_close = mark * qty
    open_pnl = pos.entry_net - cost_to_close
    return dict(
        kind="CSP" if pos.opt_type == "P" else "CC",
        opt_type=pos.opt_type, strike=pos.strike,
        expiration=str(pos.expiration.date()), dte=dte, contracts=pos.contracts,
        itm=bool(itm), mark=round(mark, 2), cost_to_close=round(cost_to_close, 0),
        entry_credit=round(pos.entry_net, 0), open_pnl=round(open_pnl, 0),
        pct_max_profit=round(open_pnl / max_profit, 3) if max_profit > 0 else 0.0,
    )


def _sell_params(trade: dict, cfg: WheelConfig, pf: PortfolioState, spot: float) -> dict:
    strike = trade["strike"]
    n = trade["contracts"]
    dte = trade.get("dte")
    premium = round(trade["cash_flow"], 0)
    if trade["opt_type"] == "P":
        collateral = strike * 100 * n
        max_budget = cfg.max_alloc_pct * cfg.portfolio_size
        yield_ann = (premium / collateral) * (365 / dte) if dte else np.nan
        return dict(strike=strike, expiration=str(pd.Timestamp(trade["expiration"]).date()),
                    dte=dte, delta=trade.get("delta"), contracts=n,
                    collateral=round(collateral, 0),
                    allocation_pct=round(collateral / max_budget, 3),
                    est_premium=premium, iv=trade.get("iv"),
                    annualized_yield_on_collateral=round(float(yield_ann), 3))
    covered = n * 100
    return dict(strike=strike, expiration=str(pd.Timestamp(trade["expiration"]).date()),
                dte=dte, delta=trade.get("delta"), contracts=n,
                shares_covered=covered,
                pct_shares_covered=round(covered / max(pf.shares, 1), 3),
                strike_vs_cost=round(strike - pf.avg_share_cost, 2),
                est_premium=premium, iv=trade.get("iv"))


def _wait_reason(cfg: WheelConfig, feats: dict, spot: float, acct: WheelAccount) -> str:
    sig = csp_signal(cfg, feats, spot)
    edge0 = cfg.csp_edges[0]
    unit = "$" if cfg.csp_signal == "abs" else ""
    val = f"{unit}{sig:.2f}" if cfg.csp_signal == "abs" else f"{sig:.0%} percentile"
    reentry = edge0 if cfg.csp_signal == "abs" else cfg.csp_reentry_edge

    if acct.waiting_for_dip:
        return (f"Post-call-away: holding cash until AAL drops back below "
                f"{unit}{reentry}. Currently {val}.")
    target = csp_ladder_target(sig, cfg.csp_edges, cfg.csp_tranches)
    if target == 0.0:
        return (f"AAL at {val} is above the CSP entry ladder (starts at {unit}{edge0}). "
                f"No new puts -- wait for a cheaper price.")
    if cfg.iv_pct_min is not None and (np.isnan(feats.get("iv_pct_1y", np.nan))
                                       or feats["iv_pct_1y"] < cfg.iv_pct_min):
        return (f"Price is in the entry zone but IV percentile "
                f"{feats.get('iv_pct_1y', float('nan')):.0%} is below the required "
                f"{cfg.iv_pct_min:.0%}. Wait for richer premium.")
    committed = acct.put_collateral + acct.stock_cost
    budget = target * cfg.max_alloc_pct * cfg.portfolio_size
    if committed >= budget - 1000:
        return (f"CSP allocation already at the ladder target for this price "
                f"(~{committed / cfg.portfolio_size:.0%} of the portfolio). Hold.")
    return "No action triggered by the rules right now."


def advise(cfg: WheelConfig, pf: PortfolioState, feats: dict, chain_df: pd.DataFrame,
           quote: Quote, hist_iv30_last: float | None = None) -> Advice:
    chain = Chain(chain_df)
    spot = quote.price
    today = pd.Timestamp(quote.timestamp.date())
    acct = _seed_account(cfg, pf, spot, today)

    recs: list[Recommendation] = []
    reviews: list[dict] = []

    for pos in acct.positions:
        rv = _review(pos, chain, spot, today)
        reviews.append(rv)
        kind = rv["kind"]
        if rv["dte"] <= ROLL_DTE or (rv["itm"] and rv["dte"] <= ROLL_ITM_DTE):
            recs.append(Recommendation(
                f"ROLL_{kind}", 1,
                f"Roll {kind} {rv['strike']:g} exp {rv['expiration']}",
                f"{rv['dte']} DTE and {'ITM' if rv['itm'] else 'OTM'}. Assignment risk is "
                f"rising. TODO: roll strike/expiration selection is not implemented yet -- "
                f"decide manually or close.",
                params=rv))
            continue
        pt = cfg.csp_profit_take if pos.opt_type == "P" else cfg.cc_profit_take
        captured = rv["pct_max_profit"]
        if pt < 1.0 and captured >= pt:
            recs.append(Recommendation(
                f"CLOSE_{kind}", 2,
                f"Buy to close {kind} {rv['strike']:g} exp {rv['expiration']}",
                f"Captured {captured:.0%} of max profit (rule: close at {pt:.0%}). "
                f"Est. cost to close ${rv['cost_to_close']:,.0f}, locking ${rv['open_pnl']:,.0f}.",
                params=rv))
        elif captured >= CONSIDER_CLOSE_AT:
            recs.append(Recommendation(
                f"CONSIDER_CLOSE_{kind}", 4,
                f"Optionally close {kind} {rv['strike']:g} (up {captured:.0%})",
                f"Captured {captured:.0%} of max profit. The rule holds to expiry, so this "
                f"is optional risk reduction, not required.",
                params=rv))

    dry = _seed_account(cfg, pf, spot, today)
    for leg in build_legs(cfg):
        leg.rebalance(dry, chain, today, spot, feats)
    for t in dry.trades:
        if t["action"] == "SELL_PUT":
            p = _sell_params(t, cfg, pf, spot)
            recs.append(Recommendation(
                "SELL_CSP", 3,
                f"Sell {p['contracts']}x CSP {p['strike']:g}P exp {p['expiration']} "
                f"({p['dte']} DTE, ~{p['delta']:.2f}delta)",
                f"AAL at {csp_signal(cfg, feats, spot):.2f} is in the entry ladder. Deploy "
                f"~{p['allocation_pct']:.0%} of the max allocation "
                f"(${p['collateral']:,.0f} collateral) for ~${p['est_premium']:,.0f} premium "
                f"({p['annualized_yield_on_collateral']:.0%} annualised).",
                params=p))
        elif t["action"] == "SELL_CALL":
            p = _sell_params(t, cfg, pf, spot)
            recs.append(Recommendation(
                "SELL_CC", 3,
                f"Sell {p['contracts']}x CC {p['strike']:g}C exp {p['expiration']} "
                f"({p['dte']} DTE, ~{p['delta']:.2f}delta)",
                f"AAL has recovered into the CC ladder. Cover {p['pct_shares_covered']:.0%} "
                f"of the {pf.shares} shares for ~${p['est_premium']:,.0f}; strike is "
                f"${p['strike_vs_cost']:+.2f} vs cost basis.",
                params=p))

    if not recs:
        recs.append(Recommendation("WAIT", 5, "Wait -- no trade",
                                   _wait_reason(cfg, feats, spot, acct)))

    recs.sort(key=lambda r: r.priority)
    prev = quote.prev_close or spot
    return Advice(
        as_of=quote.timestamp.isoformat(timespec="seconds"),
        symbol=quote.symbol, spot=spot, prev_close=prev,
        change_pct=round((spot / prev - 1.0) * 100, 2) if prev else 0.0,
        data_source=quote.source,
        signals={k: (round(v, 4) if isinstance(v, float) and not np.isnan(v) else v)
                 for k, v in feats.items()},
        portfolio=dict(cash=pf.cash, shares=pf.shares, avg_share_cost=pf.avg_share_cost,
                       open_positions=[asdict(p) for p in pf.positions],
                       waiting_for_dip=pf.waiting_for_dip),
        primary=asdict(recs[0]),
        recommendations=[asdict(r) for r in recs],
        position_reviews=reviews,
    )
