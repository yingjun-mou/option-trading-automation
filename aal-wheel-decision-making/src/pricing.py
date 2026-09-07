from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _d1_d2(s, k, t, r, sigma, q=0.0):
    s = np.asarray(s, dtype=float)
    k = np.asarray(k, dtype=float)
    t = np.maximum(np.asarray(t, dtype=float), 1e-6)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-6)
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    return d1, d1 - sigma * np.sqrt(t)


def bs_price(s, k, t, r, sigma, opt_type, q=0.0):
    d1, d2 = _d1_d2(s, k, t, r, sigma, q)
    disc, divd = np.exp(-r * t), np.exp(-q * t)
    if opt_type == "C":
        return s * divd * norm.cdf(d1) - k * disc * norm.cdf(d2)
    return k * disc * norm.cdf(-d2) - s * divd * norm.cdf(-d1)


def bs_delta(s, k, t, r, sigma, opt_type, q=0.0):
    d1, _ = _d1_d2(s, k, t, r, sigma, q)
    divd = np.exp(-q * t)
    return divd * norm.cdf(d1) if opt_type == "C" else divd * (norm.cdf(d1) - 1.0)


def bs_greeks(s, k, t, r, sigma, opt_type, q=0.0):
    d1, d2 = _d1_d2(s, k, t, r, sigma, q)
    t = np.maximum(np.asarray(t, dtype=float), 1e-6)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-6)
    divd, disc, pdf = np.exp(-q * t), np.exp(-r * t), norm.pdf(d1)
    gamma = divd * pdf / (s * sigma * np.sqrt(t))
    vega = s * divd * pdf * np.sqrt(t) / 100.0
    if opt_type == "C":
        delta = divd * norm.cdf(d1)
        theta = (-s * divd * pdf * sigma / (2 * np.sqrt(t))
                 - r * k * disc * norm.cdf(d2) + q * s * divd * norm.cdf(d1)) / 365.0
    else:
        delta = divd * (norm.cdf(d1) - 1.0)
        theta = (-s * divd * pdf * sigma / (2 * np.sqrt(t))
                 + r * k * disc * norm.cdf(-d2) - q * s * divd * norm.cdf(-d1)) / 365.0
    return dict(delta=delta, gamma=gamma, vega=vega, theta=theta)


def implied_vol(price, s, k, t, r, opt_type, q=0.0, lo=1e-3, hi=5.0, tol=1e-5):
    intrinsic = max(0.0, (s - k) if opt_type == "C" else (k - s)) * np.exp(-q * t)
    if price <= intrinsic + 1e-6:
        return np.nan
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        val = bs_price(s, k, t, r, mid, opt_type, q)
        if abs(val - price) < tol:
            return mid
        hi, lo = (mid, lo) if val > price else (hi, mid)
    return mid
