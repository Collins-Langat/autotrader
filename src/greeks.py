"""
Greeks Calculator — Black-Scholes implementation.
Sources: Natenberg (Option Volatility & Pricing), Hull (Options, Futures, and Other Derivatives)
"""

import math
from dataclasses import dataclass
from scipy.stats import norm


@dataclass
class Greeks:
    delta: float
    gamma: float
    theta: float   # Per day
    vega: float    # Per 1% change in IV
    rho: float
    theoretical_value: float
    intrinsic_value: float
    time_value: float
    iv: float


def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes d1 parameter."""
    if T <= 0 or sigma <= 0:
        return 0.0
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes d2 parameter."""
    return d1(S, K, T, r, sigma) - sigma * math.sqrt(T)


def bs_price(option_type: str, S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes option price.
    option_type: 'call' or 'put'
    S: underlying price
    K: strike price
    T: time to expiration in years
    r: risk-free rate (decimal, e.g. 0.05 for 5%)
    sigma: implied volatility (decimal, e.g. 0.20 for 20%)
    """
    if T <= 0:
        if option_type == 'call':
            return max(S - K, 0)
        return max(K - S, 0)

    _d1 = d1(S, K, T, r, sigma)
    _d2 = d2(S, K, T, r, sigma)

    if option_type == 'call':
        return S * norm.cdf(_d1) - K * math.exp(-r * T) * norm.cdf(_d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-_d2) - S * norm.cdf(-_d1)


def calculate_greeks(
    option_type: str,
    S: float,
    K: float,
    T: float,      # years to expiration
    r: float,
    sigma: float,
    market_price: float = None
) -> Greeks:
    """
    Calculate all option Greeks using Black-Scholes.

    Parameters:
        option_type: 'call' or 'put'
        S: underlying price
        K: strike price
        T: time to expiration (years)
        r: risk-free rate (decimal)
        sigma: implied volatility (decimal)
        market_price: optional — used to compute intrinsic/time value split
    """
    if T <= 0:
        T = 1e-6

    _d1 = d1(S, K, T, r, sigma)
    _d2 = d2(S, K, T, r, sigma)
    theoretical = bs_price(option_type, S, K, T, r, sigma)

    # Delta
    if option_type == 'call':
        delta = norm.cdf(_d1)
    else:
        delta = norm.cdf(_d1) - 1  # negative for puts

    # Gamma (same for calls and puts)
    gamma = norm.pdf(_d1) / (S * sigma * math.sqrt(T))

    # Theta (per calendar day)
    theta_annual = (
        -(S * norm.pdf(_d1) * sigma) / (2 * math.sqrt(T))
        - r * K * math.exp(-r * T) * (norm.cdf(_d2) if option_type == 'call' else norm.cdf(-_d2))
    )
    if option_type == 'put':
        theta_annual += r * K * math.exp(-r * T) * norm.cdf(-_d2) - r * K * math.exp(-r * T) * norm.cdf(-_d2)
    theta_daily = theta_annual / 365

    # Vega (per 1% change in volatility)
    vega_per_point = S * norm.pdf(_d1) * math.sqrt(T)
    vega_per_pct = vega_per_point / 100  # per 1% IV change, per 1 share

    # Rho
    if option_type == 'call':
        rho = K * T * math.exp(-r * T) * norm.cdf(_d2) / 100
    else:
        rho = -K * T * math.exp(-r * T) * norm.cdf(-_d2) / 100

    # Intrinsic and time value
    if option_type == 'call':
        intrinsic = max(S - K, 0)
    else:
        intrinsic = max(K - S, 0)

    price = market_price if market_price is not None else theoretical
    time_value = price - intrinsic

    return Greeks(
        delta=round(delta, 4),
        gamma=round(gamma, 6),
        theta=round(theta_daily, 4),
        vega=round(vega_per_pct, 4),
        rho=round(rho, 4),
        theoretical_value=round(theoretical, 4),
        intrinsic_value=round(intrinsic, 4),
        time_value=round(time_value, 4),
        iv=round(sigma, 4),
    )


def implied_volatility(
    option_type: str,
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    tol: float = 1e-5,
    max_iter: int = 200
) -> float:
    """
    Newton-Raphson solver for implied volatility.
    Returns IV as a decimal (e.g., 0.25 for 25%).
    Returns None if it cannot converge.
    """
    if T <= 0 or market_price <= 0:
        return None

    # Initial guess: simple approximation
    sigma = math.sqrt(2 * math.pi / T) * market_price / S

    for _ in range(max_iter):
        price = bs_price(option_type, S, K, T, r, sigma)
        vega_val = S * norm.pdf(d1(S, K, T, r, sigma)) * math.sqrt(T)

        if abs(vega_val) < 1e-10:
            break

        diff = market_price - price
        if abs(diff) < tol:
            return round(sigma, 6)

        sigma += diff / vega_val
        if sigma <= 0:
            sigma = 0.001

    return round(sigma, 6) if sigma > 0 else None


def dte_to_years(dte: int) -> float:
    """Convert days to expiration to years."""
    return dte / 365.0


def probability_of_profit(delta: float, option_type: str) -> float:
    """
    Approximate probability of profit for a SHORT option position.
    For short calls: POP ≈ 1 - delta
    For short puts: POP ≈ 1 - abs(delta)
    """
    return round(1.0 - abs(delta), 4)


def annualized_return(credit: float, collateral: float, dte: int) -> float:
    """
    Calculate annualized return on collateral.
    Used for short put/covered call evaluation.
    """
    if collateral <= 0 or dte <= 0:
        return 0.0
    return round((credit / collateral) * (365 / dte), 4)
