"""
Market Data Module
==================
Fetches live/historical data with a two-tier data strategy:

  Tier 1 (preferred) — Tradier API
      Real implied volatility, real greeks, real-time quotes.
      Activated by setting tradier.enabled=true in config.yaml.

  Tier 2 (fallback) — yfinance
      Free, no API key required.  Used automatically when Tradier
      is not configured or a Tradier request fails.

Public API is unchanged — all callers (decision_engine, backtester,
long_evaluator, stock_screener) work without modification.
"""

import warnings
import math
from datetime import datetime, timedelta
from typing import Optional
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

# ── Tradier integration (optional) ────────────────────────────────────────────
try:
    from src.tradier import get_client as _get_tradier, is_tradier_enabled
    _TRADIER_AVAILABLE = True
except ImportError:
    _TRADIER_AVAILABLE = False
    def is_tradier_enabled() -> bool:
        return False


def get_underlying_data(ticker: str) -> dict:
    """
    Fetch current underlying data for a ticker.
    Returns price, volume, 52-week range, and basic info.

    Price source: Tradier (real-time) if enabled, else yfinance.
    Fundamental info (sector, market_cap) always from yfinance.
    """
    # ── Fundamentals from yfinance (always) ───────────────────────────────────
    stock = yf.Ticker(ticker)
    info  = stock.info
    hist  = stock.history(period="1d")

    if hist.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    yf_price = float(hist['Close'].iloc[-1])
    yf_vol   = int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else 0

    # ── Price override from Tradier (real-time) ───────────────────────────────
    current_price = yf_price
    volume        = yf_vol
    data_source   = 'yfinance'

    if _TRADIER_AVAILABLE and is_tradier_enabled():
        try:
            q = _get_tradier().get_quote(ticker)
            if q.get('price', 0) > 0:
                current_price = q['price']
                volume        = q.get('volume', yf_vol)
                data_source   = 'tradier'
        except Exception:
            pass  # silent fallback to yfinance

    return {
        'ticker':      ticker,
        'price':       round(current_price, 2),
        'volume':      volume,
        'market_cap':  info.get('marketCap', None),
        'sector':      info.get('sector', 'Unknown'),
        'industry':    info.get('industry', 'Unknown'),
        'timestamp':   datetime.now().isoformat(),
        'data_source': data_source,
    }


def get_historical_volatility(ticker: str, window: int = 20) -> dict:
    """
    Calculate historical (realized) volatility.
    Returns HV for multiple windows: 10, 20, 30, 60 day.
    """
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")

    if len(hist) < window:
        raise ValueError(f"Insufficient history for {ticker}")

    log_returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()

    results = {}
    for w in [10, 20, 30, 60]:
        if len(log_returns) >= w:
            hv = log_returns.rolling(window=w).std().iloc[-1] * math.sqrt(252)
            results[f'hv{w}'] = round(float(hv), 4)
        else:
            results[f'hv{w}'] = None

    results['ticker'] = ticker
    return results


# Broad market ETFs where VIX is the authoritative IV measure.
# For these we use VIX 52-week history instead of HV-proxy for IV rank.
_VIX_LINKED_TICKERS = {
    'SPY', 'IVV', 'VOO', 'SPLG',           # S&P 500
    'QQQ', 'QQQM', 'TQQQ',                  # Nasdaq
    'IWM', 'TNA',                            # Russell 2000
    'DIA',                                   # Dow
    'MDY', 'IJH', 'VO',                      # Mid-cap
}


def get_iv_rank(ticker: str, current_iv: float) -> dict:
    """
    Calculate IV rank and IV percentile.

    When Tradier is enabled, the current_iv anchor is replaced with
    Tradier's real ATM mid_iv — significantly more accurate than
    yfinance's reverse-engineered estimate.

    Two rank methods depending on ticker type:

    A. Broad market ETFs (SPY, QQQ, IWM, etc.):
       Uses VIX 52-week history with 5th/95th percentile winsorization.

    B. Individual stocks:
       Uses HV+VRP scaling with 10th/90th percentile winsorization.
       When Tradier is enabled, current_iv is the real ATM IV.
    """
    # ── Upgrade current_iv anchor with Tradier real IV if available ───────────
    iv_source = 'estimated'
    if _TRADIER_AVAILABLE and is_tradier_enabled():
        try:
            real_iv = _get_tradier().get_atm_iv(ticker, target_dte=45)
            if real_iv and real_iv > 0:
                current_iv = real_iv
                iv_source  = 'tradier'
        except Exception:
            pass  # keep the caller-supplied estimate

    if ticker.upper() in _VIX_LINKED_TICKERS:
        result = _iv_rank_via_vix(current_iv)
    else:
        result = _iv_rank_via_hv_vrp(ticker, current_iv)

    result['iv_source'] = iv_source
    return result


def _iv_rank_via_vix(current_iv: float) -> dict:
    """
    IV rank for broad market ETFs using VIX 52-week history.
    Winsorizes at 5th/95th percentile so single spike events (e.g., a
    crash) don't collapse the rank of everything below them to near zero.
    """
    vix_ticker = yf.Ticker('^VIX')
    vix_hist = vix_ticker.history(period='1y')

    if vix_hist.empty:
        return {'iv_rank': 50, 'iv_percentile': 50, 'current_iv': round(current_iv, 4)}

    vix_series = vix_hist['Close'] / 100.0
    current_vix = float(vix_series.iloc[-1])

    lo = float(vix_series.quantile(0.05))
    hi = float(vix_series.quantile(0.95))

    if hi <= lo:
        return {'iv_rank': 50, 'iv_percentile': 50, 'current_iv': round(current_iv, 4)}

    iv_rank = (current_vix - lo) / (hi - lo) * 100
    iv_rank = max(0, min(100, iv_rank))
    iv_percentile = float((vix_series < current_vix).mean() * 100)

    # Approximate HV: VIX / typical_VRP for SPY (1.3x long-run average)
    approx_hv = current_vix / 1.3

    return {
        'iv_rank': round(iv_rank, 1),
        'iv_percentile': round(iv_percentile, 1),
        'current_iv': round(current_iv, 4),     # ATM IV from options chain
        'current_hv': round(approx_hv, 4),
        'vrp_ratio': round(current_vix / approx_hv, 2),
        'iv_52w_high': round(hi, 4),
        'iv_52w_low': round(lo, 4),
        'method': 'vix_52w_winsorized',
    }


def _iv_rank_via_hv_vrp(ticker: str, current_iv: float) -> dict:
    """
    IV rank for individual stocks via VRP-adjusted HV with winsorization.
    Uses 10th/90th percentile to trim extreme crash/spike events.
    """
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")

    if hist.empty:
        return {'iv_rank': None, 'iv_percentile': None, 'note': 'No historical data'}

    log_returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
    rolling_hv = log_returns.rolling(window=20).std() * math.sqrt(252)
    rolling_hv = rolling_hv.dropna()

    if len(rolling_hv) < 20:
        return {'iv_rank': None, 'iv_percentile': None}

    current_hv = float(rolling_hv.iloc[-1])

    if current_hv > 0:
        vrp_ratio = current_iv / current_hv
        vrp_ratio = max(0.5, min(vrp_ratio, 3.0))
    else:
        vrp_ratio = 1.2

    estimated_iv_history = rolling_hv * vrp_ratio

    # Winsorize at 10/90 for stocks (wider than ETFs — individual stocks
    # have more extreme idiosyncratic events to exclude)
    lo = float(estimated_iv_history.quantile(0.10))
    hi = float(estimated_iv_history.quantile(0.90))

    if hi <= lo:
        return {
            'iv_rank': 50, 'iv_percentile': 50,
            'current_iv': round(current_iv, 4), 'current_hv': round(current_hv, 4),
        }

    iv_rank = (current_iv - lo) / (hi - lo) * 100
    iv_rank = max(0, min(100, iv_rank))
    iv_percentile = float((estimated_iv_history < current_iv).mean() * 100)

    return {
        'iv_rank': round(iv_rank, 1),
        'iv_percentile': round(iv_percentile, 1),
        'current_iv': round(current_iv, 4),
        'current_hv': round(current_hv, 4),
        'vrp_ratio': round(vrp_ratio, 2),
        'iv_52w_high': round(hi, 4),
        'iv_52w_low': round(lo, 4),
        'method': 'hv_vrp_winsorized',
    }


def get_options_chain(
    ticker: str,
    target_dte: int = 45,
    expiry: str = None,
    apply_liquidity_filter: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Fetch an options chain with real greeks and IV.

    Data source priority:
      1. Tradier  — real IV (mid_iv), real greeks (delta/gamma/theta/vega)
      2. yfinance — fallback, IV is estimated by Yahoo (less accurate)

    If `expiry` is given (YYYY-MM-DD string), that exact expiration is fetched.
    Otherwise, the expiration closest to `target_dte` days is selected.
    Returns a DataFrame with calls and puts, enriched with DTE and mid price.
    """
    # ── Tier 1: Tradier ───────────────────────────────────────────────────────
    if _TRADIER_AVAILABLE and is_tradier_enabled():
        try:
            chain = _get_tradier().get_options_chain_near_dte(
                symbol=ticker,
                target_dte=target_dte,
                apply_liquidity_filter=apply_liquidity_filter,
                expiry=expiry,
            )
            if chain is not None and not chain.empty:
                chain['data_source'] = 'tradier'
                return chain
        except Exception:
            pass  # fall through to yfinance

    # ── Tier 2: yfinance fallback ─────────────────────────────────────────────
    stock = yf.Ticker(ticker)
    expirations = stock.options

    if not expirations:
        return None

    today = datetime.now().date()

    if expiry:
        # Use the requested expiry if available; fall back to nearest if not listed
        best_exp = expiry if expiry in expirations else None
        if best_exp is None:
            # Expired or delisted — find nearest
            best_exp = min(
                expirations,
                key=lambda e: abs(
                    (datetime.strptime(e, '%Y-%m-%d').date() - today).days
                ),
            )
    else:
        # Find expiration closest to target DTE
        best_exp = None
        best_diff = float('inf')
        for exp_str in expirations:
            exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
            dte = (exp_date - today).days
            if dte >= 0:
                diff = abs(dte - target_dte)
                if diff < best_diff:
                    best_diff = diff
                    best_exp = exp_str

    if not best_exp:
        return None

    chain = stock.option_chain(best_exp)
    calls = chain.calls.copy()
    puts = chain.puts.copy()

    exp_date = datetime.strptime(best_exp, '%Y-%m-%d').date()
    dte = (exp_date - today).days

    calls['option_type'] = 'call'
    calls['dte'] = dte
    calls['expiration'] = best_exp

    puts['option_type'] = 'put'
    puts['dte'] = dte
    puts['expiration'] = best_exp

    chain_df = pd.concat([calls, puts], ignore_index=True)
    chain_df['mid'] = (chain_df['bid'] + chain_df['ask']) / 2

    # ── Liquidity enrichment ─────────────────────────────────────────────────
    chain_df['openInterest'] = pd.to_numeric(
        chain_df.get('openInterest', 0), errors='coerce'
    ).fillna(0)
    chain_df['volume'] = pd.to_numeric(
        chain_df.get('volume', 0), errors='coerce'
    ).fillna(0)

    spread = (chain_df['ask'] - chain_df['bid']).clip(lower=0)
    mid_safe = chain_df['mid'].replace(0, float('nan'))
    chain_df['spread_pct'] = (spread / mid_safe).fillna(1.0).clip(0, 1)

    # Liquidity score 0-100: OI (40 pts) + volume (30 pts) + tight spread (30 pts)
    oi_score  = (chain_df['openInterest'].clip(0, 1000) / 1000 * 40).round(1)
    vol_score = (chain_df['volume'].clip(0, 500)        / 500  * 30).round(1)
    spd_score = ((1 - chain_df['spread_pct'])           * 30).round(1)
    chain_df['liquidity_score'] = (oi_score + vol_score + spd_score).clip(0, 100).round(1)

    # ── HR-007 liquidity filter (applied by default, skip for paper re-pricing) ──
    if apply_liquidity_filter:
        mask = (
            (chain_df['openInterest'] >= 100) &
            (chain_df['spread_pct']   <= 0.20) &
            (chain_df['mid']          >  0.01)
        )
        filtered = chain_df[mask]
        # Only use filtered chain if we keep at least 8 strikes per side
        calls_ok = (filtered['option_type'] == 'call').sum() >= 4
        puts_ok  = (filtered['option_type'] == 'put').sum() >= 4
        if calls_ok and puts_ok:
            chain_df = filtered

    chain_df['data_source'] = 'yfinance'
    return chain_df


def get_vix() -> float:
    """Fetch current VIX level."""
    vix = yf.Ticker('^VIX')
    hist = vix.history(period='1d')
    if hist.empty:
        return 20.0  # fallback
    return round(float(hist['Close'].iloc[-1]), 2)


def get_earnings_date(ticker: str) -> dict:
    """
    Fetch the next earnings date for a ticker.
    Returns:
        next_earnings_date: date or None
        days_until_earnings: int or None
        has_earnings_in_window: bool (True if earnings fall within a 45-day holding window)
        source: where the date came from
    """
    stock = yf.Ticker(ticker)
    today = datetime.now().date()
    next_date = None

    # Method 1: stock.calendar (most reliable when available)
    try:
        cal = stock.calendar
        if cal is not None and not cal.empty:
            # calendar is a DataFrame with dates as columns in some yfinance versions
            if hasattr(cal, 'columns') and 'Earnings Date' in cal.columns:
                val = cal['Earnings Date'].iloc[0]
                if hasattr(val, 'date'):
                    next_date = val.date()
                elif isinstance(val, str):
                    next_date = datetime.strptime(val[:10], '%Y-%m-%d').date()
            # Sometimes it's a dict-like object
            elif hasattr(cal, 'loc') and 'Earnings Date' in cal.index:
                val = cal.loc['Earnings Date']
                if hasattr(val, 'iloc'):
                    val = val.iloc[0]
                if hasattr(val, 'date'):
                    next_date = val.date()
    except Exception:
        pass

    # Method 2: earnings_dates DataFrame (has recent + upcoming)
    if next_date is None:
        try:
            ed = stock.earnings_dates
            if ed is not None and not ed.empty:
                # earnings_dates index is a DatetimeIndex
                future_dates = [
                    d.date() for d in ed.index
                    if hasattr(d, 'date') and d.date() >= today
                ]
                if future_dates:
                    next_date = min(future_dates)
        except Exception:
            pass

    # Method 3: quarterly earnings from income statement timestamps
    if next_date is None:
        try:
            info = stock.info
            # earningsTimestamp is a Unix timestamp
            ts = info.get('earningsTimestamp') or info.get('earningsTimestampStart')
            if ts:
                candidate = datetime.fromtimestamp(ts).date()
                if candidate >= today:
                    next_date = candidate
        except Exception:
            pass

    # Build result
    if next_date is None:
        return {
            'next_earnings_date': None,
            'days_until_earnings': None,
            'has_earnings_in_window': False,
            'earnings_within_expiration_7d': False,
            'source': 'not_found',
        }

    days_until = (next_date - today).days
    # HR-001: block if earnings fall within a 45→21 DTE holding window
    # i.e., earnings are between today and 45 days from now
    has_earnings_in_window = 0 <= days_until <= 45

    return {
        'next_earnings_date': next_date.isoformat(),
        'days_until_earnings': days_until,
        'has_earnings_in_window': has_earnings_in_window,
        'earnings_within_expiration_7d': 0 <= days_until <= 7,
        'source': 'yfinance',
    }


def screen_universe(tickers: list, min_price: float = 15, min_volume: int = 500000) -> list:
    """
    Quick screen of a list of tickers for basic liquidity criteria.
    Returns list of tickers that pass.
    """
    passed = []
    for ticker in tickers:
        try:
            data = yf.Ticker(ticker)
            hist = data.history(period='5d')
            if hist.empty:
                continue
            price = float(hist['Close'].iloc[-1])
            vol = float(hist['Volume'].mean())
            if price >= min_price and vol >= min_volume:
                passed.append(ticker)
        except Exception:
            continue
    return passed
