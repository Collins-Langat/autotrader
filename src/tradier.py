"""
Tradier API Client
==================
Wraps Tradier's REST API for options chain data, real-time quotes,
historical prices, and (later) order execution.

Phase A  -- Data only: options chain with real greeks + real IV
Phase B  -- Paper trading: place orders against Tradier sandbox
Phase C  -- Live trading: flip use_sandbox=False

Sandbox base URL : https://sandbox.tradier.com/v1/
Live base URL    : https://api.tradier.com/v1/

All methods return plain dicts / DataFrames so the rest of the
codebase doesn't need to know whether data came from Tradier or yfinance.
"""

import math
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests

# ── Config loader ─────────────────────────────────────────────────────────────

def _load_config() -> dict:
    """Load config.yaml if present, return empty dict on any failure."""
    try:
        import yaml, os
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'config.yaml'
        )
        if not os.path.exists(config_path):
            return {}
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


# ── TradierClient ─────────────────────────────────────────────────────────────

class TradierClient:
    """
    Lightweight Tradier REST client.

    Usage:
        client = TradierClient()          # reads config.yaml automatically
        client = TradierClient(token='...', sandbox=True)   # explicit
    """

    SANDBOX_URL = 'https://sandbox.tradier.com/v1'
    LIVE_URL    = 'https://api.tradier.com/v1'

    def __init__(
        self,
        token:   Optional[str] = None,
        sandbox: Optional[bool] = None,
    ):
        cfg = _load_config().get('tradier', {})

        self.enabled  = cfg.get('enabled', False)
        use_sandbox   = sandbox if sandbox is not None else cfg.get('use_sandbox', True)

        if token:
            self.token = token
        else:
            if use_sandbox:
                self.token = cfg.get('sandbox_token', '')
            else:
                self.token = cfg.get('live_token', '')

        self.base_url = self.SANDBOX_URL if use_sandbox else self.LIVE_URL
        self.sandbox  = use_sandbox

        if not self.token:
            self.enabled = False

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get(self, endpoint: str, params: dict = None) -> dict:
        """Make a GET request to the Tradier API. Returns parsed JSON."""
        if not self.enabled or not self.token:
            raise RuntimeError('Tradier client is not enabled or token is missing.')

        url = f'{self.base_url}/{endpoint}'
        headers = {
            'Authorization': f'Bearer {self.token}',
            'Accept': 'application/json',
        }
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def is_available(self) -> bool:
        """True if the client is configured and has a token."""
        return bool(self.enabled and self.token)

    # ── Market data ───────────────────────────────────────────────────────────

    def get_quote(self, symbol: str) -> dict:
        """
        Fetch a real-time quote for a single symbol.
        Returns: price, bid, ask, volume, open, high, low, close, change_pct
        """
        data = self._get('markets/quotes', params={'symbols': symbol, 'greeks': 'false'})
        quotes = data.get('quotes', {}).get('quote', {})

        # Tradier returns a list when multiple symbols; dict for single
        if isinstance(quotes, list):
            quotes = quotes[0] if quotes else {}

        return {
            'ticker':      symbol.upper(),
            'price':       float(quotes.get('last') or quotes.get('close') or 0),
            'bid':         float(quotes.get('bid') or 0),
            'ask':         float(quotes.get('ask') or 0),
            'volume':      int(quotes.get('volume') or 0),
            'open':        float(quotes.get('open') or 0),
            'high':        float(quotes.get('high') or 0),
            'low':         float(quotes.get('low') or 0),
            'prev_close':  float(quotes.get('prevclose') or 0),
            'change_pct':  float(quotes.get('change_percentage') or 0),
        }

    def get_options_expirations(self, symbol: str) -> list:
        """
        Return list of available expiration date strings (YYYY-MM-DD).
        """
        data = self._get(
            'markets/options/expirations',
            params={'symbol': symbol, 'includeAllRoots': 'false'},
        )
        exps = data.get('expirations', {}) or {}
        dates = exps.get('date', [])
        if isinstance(dates, str):
            dates = [dates]
        return sorted(dates)

    def get_options_chain(
        self,
        symbol:     str,
        expiration: str,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch a full options chain for one expiration, including Tradier greeks.

        Returns a DataFrame with columns:
            strike, option_type, bid, ask, mid, last, volume, openInterest,
            impliedVolatility (mid_iv), delta, gamma, theta, vega,
            dte, expiration, spread_pct, liquidity_score
        """
        try:
            data = self._get(
                'markets/options/chains',
                params={
                    'symbol':     symbol,
                    'expiration': expiration,
                    'greeks':     'true',
                },
            )
        except Exception:
            return None

        options = data.get('options', {}) or {}
        option_list = options.get('option', [])
        if not option_list:
            return None
        if isinstance(option_list, dict):
            option_list = [option_list]

        rows = []
        today = datetime.now().date()
        exp_date = datetime.strptime(expiration, '%Y-%m-%d').date()
        dte = (exp_date - today).days

        for opt in option_list:
            greeks = opt.get('greeks') or {}

            bid  = float(opt.get('bid')  or 0)
            ask  = float(opt.get('ask')  or 0)
            mid  = (bid + ask) / 2

            # Tradier uses 'option_type': 'call' | 'put'
            opt_type = str(opt.get('option_type', '')).lower()

            # Real implied volatility from Tradier greeks
            mid_iv = greeks.get('mid_iv') or greeks.get('smv_vol') or 0
            try:
                mid_iv = float(mid_iv)
            except (TypeError, ValueError):
                mid_iv = 0.0

            rows.append({
                'contractSymbol':    opt.get('symbol', ''),
                'strike':            float(opt.get('strike') or 0),
                'option_type':       opt_type,
                'bid':               bid,
                'ask':               ask,
                'mid':               round(mid, 4),
                'last':              float(opt.get('last') or 0),
                'volume':            int(opt.get('volume') or 0),
                'openInterest':      int(opt.get('open_interest') or 0),
                'impliedVolatility': mid_iv,          # REAL IV from Tradier
                'delta':             float(greeks.get('delta') or 0),
                'gamma':             float(greeks.get('gamma') or 0),
                'theta':             float(greeks.get('theta') or 0),
                'vega':              float(greeks.get('vega')  or 0),
                'dte':               dte,
                'expiration':        expiration,
            })

        if not rows:
            return None

        df = pd.DataFrame(rows)

        # ── Liquidity enrichment (same as yfinance path) ──────────────────────
        spread    = (df['ask'] - df['bid']).clip(lower=0)
        mid_safe  = df['mid'].replace(0, float('nan'))
        df['spread_pct'] = (spread / mid_safe).fillna(1.0).clip(0, 1)

        oi_score  = (df['openInterest'].clip(0, 1000) / 1000 * 40).round(1)
        vol_score = (df['volume'].clip(0, 500)         / 500  * 30).round(1)
        spd_score = ((1 - df['spread_pct'])            * 30).round(1)
        df['liquidity_score'] = (oi_score + vol_score + spd_score).clip(0, 100).round(1)

        return df

    def get_historical_prices(
        self,
        symbol:   str,
        start:    str,          # YYYY-MM-DD
        end:      str = None,   # YYYY-MM-DD, defaults to today
        interval: str = 'daily',
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV history from Tradier.
        Returns DataFrame indexed by date with Open/High/Low/Close/Volume columns.
        """
        if end is None:
            end = datetime.now().strftime('%Y-%m-%d')
        try:
            data = self._get(
                'markets/history',
                params={
                    'symbol':   symbol,
                    'interval': interval,
                    'start':    start,
                    'end':      end,
                },
            )
        except Exception:
            return None

        history = data.get('history', {}) or {}
        days    = history.get('day', [])
        if not days:
            return None
        if isinstance(days, dict):
            days = [days]

        df = pd.DataFrame(days)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        df = df.rename(columns={
            'open': 'Open', 'high': 'High',
            'low':  'Low',  'close': 'Close', 'volume': 'Volume',
        })
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    # ── IV helpers ────────────────────────────────────────────────────────────

    def get_atm_iv(self, symbol: str, target_dte: int = 45) -> Optional[float]:
        """
        Return the ATM implied volatility (mid_iv) for the expiration
        nearest to target_dte. Uses real Tradier greeks.

        Returns IV as a decimal (e.g., 0.28 = 28% IV).
        """
        try:
            expirations = self.get_options_expirations(symbol)
            if not expirations:
                return None

            # Find expiry closest to target_dte
            today = datetime.now().date()
            best_exp = None
            best_diff = float('inf')
            for exp_str in expirations:
                exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
                dte = (exp_date - today).days
                if dte >= 7:
                    diff = abs(dte - target_dte)
                    if diff < best_diff:
                        best_diff = diff
                        best_exp = exp_str

            if not best_exp:
                return None

            chain = self.get_options_chain(symbol, best_exp)
            if chain is None or chain.empty:
                return None

            # Get current price for ATM selection
            quote = self.get_quote(symbol)
            price = quote.get('price', 0)
            if price <= 0:
                return None

            # Find nearest strike to current price
            chain['strike_diff'] = (chain['strike'] - price).abs()
            nearest_strike = chain['strike_diff'].min()
            atm = chain[chain['strike_diff'] <= nearest_strike + 0.01]

            # Average IV of ATM call and put
            atm_ivs = atm['impliedVolatility'].replace(0, float('nan')).dropna()
            if atm_ivs.empty:
                return None

            return round(float(atm_ivs.mean()), 4)

        except Exception:
            return None

    def get_options_chain_near_dte(
        self,
        symbol:     str,
        target_dte: int = 45,
        apply_liquidity_filter: bool = True,
        expiry:     str = None,
    ) -> Optional[pd.DataFrame]:
        """
        High-level wrapper: get options chain for expiry nearest to target_dte.
        Drop-in replacement for market_data.get_options_chain().
        """
        try:
            expirations = self.get_options_expirations(symbol)
            if not expirations:
                return None

            today = datetime.now().date()

            if expiry and expiry in expirations:
                best_exp = expiry
            else:
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

            chain = self.get_options_chain(symbol, best_exp)
            if chain is None or chain.empty:
                return None

            if apply_liquidity_filter:
                mask = (
                    (chain['openInterest'] >= 100) &
                    (chain['spread_pct']   <= 0.20) &
                    (chain['mid']          >  0.01)
                )
                filtered = chain[mask]
                calls_ok = (filtered['option_type'] == 'call').sum() >= 4
                puts_ok  = (filtered['option_type'] == 'put').sum()  >= 4
                if calls_ok and puts_ok:
                    chain = filtered

            return chain

        except Exception:
            return None


# ── Module-level singleton ────────────────────────────────────────────────────

_client: Optional[TradierClient] = None

def get_client() -> TradierClient:
    """Return the module-level TradierClient singleton (lazy init)."""
    global _client
    if _client is None:
        _client = TradierClient()
    return _client


def is_tradier_enabled() -> bool:
    """Quick check: is Tradier configured and ready to use?"""
    return get_client().is_available()
