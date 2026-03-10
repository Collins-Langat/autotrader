"""
Backtester — Simulates options strategies on historical price data.

Methodology:
  - Historical daily close prices fetched from yfinance (free, no API key)
  - Option prices simulated using Black-Scholes with rolling 20-day HV as IV proxy
  - Strike selection: analytically inverts Black-Scholes d1 to find exact 16-delta strike
  - Standard monthly expirations: 3rd Friday of the expiry month
  - Entry: ~45 DTE, 16-delta short strikes
  - Exit: 50% profit target OR 2x credit stop loss OR 21 DTE forced exit

Known limitations (documented, not hidden):
  - HV used as IV proxy — real IV is typically higher than HV (volatility risk premium)
    This means credits in the backtest are slightly understated vs reality
  - Fills at Black-Scholes mid — no bid/ask spread or slippage modeled
  - Flat vol surface — no skew modeled (real puts are more expensive than calls at same delta)
  - No early assignment risk modeled

Sources: TastyTrade research methodology, Natenberg, Sinclair
"""

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

from .greeks import bs_price, dte_to_years

# ── Constants ────────────────────────────────────────────────────────────────
RISK_FREE_RATE = 0.05
TARGET_DTE = 45
EXIT_DTE = 21
PROFIT_TARGET = 0.50        # Close at 50% of credit received
STOP_LOSS_MULTIPLE = 2.0    # Close when loss = 2x credit (buyback = 3x entry credit)
TARGET_DELTA_CALL = 0.16
TARGET_DELTA_PUT = -0.16
HV_WINDOW = 20              # Rolling window for historical volatility


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class BacktestTrade:
    ticker: str
    strategy: str
    entry_date: str
    expiry_date: str
    dte_at_entry: int
    entry_price: float          # Underlying price on entry day
    call_strike: Optional[float]
    put_strike: Optional[float]
    wing_width: Optional[float] # For iron condor
    entry_credit: float         # Total credit per share (both legs combined)
    entry_hv: float             # HV used as IV proxy at entry
    contracts: int = 1
    close_date: Optional[str] = None
    dte_at_close: Optional[int] = None
    close_debit: Optional[float] = None    # Cost to buy back per share
    pnl_per_share: Optional[float] = None  # credit - debit
    pnl_dollars: Optional[float] = None   # pnl_per_share * 100 * contracts
    close_reason: str = ''
    max_adverse_excursion: float = 0.0    # Worst intraperiod loss per share


@dataclass
class BacktestResult:
    ticker: str
    strategy: str
    start_date: str
    end_date: str
    portfolio_value: float
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)  # [(date_str, cumulative_pnl), ...]

    # Aggregate metrics (populated by compute_metrics)
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_pnl_per_trade: float = 0.0
    avg_credit: float = 0.0
    avg_winner: float = 0.0
    avg_loser: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    avg_dte_held: float = 0.0
    close_reason_breakdown: dict = field(default_factory=dict)


# ── Helper functions ─────────────────────────────────────────────────────────

def get_third_friday(year: int, month: int) -> date:
    """Return the 3rd Friday of a given year/month (standard monthly options expiry)."""
    first_day = date(year, month, 1)
    # weekday(): Monday=0 ... Friday=4
    days_to_friday = (4 - first_day.weekday()) % 7
    first_friday = first_day + timedelta(days=days_to_friday)
    return first_friday + timedelta(days=14)  # 3rd Friday


def get_target_expiry(entry: date, target_dte: int = 45) -> Optional[date]:
    """
    Find the monthly expiration (3rd Friday) closest to target_dte from entry date.
    Must be at least 21 days away (otherwise gamma risk too high at entry).
    """
    candidates = []
    for offset in range(-1, 4):  # check surrounding months
        m = entry.month + offset
        y = entry.year + (m - 1) // 12
        m = ((m - 1) % 12) + 1
        try:
            candidates.append(get_third_friday(y, m))
        except ValueError:
            pass

    # Must be a future date and at least 21 DTE
    valid = [c for c in candidates if (c - entry).days >= 21]
    if not valid:
        return None

    return min(valid, key=lambda c: abs((c - entry).days - target_dte))


def compute_rolling_hv(prices: pd.Series, window: int = HV_WINDOW) -> pd.Series:
    """Annualized rolling historical volatility from log returns."""
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.rolling(window=window).std() * math.sqrt(252)


def strike_for_target_delta(
    option_type: str,
    S: float,
    target_delta: float,
    T: float,
    r: float,
    sigma: float,
) -> float:
    """
    Analytically solve for the strike K that gives a target delta.

    For a CALL with target delta d:
        N(d1) = d  =>  d1 = N_inv(d)
        d1 = [ln(S/K) + (r + σ²/2)*T] / (σ√T)
        => K = S * exp(-d1*σ√T + (r - σ²/2)*T ... )

    Derivation:
        d1 = N_inv(target_delta) for calls
        d1 = N_inv(1 + target_delta) for puts (since put delta is negative)
        ln(S/K) = d1*σ√T - (r + σ²/2)*T
        K = S * exp(-(d1*σ√T - (r + σ²/2)*T))
    """
    if T <= 0 or sigma <= 0:
        return S

    if option_type == 'call':
        d1_target = norm.ppf(target_delta)
    else:
        # put delta = N(d1) - 1, so N(d1) = 1 + target_delta (target_delta is negative)
        d1_target = norm.ppf(1.0 + target_delta)

    # d1 = [ln(S/K) + (r + σ²/2)*T] / (σ√T)
    # => ln(S/K) = d1*σ√T - (r + σ²/2)*T
    # => K = S * exp(-(d1*σ√T - (r + σ²/2)*T))
    exponent = d1_target * sigma * math.sqrt(T) - (r + 0.5 * sigma ** 2) * T
    K = S * math.exp(-exponent)
    return round(K, 2)


def round_to_strike(K: float, price: float) -> float:
    """
    Round a theoretical strike to the nearest realistic strike increment.
    Most equity options have $1 or $2.50 or $5 increments.
    """
    if price < 25:
        increment = 0.50
    elif price < 50:
        increment = 1.0
    elif price < 200:
        increment = 2.5
    elif price < 500:
        increment = 5.0
    else:
        increment = 10.0
    return round(K / increment) * increment


# ── Main engine ───────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Runs a systematic backtest of a premium-selling options strategy
    on a single underlying over a historical date range.
    """

    def __init__(self, portfolio_value: float = 25000):
        self.portfolio_value = portfolio_value

    def run(
        self,
        ticker: str,
        strategy: str,
        start_date: str,
        end_date: str,
    ) -> BacktestResult:
        """
        Run a full backtest.

        Parameters:
            ticker: stock/ETF symbol
            strategy: 'Short Strangle', 'Iron Condor', or 'Short Put'
            start_date: 'YYYY-MM-DD'
            end_date:   'YYYY-MM-DD'
        """
        result = BacktestResult(
            ticker=ticker,
            strategy=strategy,
            start_date=start_date,
            end_date=end_date,
            portfolio_value=self.portfolio_value,
        )

        # ── 1. Fetch historical data (extra buffer for HV warmup) ──
        buffer_start = (
            pd.to_datetime(start_date) - pd.DateOffset(days=60)
        ).strftime('%Y-%m-%d')

        raw = yf.Ticker(ticker).history(start=buffer_start, end=end_date)
        if raw.empty or len(raw) < HV_WINDOW + 5:
            raise ValueError(f"Insufficient historical data for {ticker}")

        prices = raw['Close'].copy()
        prices.index = pd.to_datetime(prices.index).date  # make date objects

        # ── 2. Compute rolling HV ──
        hv_series = compute_rolling_hv(pd.Series(prices.values, index=prices.index))

        # ── 3. Get trading days in the requested range ──
        start_d = pd.to_datetime(start_date).date()
        end_d = pd.to_datetime(end_date).date()
        trading_days = [d for d in prices.index if start_d <= d <= end_d]

        if not trading_days:
            raise ValueError("No trading days in the specified date range.")

        # ── 4. Iterate through trading days ──
        open_trade: Optional[BacktestTrade] = None
        cumulative_pnl = 0.0
        result.equity_curve = []

        for today in trading_days:
            price_today = float(prices.get(today, float('nan')))
            hv_today = float(hv_series.get(today, float('nan')))

            if math.isnan(price_today) or math.isnan(hv_today) or hv_today <= 0:
                result.equity_curve.append((today.isoformat(), cumulative_pnl))
                continue

            # ── Manage open trade ──
            if open_trade is not None:
                expiry = pd.to_datetime(open_trade.expiry_date).date()
                dte_remaining = (expiry - today).days

                # Compute current option value
                current_debit, mae = self._current_position_value(
                    open_trade, price_today, dte_remaining, hv_today
                )
                open_trade.max_adverse_excursion = max(
                    open_trade.max_adverse_excursion,
                    open_trade.entry_credit - current_debit  # negative when losing
                    if current_debit > open_trade.entry_credit else 0.0
                )
                # Correctly track worst loss
                intra_pnl = open_trade.entry_credit - current_debit
                if intra_pnl < -open_trade.max_adverse_excursion:
                    open_trade.max_adverse_excursion = abs(intra_pnl)

                close_reason = self._check_exit(
                    open_trade, current_debit, dte_remaining
                )

                if close_reason:
                    open_trade = self._close_trade(
                        open_trade, today, dte_remaining,
                        current_debit, close_reason
                    )
                    cumulative_pnl += open_trade.pnl_dollars or 0.0
                    result.trades.append(open_trade)
                    open_trade = None

            # ── Try to open a new trade (if no position open) ──
            if open_trade is None:
                expiry = get_target_expiry(today, TARGET_DTE)
                if expiry is None:
                    result.equity_curve.append((today.isoformat(), cumulative_pnl))
                    continue

                dte = (expiry - today).days
                if dte < 21 or dte > 70:
                    result.equity_curve.append((today.isoformat(), cumulative_pnl))
                    continue

                trade = self._try_open_trade(
                    ticker, strategy, today, expiry, dte,
                    price_today, hv_today
                )
                if trade is not None:
                    open_trade = trade

            result.equity_curve.append((today.isoformat(), cumulative_pnl))

        # Close any trade still open at end of backtest period
        if open_trade is not None:
            last_day = trading_days[-1]
            last_price = float(prices.get(last_day, float('nan')))
            last_hv = float(hv_series.get(last_day, float('nan')))
            expiry = pd.to_datetime(open_trade.expiry_date).date()
            dte_remaining = max((expiry - last_day).days, 0)
            if not math.isnan(last_price) and not math.isnan(last_hv):
                current_debit, _ = self._current_position_value(
                    open_trade, last_price, dte_remaining, last_hv
                )
                open_trade = self._close_trade(
                    open_trade, last_day, dte_remaining,
                    current_debit, 'end_of_backtest'
                )
                cumulative_pnl += open_trade.pnl_dollars or 0.0
                result.trades.append(open_trade)

        self._compute_metrics(result)
        return result

    # ── Trade construction ────────────────────────────────────────────────────

    def _try_open_trade(
        self,
        ticker: str,
        strategy: str,
        entry_date: date,
        expiry: date,
        dte: int,
        price: float,
        hv: float,
    ) -> Optional[BacktestTrade]:
        """
        Attempt to open a new trade. Returns None if entry conditions not met.
        """
        T = dte_to_years(dte)
        r = RISK_FREE_RATE

        if strategy == 'Short Strangle':
            return self._open_strangle(ticker, entry_date, expiry, dte, price, hv, T, r)
        elif strategy == 'Iron Condor':
            return self._open_iron_condor(ticker, entry_date, expiry, dte, price, hv, T, r)
        elif strategy == 'Short Put':
            return self._open_short_put(ticker, entry_date, expiry, dte, price, hv, T, r)
        elif strategy == 'Jade Lizard':
            return self._open_jade_lizard(ticker, entry_date, expiry, dte, price, hv, T, r)
        return None

    def _open_strangle(self, ticker, entry_date, expiry, dte, price, hv, T, r):
        call_strike = round_to_strike(
            strike_for_target_delta('call', price, TARGET_DELTA_CALL, T, r, hv), price
        )
        put_strike = round_to_strike(
            strike_for_target_delta('put', price, TARGET_DELTA_PUT, T, r, hv), price
        )

        # Sanity: call must be above price, put must be below
        if call_strike <= price or put_strike >= price:
            return None

        call_credit = bs_price('call', price, call_strike, T, r, hv)
        put_credit = bs_price('put', price, put_strike, T, r, hv)
        total_credit = call_credit + put_credit

        if total_credit < 0.50:  # EN-004: minimum credit
            return None

        contracts = self._size_position_undefined(total_credit)

        return BacktestTrade(
            ticker=ticker,
            strategy='Short Strangle',
            entry_date=entry_date.isoformat(),
            expiry_date=expiry.isoformat(),
            dte_at_entry=dte,
            entry_price=price,
            call_strike=call_strike,
            put_strike=put_strike,
            wing_width=None,
            entry_credit=round(total_credit, 4),
            entry_hv=round(hv, 4),
            contracts=contracts,
        )

    def _open_iron_condor(self, ticker, entry_date, expiry, dte, price, hv, T, r):
        call_strike = round_to_strike(
            strike_for_target_delta('call', price, TARGET_DELTA_CALL, T, r, hv), price
        )
        put_strike = round_to_strike(
            strike_for_target_delta('put', price, TARGET_DELTA_PUT, T, r, hv), price
        )

        if call_strike <= price or put_strike >= price:
            return None

        # Determine wing width based on price
        wing = self._wing_width(price)

        long_call_strike = call_strike + wing
        long_put_strike = put_strike - wing

        sc = bs_price('call', price, call_strike, T, r, hv)
        sp = bs_price('put', price, put_strike, T, r, hv)
        lc = bs_price('call', price, long_call_strike, T, r, hv)
        lp = bs_price('put', price, long_put_strike, T, r, hv)

        net_credit = (sc + sp) - (lc + lp)

        # EN-004: credit must be at least 1/3 of wing width
        if net_credit < wing * 0.33:
            return None

        max_loss_per_share = wing - net_credit
        contracts = self._size_position_defined(max_loss_per_share)

        return BacktestTrade(
            ticker=ticker,
            strategy='Iron Condor',
            entry_date=entry_date.isoformat(),
            expiry_date=expiry.isoformat(),
            dte_at_entry=dte,
            entry_price=price,
            call_strike=call_strike,
            put_strike=put_strike,
            wing_width=wing,
            entry_credit=round(net_credit, 4),
            entry_hv=round(hv, 4),
            contracts=contracts,
        )

    def _open_short_put(self, ticker, entry_date, expiry, dte, price, hv, T, r):
        # 30-delta put for short put strategy
        put_strike = round_to_strike(
            strike_for_target_delta('put', price, -0.30, T, r, hv), price
        )

        if put_strike >= price:
            return None

        put_credit = bs_price('put', price, put_strike, T, r, hv)

        if put_credit < 0.25:
            return None

        # Max loss = put_strike * 100 (stock goes to zero)
        contracts = self._size_position_defined(put_strike)

        return BacktestTrade(
            ticker=ticker,
            strategy='Short Put',
            entry_date=entry_date.isoformat(),
            expiry_date=expiry.isoformat(),
            dte_at_entry=dte,
            entry_price=price,
            call_strike=None,
            put_strike=put_strike,
            wing_width=None,
            entry_credit=round(put_credit, 4),
            entry_hv=round(hv, 4),
            contracts=contracts,
        )

    def _open_jade_lizard(self, ticker, entry_date, expiry, dte, price, hv, T, r):
        """
        Jade Lizard: Short OTM Put (~30-delta) + Short OTM Call (~20-delta) + Long OTM Call.
        Key constraint: total_credit > call_spread_wing (zero upside risk).
        wing_width field stores the call spread width for later P&L computation.
        """
        put_strike  = round_to_strike(
            strike_for_target_delta('put',  price, -0.30, T, r, hv), price
        )
        call_strike = round_to_strike(
            strike_for_target_delta('call', price,  0.20, T, r, hv), price
        )

        if put_strike >= price or call_strike <= price:
            return None

        jade_wing = self._jade_wing_width(price)
        long_call_strike = call_strike + jade_wing

        sp = bs_price('put',  price, put_strike,       T, r, hv)
        sc = bs_price('call', price, call_strike,      T, r, hv)
        lc = bs_price('call', price, long_call_strike, T, r, hv)

        total_credit = sp + sc - lc

        # KEY RULE: credit must exceed call spread width for zero upside risk
        if total_credit <= jade_wing or total_credit < 0.50:
            return None

        # Sized as undefined risk (only the put side is naked)
        contracts = self._size_position_undefined(total_credit)

        return BacktestTrade(
            ticker=ticker,
            strategy='Jade Lizard',
            entry_date=entry_date.isoformat(),
            expiry_date=expiry.isoformat(),
            dte_at_entry=dte,
            entry_price=price,
            call_strike=call_strike,
            put_strike=put_strike,
            wing_width=jade_wing,       # call spread width
            entry_credit=round(total_credit, 4),
            entry_hv=round(hv, 4),
            contracts=contracts,
        )

    # ── Position valuation ─────────────────────────────────────────────────────

    def _current_position_value(
        self,
        trade: BacktestTrade,
        current_price: float,
        dte_remaining: int,
        current_hv: float,
    ) -> tuple[float, float]:
        """
        Compute the current cost to close (buy back) the position.
        Returns (current_debit_per_share, intraday_max_adverse_excursion).
        At expiration (dte=0), returns intrinsic value.
        """
        T = max(dte_to_years(dte_remaining), 1e-6)
        r = RISK_FREE_RATE
        sigma = max(current_hv, 0.05)  # floor at 5% to avoid math errors
        total = 0.0

        if trade.call_strike:
            call_val = bs_price('call', current_price, trade.call_strike, T, r, sigma)
            total += max(call_val, 0.0)

        if trade.put_strike:
            put_val = bs_price('put', current_price, trade.put_strike, T, r, sigma)
            total += max(put_val, 0.0)

        # For iron condor: subtract long wing values (they offset)
        if trade.strategy == 'Iron Condor' and trade.wing_width:
            lc_strike = trade.call_strike + trade.wing_width
            lp_strike = trade.put_strike - trade.wing_width
            lc_val = bs_price('call', current_price, lc_strike, T, r, sigma)
            lp_val = bs_price('put', current_price, lp_strike, T, r, sigma)
            total -= max(lc_val + lp_val, 0.0)
            # Net debit to close can't exceed wing width (max loss is capped)
            if trade.wing_width:
                total = min(total, trade.wing_width)

        # For jade lizard: subtract long call value only (no long put side)
        # Call spread debit is capped at wing_width; put side is unlimited
        if trade.strategy == 'Jade Lizard' and trade.wing_width and trade.call_strike:
            lc_strike = trade.call_strike + trade.wing_width
            lc_val = bs_price('call', current_price, lc_strike, T, r, sigma)
            total -= max(lc_val, 0.0)
            # Cap the call spread component at wing_width (max loss on call side)
            call_spread_debit = max(0.0, (
                bs_price('call', current_price, trade.call_strike, T, r, sigma) - lc_val
            ))
            call_spread_debit = min(call_spread_debit, trade.wing_width)

        # Clamp to 0 if long wing offsets more than short (can happen at very low values)
        total = max(total, 0.0)

        mae = max(0.0, total - trade.entry_credit)
        return round(total, 4), round(mae, 4)

    # ── Exit logic ────────────────────────────────────────────────────────────

    def _check_exit(
        self,
        trade: BacktestTrade,
        current_debit: float,
        dte_remaining: int,
    ) -> Optional[str]:
        """
        Check all exit conditions. Returns exit reason string, or None if staying.
        """
        pnl_per_share = trade.entry_credit - current_debit

        # EX-001: 50% profit target
        if pnl_per_share >= trade.entry_credit * PROFIT_TARGET:
            return 'profit_target'

        # HR-003 / EX-002: 2x credit stop loss
        # Loss of 2x credit means P&L = -2*credit
        # => current_debit = entry_credit + 2*entry_credit = 3*entry_credit
        if pnl_per_share <= -(STOP_LOSS_MULTIPLE * trade.entry_credit):
            return 'stop_loss'

        # HR-004 / EX-003: 21 DTE forced exit
        if dte_remaining <= EXIT_DTE:
            return 'dte_exit'

        # Expiration: close at intrinsic if at/past expiry
        if dte_remaining <= 0:
            return 'expired'

        return None

    def _close_trade(
        self,
        trade: BacktestTrade,
        close_date: date,
        dte_remaining: int,
        close_debit: float,
        close_reason: str,
    ) -> BacktestTrade:
        """Record the closing details on a trade."""
        trade.close_date = close_date.isoformat()
        trade.dte_at_close = dte_remaining
        trade.close_debit = round(close_debit, 4)
        trade.pnl_per_share = round(trade.entry_credit - close_debit, 4)
        trade.pnl_dollars = round(
            trade.pnl_per_share * 100 * trade.contracts, 2
        )
        trade.close_reason = close_reason
        return trade

    # ── Position sizing ───────────────────────────────────────────────────────

    def _size_position_undefined(self, credit: float) -> int:
        """Size undefined-risk positions (strangles): max 2% portfolio notional."""
        max_risk = self.portfolio_value * 0.02
        contracts = int(max_risk / (credit * 100)) if credit > 0 else 1
        return max(1, min(contracts, 5))

    def _size_position_defined(self, max_loss_per_share: float) -> int:
        """Size defined-risk positions: max 2% portfolio loss."""
        max_risk = self.portfolio_value * 0.02
        contracts = int(max_risk / (max_loss_per_share * 100)) if max_loss_per_share > 0 else 1
        return max(1, min(contracts, 10))

    def _wing_width(self, price: float) -> float:
        """Determine iron condor wing width from underlying price."""
        if price < 50:
            return 2.5
        elif price < 200:
            return 5.0
        elif price < 500:
            return 10.0
        else:
            return 25.0

    def _jade_wing_width(self, price: float) -> float:
        """
        Jade Lizard call spread wing — tighter than iron condor wing.
        Must be narrow enough for total_credit > wing to hold.
        """
        if price < 50:
            return 1.0
        elif price < 200:
            return 2.5
        elif price < 500:
            return 5.0
        else:
            return 10.0

    # ── Metrics computation ───────────────────────────────────────────────────

    def _compute_metrics(self, result: BacktestResult):
        """Populate aggregate performance metrics on the BacktestResult."""
        trades = [t for t in result.trades if t.pnl_dollars is not None]
        result.total_trades = len(trades)

        if result.total_trades == 0:
            return

        pnls = [t.pnl_dollars for t in trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p < 0]

        result.winning_trades = len(winners)
        result.losing_trades = len(losers)
        result.breakeven_trades = result.total_trades - len(winners) - len(losers)
        result.win_rate = round(len(winners) / result.total_trades, 4)
        result.total_pnl = round(sum(pnls), 2)
        result.avg_pnl_per_trade = round(result.total_pnl / result.total_trades, 2)
        result.avg_credit = round(
            sum(t.entry_credit for t in trades) / len(trades), 4
        )
        result.avg_winner = round(sum(winners) / len(winners), 2) if winners else 0.0
        result.avg_loser = round(sum(losers) / len(losers), 2) if losers else 0.0
        result.avg_dte_held = round(
            sum(
                (t.dte_at_entry - (t.dte_at_close or 0))
                for t in trades
            ) / len(trades), 1
        )

        # Max drawdown from equity curve
        curve_vals = [v for _, v in result.equity_curve]
        if curve_vals:
            peak = curve_vals[0]
            max_dd = 0.0
            for v in curve_vals:
                peak = max(peak, v)
                dd = peak - v
                max_dd = max(max_dd, dd)
            result.max_drawdown = round(max_dd, 2)

        # Profit factor = gross profit / gross loss
        gross_profit = sum(winners) if winners else 0
        gross_loss = abs(sum(losers)) if losers else 0
        result.profit_factor = round(
            gross_profit / gross_loss if gross_loss > 0 else float('inf'), 2
        )

        # Sharpe ratio (annualized, assuming 252 trading days/year)
        if len(pnls) > 1:
            pnl_array = np.array(pnls)
            avg = np.mean(pnl_array)
            std = np.std(pnl_array, ddof=1)
            # Approximate number of trades per year
            try:
                days_span = (
                    pd.to_datetime(result.end_date) - pd.to_datetime(result.start_date)
                ).days
                trades_per_year = result.total_trades / (days_span / 365) if days_span > 0 else 12
            except Exception:
                trades_per_year = 12
            result.sharpe_ratio = round(
                (avg / std) * math.sqrt(trades_per_year) if std > 0 else 0.0, 2
            )
        else:
            result.sharpe_ratio = 0.0

        # Close reason breakdown
        from collections import Counter
        result.close_reason_breakdown = dict(
            Counter(t.close_reason for t in trades)
        )
