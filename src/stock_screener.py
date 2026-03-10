"""
Stock Value Screener
Identifies potentially undervalued stocks by combining:
  - Fundamental analysis (analyst targets, P/E, earnings growth)
  - Technical analysis (RSI-14, 52-week low proximity)
  - Options context (IV rank — cheap options on beaten-down stock = buy call opportunity)

Usage:
    from src.stock_screener import StockScreener
    screener = StockScreener()
    result = screener.screen('AAPL')
    results = screener.scan_watchlist(['AAPL', 'MSFT', 'AMD'])
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from src.market_data import get_iv_rank


# ── Score thresholds ───────────────────────────────────────────────────────────
STRONG_SIGNAL  = 75
WATCH_LIST     = 55
NEUTRAL        = 35

# Options action IV-rank thresholds
BUY_CALL_MAX_IV    = 35   # iv_rank <= 35 => "Buy Call"
DEBIT_SPREAD_MAX_IV = 50  # iv_rank 36-50 => "Debit Spread"


@dataclass
class StockValueEval:
    """Full evaluation result for one ticker."""

    ticker: str
    price: float = 0.0

    # Fundamental data
    analyst_target: float = 0.0          # mean analyst price target
    discount_to_target_pct: float = 0.0  # how far below target (positive = below)
    trailing_pe: float = 0.0
    forward_pe: float = 0.0
    earnings_growth: float = 0.0         # yoy earnings growth (0.12 = 12%)
    profit_margin: float = 0.0
    analyst_rating: float = 0.0          # 1=Strong Buy ... 5=Sell (lower = better)

    # Technical data
    week52_low: float = 0.0
    week52_high: float = 0.0
    pct_above_52w_low: float = 0.0       # 0% = at 52-week low, 100% = at 52-week high
    rsi_14: float = 50.0

    # Scores
    fundamental_score: float = 0.0      # 0-50
    technical_score: float = 0.0        # 0-50
    total_score: float = 0.0            # 0-100
    score_label: str = ''               # Strong Value Signal / Watch List / Neutral / No Signal

    # Plain-English explanation
    plain_english: str = ''

    # Options linkage
    iv_rank: float = 0.0
    options_action: str = 'wait'         # 'buy_call' | 'debit_spread' | 'wait'
    options_reason: str = ''

    # Errors / data gaps
    data_missing: list = field(default_factory=list)   # which fields were unavailable
    error: Optional[str] = None


class StockScreener:
    """
    Screens stocks for undervaluation using a 100-point scoring system:
      - 50 pts fundamental (analyst target discount, P/E, earnings growth)
      - 50 pts technical  (52-week low proximity, RSI-14 oversold)

    When a stock scores >= 55 AND options are cheap (low IV rank), it
    automatically suggests buying a call or debit spread.
    """

    def screen(self, ticker: str) -> StockValueEval:
        """
        Full screen of a single ticker.
        Returns a StockValueEval (even on partial data — degrades gracefully).
        """
        ev = StockValueEval(ticker=ticker.upper())

        try:
            stock = yf.Ticker(ticker)

            # ── Price history (60 days for RSI + HV) ─────────────────────────
            hist = stock.history(period='3mo')
            if hist.empty or len(hist) < 14:
                ev.error = f"Insufficient price history for {ticker}"
                return ev

            ev.price = round(float(hist['Close'].iloc[-1]), 2)

            # ── yfinance info dict ────────────────────────────────────────────
            info = stock.info or {}

            # ── Fundamental fields ────────────────────────────────────────────
            ev.analyst_target  = float(info.get('targetMeanPrice') or 0)
            ev.trailing_pe     = float(info.get('trailingPE') or 0)
            ev.forward_pe      = float(info.get('forwardPE') or 0)
            ev.earnings_growth = float(info.get('earningsGrowth') or 0)
            ev.profit_margin   = float(info.get('profitMargins') or 0)
            ev.analyst_rating  = float(info.get('recommendationMean') or 3.0)

            if ev.analyst_target > 0 and ev.price > 0:
                ev.discount_to_target_pct = round(
                    (ev.analyst_target - ev.price) / ev.analyst_target * 100, 1
                )

            # Track missing fundamentals
            if ev.analyst_target == 0:
                ev.data_missing.append('analyst_target')
            if ev.forward_pe == 0:
                ev.data_missing.append('forward_pe')
            if ev.earnings_growth == 0:
                ev.data_missing.append('earnings_growth')

            # ── Technical fields ──────────────────────────────────────────────
            ev.week52_low  = float(info.get('fiftyTwoWeekLow')  or hist['Low'].min())
            ev.week52_high = float(info.get('fiftyTwoWeekHigh') or hist['High'].max())

            if ev.week52_high > ev.week52_low:
                ev.pct_above_52w_low = round(
                    (ev.price - ev.week52_low) / (ev.week52_high - ev.week52_low) * 100, 1
                )

            ev.rsi_14 = round(self._compute_rsi(hist['Close']), 1)

            # ── Scoring ───────────────────────────────────────────────────────
            ev.fundamental_score = self._fundamental_score(ev)
            ev.technical_score   = self._technical_score(ev)
            ev.total_score       = round(ev.fundamental_score + ev.technical_score, 1)
            ev.score_label       = self._score_label(ev.total_score)

            # ── IV rank (reuse existing market_data function) ─────────────────
            try:
                ev.iv_rank = get_iv_rank(ticker)
            except Exception:
                ev.iv_rank = 0.0

            # ── Options action ─────────────────────────────────────────────────
            ev.options_action, ev.options_reason = self._options_action(ev)

            # ── Plain-English summary ─────────────────────────────────────────
            ev.plain_english = self._plain_english(ev)

        except Exception as e:
            ev.error = str(e)

        return ev

    def scan_watchlist(self, tickers: list) -> list:
        """
        Screen a list of tickers and return results sorted best-first.
        Errors and no-data results are placed at the end.
        """
        results = []
        for ticker in tickers:
            ev = self.screen(ticker)
            results.append(ev)

        # Sort: valid scores descending, errors last
        results.sort(
            key=lambda e: (0 if e.error else 1, -e.total_score)
        )
        return results

    # ── Private helpers ────────────────────────────────────────────────────────

    def _compute_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI-14 from a price series."""
        delta = prices.diff().dropna()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)

        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

        last_gain = float(avg_gain.iloc[-1])
        last_loss = float(avg_loss.iloc[-1])

        if last_loss == 0:
            return 100.0
        rs  = last_gain / last_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return max(0.0, min(100.0, rsi))

    def _fundamental_score(self, ev: StockValueEval) -> float:
        """
        Score 0-50 based on fundamental data.

        Components:
          Analyst target discount (0-25 pts)
          Forward P/E                (0-15 pts)
          Earnings growth            (0-10 pts)
        """
        score = 0.0
        available_max = 0  # track how many components have data

        # ── Analyst target discount (0-25 pts) ────────────────────────────────
        if ev.analyst_target > 0:
            available_max += 25
            d = ev.discount_to_target_pct
            if d >= 30:
                score += 25
            elif d >= 20:
                score += 18
            elif d >= 10:
                score += 10
            elif d >= 5:
                score += 5
            # Negative discount (stock above target) = 0 pts

        # ── Forward P/E (0-15 pts) ────────────────────────────────────────────
        if ev.forward_pe > 0:
            available_max += 15
            pe = ev.forward_pe
            if pe <= 12:
                score += 15
            elif pe <= 15:
                score += 12
            elif pe <= 20:
                score += 8
            elif pe <= 25:
                score += 4
            # PE > 25 = 0 pts (expensive)

        # ── Earnings growth (0-10 pts) ────────────────────────────────────────
        if ev.earnings_growth != 0:
            available_max += 10
            g = ev.earnings_growth
            if g >= 0.20:
                score += 10
            elif g >= 0.10:
                score += 7
            elif g >= 0.05:
                score += 4
            elif g > 0:
                score += 2
            # Negative growth = 0 pts

        # ── Scale to 0-50 if some components are missing ─────────────────────
        if available_max == 0:
            return 0.0
        return round(score / available_max * 50, 1)

    def _technical_score(self, ev: StockValueEval) -> float:
        """
        Score 0-50 based on technical data.

        Components:
          52-week low proximity (0-25 pts)  — lower price position = better
          RSI-14 oversold       (0-25 pts)  — lower RSI = more oversold = better
        """
        score = 0.0

        # ── 52-week low proximity (0-25 pts) ──────────────────────────────────
        pos = ev.pct_above_52w_low  # 0% = at 52w low, 100% = at 52w high
        if pos <= 10:
            score += 25   # Very near 52-week low
        elif pos <= 20:
            score += 20
        elif pos <= 35:
            score += 12
        elif pos <= 50:
            score += 6
        # > 50% above 52w low = 0 pts

        # ── RSI (0-25 pts) ────────────────────────────────────────────────────
        rsi = ev.rsi_14
        if rsi <= 25:
            score += 25   # Extremely oversold
        elif rsi <= 30:
            score += 20
        elif rsi <= 40:
            score += 12
        elif rsi <= 50:
            score += 5
        # RSI > 50 = 0 pts (not oversold)

        return round(score, 1)

    def _score_label(self, score: float) -> str:
        if score >= STRONG_SIGNAL:
            return 'Strong Value Signal'
        if score >= WATCH_LIST:
            return 'Watch List'
        if score >= NEUTRAL:
            return 'Neutral'
        return 'No Signal'

    def _options_action(self, ev: StockValueEval) -> tuple:
        """
        Return (action_str, plain_reason_str) based on value score + IV rank.
        Only recommends buying if the stock is genuinely undervalued (score >= 55).
        """
        if ev.total_score < WATCH_LIST:
            return (
                'wait',
                'The stock does not yet show strong enough undervaluation signals to warrant buying options.'
            )

        if ev.iv_rank == 0:
            return (
                'wait',
                'Could not retrieve options pricing data. Check back when market is open.'
            )

        if ev.iv_rank <= BUY_CALL_MAX_IV:
            return (
                'buy_call',
                (
                    f'Options are cheap right now (IV rank {ev.iv_rank:.0f}/100). '
                    f'This is a good time to buy a call option. If the stock '
                    f'recovers toward analyst targets (${ev.analyst_target:.0f}), '
                    f'the option could be worth significantly more. '
                    f'Target: 45-60 DTE, strike near the current price.'
                )
            )

        if ev.iv_rank <= DEBIT_SPREAD_MAX_IV:
            return (
                'debit_spread',
                (
                    f'Options are moderately priced (IV rank {ev.iv_rank:.0f}/100). '
                    f'A debit call spread is better than a naked call — it reduces your '
                    f'upfront cost while keeping the profit potential if the stock '
                    f'recovers. Buy the ATM call, sell a call ~10-15% above the current price.'
                )
            )

        return (
            'wait',
            (
                f'Options are expensive right now (IV rank {ev.iv_rank:.0f}/100). '
                f'Buying options when IV is high means you overpay. '
                f'The stock may be undervalued, but wait for IV to come down before buying calls. '
                f'Add to your watch list and check again when IV rank drops below 50.'
            )
        )

    def _plain_english(self, ev: StockValueEval) -> str:
        """
        Generate a non-technical plain-English summary for the user.
        """
        lines = []

        # ── What the score means ──────────────────────────────────────────────
        label = ev.score_label
        score = ev.total_score
        lines.append(f'OVERALL RATING: {label.upper()} ({score:.0f}/100)')
        lines.append('')

        # ── Fundamental section ───────────────────────────────────────────────
        lines.append('WHAT ANALYSTS THINK')
        if ev.analyst_target > 0 and ev.discount_to_target_pct > 0:
            lines.append(
                f'  Wall Street analysts have a consensus price target of '
                f'${ev.analyst_target:.0f} for {ev.ticker}. The stock is currently '
                f'trading at ${ev.price:.2f}, which is {ev.discount_to_target_pct:.0f}% '
                f'below that target. In other words, analysts think there is '
                f'significant upside potential from here.'
            )
        elif ev.analyst_target > 0:
            lines.append(
                f'  Analysts have a target of ${ev.analyst_target:.0f} but the stock '
                f'is already near or above that target (${ev.price:.2f}).'
            )
        else:
            lines.append(f'  No analyst price target available for {ev.ticker}.')

        if ev.forward_pe > 0:
            if ev.forward_pe <= 15:
                pe_comment = f'This is quite cheap relative to historical market averages (usually 18-22x).'
            elif ev.forward_pe <= 20:
                pe_comment = f'This is a fair valuation — neither cheap nor expensive.'
            else:
                pe_comment = f'This is above average — the market is pricing in high growth.'
            lines.append(
                f'  The stock trades at {ev.forward_pe:.1f}x next year\'s expected earnings. '
                f'{pe_comment}'
            )

        if ev.earnings_growth > 0:
            lines.append(
                f'  Earnings are growing at {ev.earnings_growth*100:.0f}% year-over-year '
                f'— the company is still expanding, not shrinking.'
            )
        elif ev.earnings_growth < 0:
            lines.append(
                f'  Warning: earnings are down {abs(ev.earnings_growth)*100:.0f}% '
                f'year-over-year. The company\'s profitability is currently declining.'
            )

        lines.append('')

        # ── Technical section ─────────────────────────────────────────────────
        lines.append('HOW THE STOCK IS TRADING')
        pos = ev.pct_above_52w_low
        if pos <= 15:
            lines.append(
                f'  {ev.ticker} is near its 52-week low (${ev.week52_low:.2f}). '
                f'The current price of ${ev.price:.2f} is only {pos:.0f}% above the low. '
                f'This often signals the stock has been heavily sold off and may be oversold.'
            )
        elif pos <= 35:
            lines.append(
                f'  {ev.ticker} has pulled back from its highs. It is {pos:.0f}% above its '
                f'52-week low of ${ev.week52_low:.2f}. There has been meaningful weakness '
                f'recently, which may represent a buying opportunity.'
            )
        else:
            lines.append(
                f'  {ev.ticker} is {pos:.0f}% above its 52-week low. The stock has not '
                f'pulled back significantly from recent highs (52-week high: ${ev.week52_high:.2f}).'
            )

        rsi = ev.rsi_14
        if rsi <= 30:
            lines.append(
                f'  RSI is {rsi:.0f} — this is in "oversold" territory (below 30). '
                f'Historically, stocks at this RSI level are due for at least a short-term bounce. '
                f'Think of RSI as a "rubber band" — the more stretched it is, the more likely a snap-back.'
            )
        elif rsi <= 40:
            lines.append(
                f'  RSI is {rsi:.0f} — the stock is showing weakness and approaching oversold '
                f'territory (oversold = below 30). The selling pressure is elevated.'
            )
        elif rsi >= 70:
            lines.append(
                f'  RSI is {rsi:.0f} — the stock is in "overbought" territory (above 70). '
                f'This is not an ideal entry point for value buyers.'
            )
        else:
            lines.append(
                f'  RSI is {rsi:.0f} — the stock is in a neutral range, '
                f'neither clearly oversold nor overbought.'
            )

        lines.append('')

        # ── Options angle ─────────────────────────────────────────────────────
        lines.append('OPTIONS ANGLE')
        if ev.options_action == 'buy_call':
            lines.append(
                f'  RECOMMENDED: Buy a call option on {ev.ticker}. '
                f'Options are cheap right now (IV rank: {ev.iv_rank:.0f}/100 — anything '
                f'below 35 is considered inexpensive). A call option gives you the right '
                f'to benefit if the stock rises toward its analyst target of '
                f'${ev.analyst_target:.0f} without having to own the full shares.'
            )
            lines.append(
                f'  Suggested approach: Buy a 45-60 day call near the current price (${ev.price:.2f}). '
                f'This limits your risk to just the premium paid while keeping upside potential.'
            )
        elif ev.options_action == 'debit_spread':
            lines.append(
                f'  SUGGESTED: Consider a call debit spread on {ev.ticker}. '
                f'Options are moderately priced (IV rank: {ev.iv_rank:.0f}/100). '
                f'A spread means you buy one call and sell another at a higher strike — '
                f'this cuts your upfront cost and reduces the impact of expensive options.'
            )
        else:
            lines.append(
                f'  WAIT: {ev.options_reason}'
            )

        if 'analyst_target' in ev.data_missing:
            lines.append('')
            lines.append(
                f'  Note: No analyst price target available for {ev.ticker}. '
                f'The fundamental score is based only on available data (P/E and growth).'
            )

        return '\n'.join(lines)
