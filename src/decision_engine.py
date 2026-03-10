"""
Decision Engine — The "trader brain."
Analyzes market data against the knowledge base rules and produces trade recommendations.

Sources: TastyTrade research, Natenberg, Sinclair
"""

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

from .knowledge_base import KnowledgeBase
from .greeks import calculate_greeks, implied_volatility, dte_to_years, probability_of_profit
from .market_data import get_underlying_data, get_historical_volatility, get_iv_rank, get_options_chain, get_vix, get_earnings_date
from .macro_calendar import check_macro_blackout, get_upcoming_events


@dataclass
class TradeRecommendation:
    ticker: str
    strategy: str
    action: str              # 'open' or 'pass'
    rationale: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    legs: list[dict] = field(default_factory=list)
    entry_credit: float = 0.0
    max_profit: float = 0.0
    max_loss: float = 0.0
    probability_of_profit: float = 0.0
    suggested_contracts: int = 0
    regime: str = ''
    iv_rank: float = 0.0
    iv_percentile: float = 0.0
    current_iv: float = 0.0   # ATM implied volatility (annualized)
    current_hv: float = 0.0   # 20-day realized volatility
    vrp_ratio: float = 0.0    # IV / HV — the volatility risk premium
    vix: float = 0.0
    next_earnings_date: Optional[str] = None
    days_until_earnings: Optional[int] = None
    upcoming_macro_events: list = field(default_factory=list)  # events in the 45-day window
    current_price: float = 0.0        # underlying price at time of analysis
    trade_score: float = 0.0          # 0-100 quality score
    trade_score_label: str = ''       # "Strong" / "Good" / "Marginal" / "Avoid"
    plain_english: str = ''           # plain-language summary for non-technical readers


class DecisionEngine:
    """
    Core decision engine. Given a ticker and portfolio context,
    evaluates whether and how to open a trade.
    """

    RISK_FREE_RATE = 0.05  # approximate

    def __init__(self, portfolio_value: float = 25000):
        self.kb = KnowledgeBase()
        self.portfolio_value = portfolio_value

    def analyze(self, ticker: str, strategy_name: str = 'auto') -> TradeRecommendation:
        """
        Full analysis pipeline for a ticker.
        Returns a TradeRecommendation with action, rationale, and trade details.
        """
        rec = TradeRecommendation(ticker=ticker, strategy=strategy_name, action='pass')

        # --- Step 1: Fetch market data ---
        try:
            underlying = get_underlying_data(ticker)
            price = underlying['price']
            rec.current_price = price
        except Exception as e:
            rec.rationale.append(f"Could not fetch data for {ticker}: {e}")
            return rec

        # --- Step 2: Liquidity check ---
        if price < 15:
            rec.rationale.append(f"FAIL [HR-007] Price ${price:.2f} < $15 minimum.")
            return rec
        if underlying.get('volume', 0) < 100000:
            rec.warnings.append(f"Low volume: {underlying.get('volume', 0):,}")

        # --- Step 2b: HR-001 — Earnings blackout check ---
        try:
            earnings_info = get_earnings_date(ticker)
            rec.next_earnings_date = earnings_info.get('next_earnings_date')
            rec.days_until_earnings = earnings_info.get('days_until_earnings')

            if earnings_info.get('has_earnings_in_window'):
                days = earnings_info['days_until_earnings']
                rec.rationale.append(
                    f"FAIL [HR-001] Earnings in {days} day{'s' if days != 1 else ''} "
                    f"({earnings_info['next_earnings_date']}). "
                    f"Blackout: earnings fall within the 45-day holding window. "
                    f"Rule: never hold short premium through an earnings event."
                )
                return rec
            elif rec.next_earnings_date:
                days = earnings_info['days_until_earnings']
                rec.rationale.append(
                    f"PASS [HR-001] Earnings in {days} days ({rec.next_earnings_date}) -- "
                    f"outside the holding window."
                )
            else:
                rec.rationale.append("PASS [HR-001] No earnings date found -- no blackout active.")
        except Exception as e:
            rec.warnings.append(f"Could not check earnings date: {e}. Proceeding with caution.")

        # --- Step 2c: HR-010 -- Macro event blackout check ---
        try:
            blocked, imminent = check_macro_blackout(days_before=2)
            upcoming_45 = get_upcoming_events(days_ahead=45)
            rec.upcoming_macro_events = [
                {'type': e['type'], 'date': e['date_str'], 'days_away': e['days_away'],
                 'description': e['description']}
                for e in upcoming_45
            ]
            if blocked:
                event_list = ', '.join(
                    f"{e['type']} {e['date_str']} ({e['days_away']}d)"
                    for e in imminent
                )
                rec.rationale.append(
                    f"FAIL [HR-010] Macro event blackout: {event_list}. "
                    f"No new positions within 2 days of FOMC/CPI/NFP."
                )
                return rec
            elif upcoming_45:
                next_e = upcoming_45[0]
                rec.rationale.append(
                    f"PASS [HR-010] Next macro event: {next_e['type']} in "
                    f"{next_e['days_away']} days ({next_e['date_str']}) -- outside 2-day blackout."
                )
            else:
                rec.rationale.append("PASS [HR-010] No high-impact macro events in the next 45 days.")
        except Exception as e:
            rec.warnings.append(f"Could not check macro calendar: {e}. Proceeding with caution.")

        # --- Step 3: Get options chain ---
        chain = get_options_chain(ticker, target_dte=45)
        if chain is None or chain.empty:
            rec.rationale.append("No options data available.")
            return rec

        dte = int(chain['dte'].iloc[0])
        if dte < 20 or dte > 70:
            rec.rationale.append(f"FAIL [EN-002] DTE {dte} outside acceptable range 20-70.")
            return rec

        # --- Step 4: Compute HV and IV rank ---
        hv_data = get_historical_volatility(ticker)
        hv20 = hv_data.get('hv20', 0.20)

        # Get ATM option to extract IV
        atm_options = self._get_atm_options(chain, price, dte)
        if atm_options is None:
            rec.rationale.append("Could not find ATM options.")
            return rec

        current_iv = self._estimate_iv(atm_options, price, dte)
        if current_iv is None or current_iv <= 0:
            current_iv = hv20  # fallback to HV if IV calc fails

        iv_rank_data = get_iv_rank(ticker, current_iv)
        iv_rank = iv_rank_data.get('iv_rank', 50)
        rec.iv_rank = iv_rank
        rec.iv_percentile = iv_rank_data.get('iv_percentile', 50)
        rec.current_iv = iv_rank_data.get('current_iv', current_iv)
        # Always use actual realized HV20 for display (not the VIX-method approx)
        rec.current_hv = hv20
        rec.vrp_ratio = round(current_iv / hv20, 2) if hv20 > 0 else 0.0

        vix = get_vix()
        rec.vix = vix

        # --- Step 5: Market regime ---
        regime = self.kb.determine_regime(vix, iv_rank)
        rec.regime = regime

        # --- Step 6: Auto-select strategy based on regime ---
        if strategy_name == 'auto':
            strategy_name = self._select_strategy(regime, iv_rank)
            rec.strategy = strategy_name

        strategy = self.kb.get_strategy(strategy_name)
        if not strategy:
            rec.rationale.append(f"Strategy '{strategy_name}' not found in knowledge base.")
            return rec

        # --- Step 7: Entry condition checks ---
        entry = strategy.get('entry_conditions', {})
        iv_min = entry.get('iv_rank_minimum', 30)

        if iv_rank < iv_min:
            rec.rationale.append(
                f"FAIL [EN-001] IV rank {iv_rank:.0f} < {iv_min} minimum for {strategy_name}. "
                f"Not enough premium. Regime: {regime}."
            )
            return rec

        # --- Step 8: Find optimal strikes ---
        legs_result = self._find_strikes(strategy, chain, price, dte, current_iv)
        if legs_result is None:
            rec.rationale.append("Could not construct trade: strikes/credit criteria not met.")
            return rec

        legs, credit, max_loss = legs_result

        # Min credit check
        min_credit = entry.get('min_credit_per_contract', 0.50)
        if credit < min_credit:
            rec.rationale.append(
                f"FAIL [EN-004] Credit ${credit:.2f} < minimum ${min_credit:.2f}."
            )
            return rec

        # --- Step 9: Position sizing ---
        contracts = self._size_position(strategy, credit, max_loss)

        # --- Step 10: Build recommendation ---
        rec.action = 'open'
        rec.legs = legs
        rec.entry_credit = round(credit, 2)
        rec.max_profit = round(credit * 100 * contracts, 2)
        rec.max_loss = round(max_loss * 100 * contracts, 2) if max_loss else None
        rec.suggested_contracts = contracts
        rec.probability_of_profit = self._estimate_pop(legs)

        rec.rationale.extend([
            f"PASS [EN-001] IV rank {iv_rank:.0f} >= {iv_min} minimum.",
            f"PASS [EN-002] DTE {dte} is within 30-60 window.",
            f"Market regime: {regime.replace('_', ' ').title()}",
            f"VIX: {vix:.1f}",
            f"Strategy selected: {strategy_name} (credit ${credit:.2f}/contract)",
        ])

        # Strategy-specific notes
        if strategy_name == 'Jade Lizard':
            call_wing = next(
                (l for l in legs if l['action'] == 'BUY' and l['type'] == 'CALL'), None
            )
            short_call = next(
                (l for l in legs if l['action'] == 'SELL' and l['type'] == 'CALL'), None
            )
            if call_wing and short_call:
                wing_w = call_wing['strike'] - short_call['strike']
                rec.rationale.append(
                    f"Jade Lizard: credit ${credit:.2f} > call spread width ${wing_w:.1f} "
                    f"-- ZERO upside risk. Only downside exposure (short put)."
                )
        elif strategy_name == 'Covered Call':
            rec.warnings.append(
                "COVERED CALL REQUIRES 100 SHARES: This trade is only valid if you already "
                f"own 100 shares of {ticker}. The engine constructs the call leg only."
            )

        if hv20:
            edge = round((current_iv - hv20) * 100, 1)
            if edge > 0:
                rec.rationale.append(
                    f"Natenberg edge: IV ({current_iv*100:.1f}%) > HV20 ({hv20*100:.1f}%) by {edge:.1f}pp. "
                    f"Selling expensive vol."
                )
            else:
                rec.warnings.append(
                    f"IV ({current_iv*100:.1f}%) <= HV20 ({hv20*100:.1f}%). Edge is thin -- Sinclair caution."
                )

        # --- Step 11: Trade quality score + plain-English summary ---
        rec.trade_score = self._compute_trade_score(rec)
        rec.trade_score_label = self._score_label(rec.trade_score)
        rec.plain_english = self._plain_english(rec, price, dte)

        return rec

    # ── Quality score helpers ─────────────────────────────────────────────────

    def _compute_trade_score(self, rec: 'TradeRecommendation') -> float:
        """Combine key metrics into a 0-100 trade quality score."""
        score = 0.0

        # IV rank (25 pts) — want high IV for premium selling
        iv = rec.iv_rank
        if iv >= 60:   score += 25
        elif iv >= 50: score += 20
        elif iv >= 40: score += 14
        elif iv >= 30: score += 7

        # POP (25 pts) — probability of keeping the premium
        pop = rec.probability_of_profit
        if pop >= 0.80:   score += 25
        elif pop >= 0.70: score += 20
        elif pop >= 0.65: score += 14
        elif pop >= 0.60: score += 7

        # VRP (20 pts) — are options expensive vs recent moves?
        vrp = rec.vrp_ratio
        if vrp >= 1.30:   score += 20
        elif vrp >= 1.15: score += 15
        elif vrp >= 1.00: score += 8

        # Macro event risk (15 pts) — fewer upcoming events = better
        n_events = len(rec.upcoming_macro_events)
        if n_events == 0:   score += 15
        elif n_events == 1: score += 10
        elif n_events == 2: score += 5

        # Credit size (15 pts) — meaningful income?
        c = rec.entry_credit
        if c >= 3.0:   score += 15
        elif c >= 1.5: score += 10
        elif c >= 0.5: score += 5

        return round(min(score, 100.0), 1)

    @staticmethod
    def _score_label(score: float) -> str:
        if score >= 80: return 'Strong'
        if score >= 60: return 'Good'
        if score >= 40: return 'Marginal'
        return 'Avoid'

    def _plain_english(self, rec: 'TradeRecommendation', price: float, dte: int) -> str:
        """Return a non-technical, plain-language summary of the trade."""
        s = rec.strategy
        t = rec.ticker
        c = rec.entry_credit
        contracts = rec.suggested_contracts
        total_income = round(c * 100 * contracts, 0)
        pop_pct = round(rec.probability_of_profit * 100)

        # Find key strikes
        short_put  = next((l for l in rec.legs if l['action']=='SELL' and l['type']=='PUT'),  None)
        short_call = next((l for l in rec.legs if l['action']=='SELL' and l['type']=='CALL'), None)

        put_strike  = short_put['strike']  if short_put  else None
        call_strike = short_call['strike'] if short_call else None

        lines = []
        lines.append(f"WHAT IS THIS TRADE?")

        if s == 'Short Strangle':
            lines.append(
                f"You sell two options on {t} at the same time: "
                f"a put at ${put_strike} (below the stock) and a call at ${call_strike} (above it). "
                f"{t} is currently at ${price:.2f}. "
                f"As long as {t} stays between ${put_strike} and ${call_strike} until expiry "
                f"({dte} days from now), you keep the full income."
            )
        elif s == 'Iron Condor':
            buy_put  = next((l for l in rec.legs if l['action']=='BUY'  and l['type']=='PUT'),  None)
            buy_call = next((l for l in rec.legs if l['action']=='BUY'  and l['type']=='CALL'), None)
            bp = buy_put['strike']  if buy_put  else '?'
            bc = buy_call['strike'] if buy_call else '?'
            lines.append(
                f"An Iron Condor on {t} (currently ${price:.2f}) profits if the stock stays roughly flat. "
                f"You sell options at ${put_strike} and ${call_strike}, "
                f"and buy protection at ${bp} and ${bc}. "
                f"Your maximum loss is capped, which makes this safer than a strangle."
            )
        elif s == 'Short Put':
            lines.append(
                f"You sell a put option on {t} at ${put_strike}. "
                f"{t} is at ${price:.2f} — so you need it to stay above ${put_strike} "
                f"({round((price - put_strike) / price * 100, 1)}% buffer) for {dte} days. "
                f"Think of it as: you agree to buy {t} at ${put_strike} if it falls there, "
                f"and get paid ${c:.2f}/share for taking that risk."
            )
        elif s == 'Jade Lizard':
            lines.append(
                f"A Jade Lizard on {t} sells a put at ${put_strike} and a call spread above "
                f"the stock. You collect ${c:.2f}/share with ZERO risk if {t} rallies — "
                f"your only exposure is if the stock drops below ${put_strike}."
            )
        elif s == 'Covered Call':
            lines.append(
                f"You sell a call option against 100 shares of {t} you already own. "
                f"If {t} stays below ${call_strike} for {dte} days, you pocket ${c:.2f}/share "
                f"in income. If it rises above ${call_strike}, your shares get called away "
                f"at that price — still a profit on the shares."
            )
        else:
            lines.append(f"Strategy: {s} on {t} at ${price:.2f}.")

        lines.append('')
        lines.append(f"WHAT YOU EARN:")
        lines.append(
            f"${c:.2f} per share x 100 shares = ${c*100:.0f} per contract. "
            f"With {contracts} contract(s): ${total_income:.0f} collected upfront."
        )

        lines.append('')
        lines.append(f"YOUR ODDS:")
        lines.append(
            f"Based on current options pricing, there is roughly a {pop_pct}% chance "
            f"this trade is profitable at expiry."
        )

        lines.append('')
        lines.append(f"WHEN TO CLOSE EARLY (do not wait until expiry):")
        profit_target = round(total_income * 0.5)
        stop_loss = round(total_income * 2)
        lines.append(f"  Take profit when you have made ${profit_target} (50% of max income).")
        lines.append(f"  Cut losses if the position is down ${stop_loss} (2x what you collected).")
        lines.append(f"  Always close with 21 days remaining regardless.")

        # Macro warning
        if rec.upcoming_macro_events:
            next_e = rec.upcoming_macro_events[0]
            lines.append('')
            lines.append(
                f"HEADS UP: There is a {next_e['type']} announcement in "
                f"{next_e['days_away']} days ({next_e['date']}). "
                f"These events can cause sudden moves in the market. "
                f"Consider closing before the event if it falls inside your holding period."
            )

        return '\n'.join(lines)

    def _select_strategy(self, regime: str, iv_rank: float) -> str:
        """
        Select the appropriate strategy based on market regime and IV rank.

        Strategy map:
          high_iv_environment  → Short Strangle (undefined risk, max premium)
          low_iv_environment   → Iron Condor (defined risk, capital efficiency)
          trending_market      → Jade Lizard (IV>=40) or Short Put (IV>=25)
                                 Slightly bullish bias fits a trending market.
                                 Jade Lizard: zero upside risk + big put premium.
          neutral              → Short Strangle (IV>=50) or Iron Condor (IV>=30)
        """
        if regime == 'high_iv_environment':
            return 'Short Strangle'
        elif regime == 'low_iv_environment':
            return 'Iron Condor'
        else:
            # Normal environment: VIX 15-25, IV rank 30-60
            # IV rank gradient:
            #   >= 50  → Short Strangle: highest premium, justified undefined risk
            #   >= 40  → Jade Lizard: structured — exploit skew, zero upside risk
            #   >= 30  → Iron Condor: fully defined risk, conservative
            #   <  30  → Iron Condor: low IV, minimize capital at risk
            if iv_rank >= 50:
                return 'Short Strangle'
            elif iv_rank >= 40:
                return 'Jade Lizard'
            elif iv_rank >= 30:
                return 'Iron Condor'
            else:
                return 'Iron Condor'

    def _get_atm_options(self, chain: pd.DataFrame, price: float, dte: int) -> Optional[pd.DataFrame]:
        """Get options closest to ATM (calls + puts) for IV estimation."""
        atm = chain.copy()
        atm['strike_diff'] = abs(atm['strike'] - price)
        if atm.empty:
            return None
        return atm.nsmallest(6, 'strike_diff')  # 3 calls + 3 puts near ATM

    def _estimate_iv(self, atm_options: pd.DataFrame, price: float, dte: int) -> Optional[float]:
        """
        Estimate current ATM implied volatility.

        Priority:
        1. Newton-Raphson on mid price for ATM calls (most accurate)
        2. Newton-Raphson on mid price for ATM puts (fallback)
        3. Average of yfinance impliedVolatility column (last resort — based on
           last traded price, not mid, but better than nothing)
        """
        T = dte_to_years(dte)
        ivs = []

        # Pass 1: calls via Newton-Raphson on mid
        calls = atm_options[atm_options['option_type'] == 'call']
        for _, row in calls.iterrows():
            if row.get('mid', 0) > 0 and row.get('strike', 0) > 0:
                iv = implied_volatility(
                    'call', float(row['mid']), price,
                    float(row['strike']), T, self.RISK_FREE_RATE
                )
                if iv and 0.05 < iv < 3.0:
                    ivs.append(iv)

        # Pass 2: puts via Newton-Raphson on mid (if calls didn't yield enough)
        if len(ivs) < 2:
            puts = atm_options[atm_options['option_type'] == 'put']
            for _, row in puts.iterrows():
                if row.get('mid', 0) > 0 and row.get('strike', 0) > 0:
                    iv = implied_volatility(
                        'put', float(row['mid']), price,
                        float(row['strike']), T, self.RISK_FREE_RATE
                    )
                    if iv and 0.05 < iv < 3.0:
                        ivs.append(iv)

        if ivs:
            return round(sum(ivs) / len(ivs), 4)

        # Pass 3: fall back to yfinance impliedVolatility column
        yf_ivs = atm_options['impliedVolatility'].dropna()
        yf_ivs = yf_ivs[(yf_ivs > 0.05) & (yf_ivs < 3.0)]
        if not yf_ivs.empty:
            return round(float(yf_ivs.mean()), 4)

        return None

    def _find_strikes(
        self, strategy: dict, chain: pd.DataFrame, price: float, dte: int, iv: float
    ) -> Optional[tuple[list, float, float]]:
        """Find optimal strikes for the strategy. Returns (legs, credit, max_loss)."""
        strategy_name = strategy['name']
        T = dte_to_years(dte)

        if strategy_name == 'Short Strangle':
            return self._construct_strangle(chain, price, T, iv)
        elif strategy_name == 'Iron Condor':
            return self._construct_iron_condor(chain, price, T, iv)
        elif strategy_name == 'Short Put':
            return self._construct_short_put(chain, price, T, iv)
        elif strategy_name == 'Jade Lizard':
            return self._construct_jade_lizard(chain, price, T, iv)
        elif strategy_name == 'Covered Call':
            return self._construct_covered_call(chain, price, T, iv)
        return None

    def _construct_strangle(
        self, chain: pd.DataFrame, price: float, T: float, iv: float
    ) -> Optional[tuple[list, float, float]]:
        """Construct a short strangle at ~16-delta strikes."""
        calls = chain[chain['option_type'] == 'call'].copy()
        puts = chain[chain['option_type'] == 'put'].copy()

        if calls.empty or puts.empty:
            return None

        # Find strikes closest to 16-delta
        short_call = self._find_strike_by_delta(calls, 0.16, price, T, iv, 'call')
        short_put = self._find_strike_by_delta(puts, -0.16, price, T, iv, 'put')

        if short_call is None or short_put is None:
            return None

        call_mid = float(short_call.get('mid', short_call.get('lastPrice', 0)))
        put_mid = float(short_put.get('mid', short_put.get('lastPrice', 0)))
        credit = call_mid + put_mid

        legs = [
            {
                'action': 'SELL',
                'type': 'CALL',
                'strike': float(short_call['strike']),
                'mid': round(call_mid, 2),
                'delta': self._calc_delta('call', price, float(short_call['strike']), T, iv, short_call),
            },
            {
                'action': 'SELL',
                'type': 'PUT',
                'strike': float(short_put['strike']),
                'mid': round(put_mid, 2),
                'delta': self._calc_delta('put', price, float(short_put['strike']), T, iv, short_put),
            },
        ]

        return legs, round(credit, 2), None  # undefined risk

    def _construct_iron_condor(
        self, chain: pd.DataFrame, price: float, T: float, iv: float
    ) -> Optional[tuple[list, float, float]]:
        """Construct an iron condor with 16-delta short strikes and 5-wide wings."""
        calls = chain[chain['option_type'] == 'call'].copy()
        puts = chain[chain['option_type'] == 'put'].copy()

        if calls.empty or puts.empty:
            return None

        short_call = self._find_strike_by_delta(calls, 0.16, price, T, iv, 'call')
        short_put = self._find_strike_by_delta(puts, -0.16, price, T, iv, 'put')

        if short_call is None or short_put is None:
            return None

        sc_strike = float(short_call['strike'])
        sp_strike = float(short_put['strike'])

        # Find long wings ~5 points away
        wing_width = self._determine_wing_width(price)
        lc_strike = sc_strike + wing_width
        lp_strike = sp_strike - wing_width

        long_call = self._find_nearest_strike(calls, lc_strike)
        long_put = self._find_nearest_strike(puts, lp_strike)

        if long_call is None or long_put is None:
            return None

        sc_mid = float(short_call.get('mid', 0))
        sp_mid = float(short_put.get('mid', 0))
        lc_mid = float(long_call.get('mid', 0))
        lp_mid = float(long_put.get('mid', 0))

        credit = (sc_mid + sp_mid) - (lc_mid + lp_mid)
        actual_wing = float(long_call['strike']) - sc_strike
        max_loss = actual_wing - credit if actual_wing > 0 else wing_width - credit

        legs = [
            {'action': 'SELL', 'type': 'CALL', 'strike': sc_strike, 'mid': round(sc_mid, 2)},
            {'action': 'BUY', 'type': 'CALL', 'strike': float(long_call['strike']), 'mid': round(lc_mid, 2)},
            {'action': 'SELL', 'type': 'PUT', 'strike': sp_strike, 'mid': round(sp_mid, 2)},
            {'action': 'BUY', 'type': 'PUT', 'strike': float(long_put['strike']), 'mid': round(lp_mid, 2)},
        ]

        return legs, round(credit, 2), round(max_loss, 2)

    def _construct_short_put(
        self, chain: pd.DataFrame, price: float, T: float, iv: float
    ) -> Optional[tuple[list, float, float]]:
        """Construct a short put at ~30-delta."""
        puts = chain[chain['option_type'] == 'put'].copy()
        if puts.empty:
            return None

        short_put = self._find_strike_by_delta(puts, -0.30, price, T, iv, 'put')
        if short_put is None:
            return None

        put_mid = float(short_put.get('mid', short_put.get('lastPrice', 0)))
        strike = float(short_put['strike'])
        max_loss = strike - put_mid  # if stock goes to zero

        legs = [
            {
                'action': 'SELL',
                'type': 'PUT',
                'strike': strike,
                'mid': round(put_mid, 2),
                'delta': self._calc_delta('put', price, strike, T, iv, short_put),
            }
        ]

        return legs, round(put_mid, 2), round(max_loss, 2)

    def _construct_jade_lizard(
        self, chain: pd.DataFrame, price: float, T: float, iv: float
    ) -> Optional[tuple[list, float, float]]:
        """
        Construct a Jade Lizard: Short OTM Put + Short OTM Call Spread.

        Structure:
          SELL ~30-delta put
          SELL ~20-delta call  }  call spread
          BUY  ~20-delta call + wing_width above

        Key constraint (TastyTrade): total_credit > call_spread_width
        When satisfied, there is ZERO upside risk. Only downside risk from the put.
        Returns None if the credit constraint cannot be met.
        """
        calls = chain[chain['option_type'] == 'call'].copy()
        puts  = chain[chain['option_type'] == 'put'].copy()

        if calls.empty or puts.empty:
            return None

        # Short put at ~30-delta
        short_put = self._find_strike_by_delta(puts, -0.30, price, T, iv, 'put')
        # Short call at ~20-delta (tighter than strangle to leave room for long call)
        short_call = self._find_strike_by_delta(calls, 0.20, price, T, iv, 'call')

        if short_put is None or short_call is None:
            return None

        sc_strike = float(short_call['strike'])
        sp_strike = float(short_put['strike'])

        # Jade lizard uses a TIGHT call spread (purpose: cap upside cheaply).
        # Wings are intentionally narrower than iron condor to satisfy
        # the credit > width constraint. Try progressively wider wings.
        jl_wing = self._determine_jade_wing_width(price)
        result = None
        for multiplier in (1, 2):           # try narrow, then 2x narrow
            attempt_wing = jl_wing * multiplier
            lc_candidate = self._find_nearest_strike(calls, sc_strike + attempt_wing)
            if lc_candidate is None:
                continue
            sp_mid_t = float(short_put.get('mid',  short_put.get('lastPrice', 0)) or 0)
            sc_mid_t = float(short_call.get('mid', short_call.get('lastPrice', 0)) or 0)
            lc_mid_t = float(lc_candidate.get('mid', lc_candidate.get('lastPrice', 0)) or 0)
            credit_t = sp_mid_t + sc_mid_t - lc_mid_t
            wing_t   = float(lc_candidate['strike']) - sc_strike
            if credit_t > wing_t and credit_t > 0:
                result = (lc_candidate, sp_mid_t, sc_mid_t, lc_mid_t, credit_t, wing_t)
                break

        if result is None:
            return None  # Can't satisfy zero-upside-risk constraint

        long_call, sp_mid, sc_mid, lc_mid, total_credit, actual_wing = result

        max_loss = sp_strike - total_credit  # Downside only; upside risk = 0

        legs = [
            {
                'action': 'SELL',
                'type': 'PUT',
                'strike': sp_strike,
                'mid': round(sp_mid, 2),
                'delta': self._calc_delta('put', price, sp_strike, T, iv, short_put),
            },
            {
                'action': 'SELL',
                'type': 'CALL',
                'strike': sc_strike,
                'mid': round(sc_mid, 2),
                'delta': self._calc_delta('call', price, sc_strike, T, iv, short_call),
            },
            {
                'action': 'BUY',
                'type': 'CALL',
                'strike': float(long_call['strike']),
                'mid': round(lc_mid, 2),
            },
        ]

        return legs, round(total_credit, 2), round(max_loss, 2)

    def _construct_covered_call(
        self, chain: pd.DataFrame, price: float, T: float, iv: float
    ) -> Optional[tuple[list, float, float]]:
        """
        Construct a covered call: SELL ~30-delta OTM call.

        IMPORTANT: Requires 100 long shares of the underlying.
        The engine constructs the call leg only; stock ownership is assumed.
        Max profit = credit + (strike - price) if stock is called away.
        """
        calls = chain[chain['option_type'] == 'call'].copy()
        if calls.empty:
            return None

        short_call = self._find_strike_by_delta(calls, 0.30, price, T, iv, 'call')
        if short_call is None:
            return None

        call_mid = float(short_call.get('mid', short_call.get('lastPrice', 0)) or 0)
        strike   = float(short_call['strike'])

        if call_mid <= 0:
            return None

        # If assigned: profit = credit + (strike - current_price) per share
        # If expires worthless: profit = credit
        # Undefined downside: stock can go to zero (but offset by stock ownership)
        legs = [
            {
                'action': 'SELL',
                'type': 'CALL',
                'strike': strike,
                'mid': round(call_mid, 2),
                'delta': self._calc_delta('call', price, strike, T, iv, short_call),
                'note': 'Requires 100 shares long',
            }
        ]

        return legs, round(call_mid, 2), None  # undefined downside (stock can drop)

    def _find_strike_by_delta(
        self, options: pd.DataFrame, target_delta: float,
        price: float, T: float, iv: float, option_type: str
    ) -> Optional[pd.Series]:
        """
        Find the option strike whose delta is closest to target_delta.

        Uses each option's own implied volatility from the yfinance chain for
        accurate delta computation. This correctly handles the volatility
        surface skew: OTM puts carry significantly higher IV than ATM (the
        put skew), so using a flat vol would select the wrong strike.
        Falls back to global iv if the option-specific IV is missing/invalid.
        """
        best = None
        best_diff = float('inf')

        for _, row in options.iterrows():
            strike = float(row['strike'])
            if strike <= 0:
                continue
            # Use option-specific IV to account for vol surface skew
            yf_iv = float(row.get('impliedVolatility', 0) or 0)
            option_iv = yf_iv if 0.05 < yf_iv < 3.0 else iv
            g = calculate_greeks(option_type, price, strike, T, self.RISK_FREE_RATE, option_iv)
            diff = abs(g.delta - target_delta)
            if diff < best_diff:
                best_diff = diff
                best = row

        return best

    def _find_nearest_strike(self, options: pd.DataFrame, target_strike: float) -> Optional[pd.Series]:
        """Find the option row with strike closest to target_strike."""
        options = options.copy()
        options['diff'] = abs(options['strike'] - target_strike)
        if options.empty:
            return None
        return options.nsmallest(1, 'diff').iloc[0]

    def _calc_delta(
        self, option_type: str, price: float, strike: float,
        T: float, iv: float, row: Optional[pd.Series] = None
    ) -> float:
        """Compute delta, using the option's own IV from the chain if available."""
        if row is not None:
            yf_iv = float(row.get('impliedVolatility', 0) or 0)
            if 0.05 < yf_iv < 3.0:
                iv = yf_iv
        g = calculate_greeks(option_type, price, strike, T, self.RISK_FREE_RATE, iv)
        return round(g.delta, 3)

    def _determine_wing_width(self, price: float) -> float:
        """Determine iron condor wing width based on underlying price."""
        if price < 50:
            return 2.5
        elif price < 200:
            return 5.0
        elif price < 500:
            return 10.0
        else:
            return 25.0

    def _determine_jade_wing_width(self, price: float) -> float:
        """
        Jade Lizard call spread wing — intentionally NARROWER than iron condor.
        The call spread's purpose is only to cap upside cheaply so that
        credit > wing_width (zero upside risk). A wider wing would make
        the long call cheaper but costs more to buy, reducing net credit.
        """
        if price < 50:
            return 1.0
        elif price < 200:
            return 2.5
        elif price < 500:
            return 5.0
        else:
            return 10.0

    def _size_position(self, strategy: dict, credit: float, max_loss: Optional[float]) -> int:
        """
        Calculate number of contracts based on portfolio risk rules.
        HR-002: Max 2% of portfolio per trade.
        """
        if max_loss and max_loss > 0:
            # Defined risk: size by max loss
            max_risk = self.portfolio_value * 0.02
            contracts = int(max_risk / (max_loss * 100))
        else:
            # Undefined risk: size by buying power / notional cap
            max_notional = self.portfolio_value * 0.02
            contracts = int(max_notional / (credit * 100)) if credit > 0 else 1

        return max(1, min(contracts, 10))  # floor 1, cap 10 for safety

    def _estimate_pop(self, legs: list) -> float:
        """Estimate probability of profit from leg deltas."""
        short_legs = [l for l in legs if l['action'] == 'SELL']
        if not short_legs:
            return 0.5
        deltas = [abs(l.get('delta', 0.16)) for l in short_legs]
        avg_delta = sum(deltas) / len(deltas)
        return round(1.0 - avg_delta, 3)

    def check_hard_risk_violations(self, ticker: str, earnings_days: int = None) -> list:
        """Run hard risk rule checks. Returns list of violations."""
        violations = []

        if earnings_days is not None and earnings_days <= 7:
            violations.append(f"HR-001: Earnings in {earnings_days} days — BLACKOUT ACTIVE")

        return violations

    def format_recommendation(self, rec: TradeRecommendation) -> str:
        """Format a TradeRecommendation for display."""
        vrp_str = f"  (IV/HV ratio: {rec.vrp_ratio:.2f}x)" if rec.vrp_ratio else ""
        iv_str = (
            f"{rec.current_iv*100:.1f}%"
            if rec.current_iv else "n/a"
        )
        hv_str = (
            f"{rec.current_hv*100:.1f}%"
            if rec.current_hv else "n/a"
        )

        lines = [
            f"\n{'='*60}",
            f"  TRADE ANALYSIS: {rec.ticker}",
            f"{'='*60}",
            f"  Action:     {'[OPEN TRADE]' if rec.action == 'open' else '[PASS]'}",
            f"  Strategy:   {rec.strategy}",
            f"  Regime:     {rec.regime.replace('_', ' ').title()}",
            f"  VIX:        {rec.vix:.1f}",
            f"  IV (ATM):   {iv_str}  |  HV20: {hv_str}{vrp_str}",
            f"  IV Rank:    {rec.iv_rank:.0f}  |  IV Pct: {rec.iv_percentile:.0f}",
            f"  Earnings:   {rec.next_earnings_date or 'Unknown'}"
            + (f" ({rec.days_until_earnings}d away)" if rec.days_until_earnings is not None else ""),
        ]

        if rec.action == 'open':
            lines += [
                f"\n  Trade Construction:",
            ]
            for leg in rec.legs:
                lines.append(
                    f"    {leg['action']:4} {leg['type']:4} @ ${leg['strike']:.1f}  "
                    f"mid=${leg['mid']:.2f}"
                    + (f"  delta={leg.get('delta', '')}" if 'delta' in leg else "")
                )
            lines += [
                f"\n  Net Credit:     ${rec.entry_credit:.2f}/contract",
                f"  Max Profit:     ${rec.max_profit:.2f} ({rec.suggested_contracts} contracts)",
                f"  Max Loss:       {'Undefined (2x stop)' if rec.max_loss is None else f'${rec.max_loss:.2f}'}",
                f"  POP (est.):     {rec.probability_of_profit*100:.0f}%",
                f"  Contracts:      {rec.suggested_contracts}",
                f"\n  Target Exit:    50% profit (${rec.max_profit*0.5:.2f}) or 21 DTE",
                f"  Stop Loss:      2x credit = ${rec.entry_credit * 2 * 100 * rec.suggested_contracts:.2f}",
            ]

        lines.append(f"\n  Rationale:")
        for r in rec.rationale:
            lines.append(f"    - {r}")

        if rec.warnings:
            lines.append(f"\n  Warnings:")
            for w in rec.warnings:
                lines.append(f"    [!] {w}")

        lines.append(f"{'='*60}\n")
        return '\n'.join(lines)
