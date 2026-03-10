"""
Long Options Evaluator
Evaluates whether buying a call or put on a given ticker makes sense.

Key difference from the short-premium engine:
  - IV rank threshold is INVERTED: want LOW IV (cheap options), not high
  - Theta is an expense, not income
  - Edge comes from directional conviction + correct timing, not volatility mean-reversion

Outputs per evaluation:
  - IV assessment (cheap / fair / elevated / expensive)
  - Expected move at target DTE (1-SD and 2-SD)
  - Strike candidates at ATM, near-OTM, far-OTM, deep-OTM
  - Debit spread alternatives (buy spread instead of naked long to cut theta)
  - Earnings / macro warnings
"""

import math
from dataclasses import dataclass, field
from typing import Optional

from .market_data import (
    get_underlying_data, get_historical_volatility,
    get_iv_rank, get_options_chain, get_earnings_date,
)
from .greeks import calculate_greeks, dte_to_years
from .macro_calendar import check_macro_blackout


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class StrikeCandidate:
    """Analysis for a single strike / expiry combination."""
    label: str              # "ATM", "Near OTM (35d)", "Far OTM (20d)", "Deep OTM (10d)"
    strike: float
    delta: float            # signed delta from Black-Scholes at that strike's IV
    premium: float          # real mid price from options chain (per share)
    break_even: float       # strike + premium (call) or strike - premium (put)
    move_needed: float      # abs(break_even - current_price) — how far the stock must move
    move_needed_pct: float  # move_needed / price * 100
    daily_theta: float      # $ per share per day (negative = costs you)
    theta_pct_daily: float  # |daily_theta| / premium * 100  — daily time decay as % of cost
    pop_to_be: float        # probability option finishes above/below break-even at expiry
    vega: float             # $ per 1% IV move
    gamma: float
    implied_vol: float      # option's own IV from the chain


@dataclass
class DebitSpread:
    """A vertical debit spread (buy lower strike, sell higher for calls; reverse for puts)."""
    label: str
    long_strike: float
    short_strike: float
    net_debit: float        # cost to open per share
    max_profit: float       # width - net_debit per share
    break_even: float       # where you start making money at expiry
    max_return_pct: float   # max_profit / net_debit * 100
    theta_per_day: float    # net daily theta (reduced vs naked long)
    theta_vs_naked_pct: float   # how much of the naked-long theta you eliminated
    long_premium: float
    short_premium: float
    width: float            # strike distance


@dataclass
class LongOptionEval:
    """Full evaluation result for buying an option."""
    ticker: str
    opt_type: str           # 'call' or 'put'
    target_dte: int
    expiry: str
    actual_dte: int
    price: float

    # Volatility context
    iv_rank: float
    iv_percentile: float
    current_iv: float       # ATM implied vol (annualized)
    hv20: float             # 20-day realized vol
    vrp_ratio: float        # IV / HV — >1 means options are expensive vs realized vol

    # IV verdict
    iv_assessment: str      # 'cheap' | 'fair' | 'elevated' | 'expensive'
    iv_note: str            # human-readable note

    # Expected move at target DTE (IV-based formula)
    expected_move_1sd: float
    expected_move_2sd: float
    expected_move_pct_1sd: float
    upper_1sd: float
    lower_1sd: float
    # Straddle-based expected move (more market-accurate)
    straddle_move: float = 0.0
    straddle_upper: float = 0.0
    straddle_lower: float = 0.0

    # Strike analysis
    candidates: list = field(default_factory=list)   # list[StrikeCandidate]

    # Debit spread alternatives
    debit_spreads: list = field(default_factory=list)  # list[DebitSpread]

    # Context
    warnings: list = field(default_factory=list)
    next_earnings_date: Optional[str] = None
    days_until_earnings: Optional[int] = None
    upcoming_macro_events: list = field(default_factory=list)

    # Quality scoring
    trade_score: float = 0.0
    trade_score_label: str = ''
    plain_english: str = ''

    error: Optional[str] = None


# ── Evaluator ─────────────────────────────────────────────────────────────────

class LongEvaluator:
    """
    Evaluates a long call or put buy on a given ticker and DTE.

    Usage:
        ev = LongEvaluator()
        result = ev.evaluate('NVDA', 'call', target_dte=60)
    """

    RISK_FREE_RATE = 0.05

    # Delta targets for candidate strikes
    DELTA_TARGETS = [
        ("ATM",              0.50),
        ("Near OTM (35d)",   0.35),
        ("Far OTM (20d)",    0.20),
        ("Deep OTM (10d)",   0.10),
    ]

    def evaluate(
        self,
        ticker: str,
        opt_type: str = 'call',
        target_dte: int = 60,
    ) -> LongOptionEval:
        """
        Full evaluation pipeline.
        Returns a LongOptionEval with all analysis populated.
        """
        opt_type = opt_type.lower()
        r = self.RISK_FREE_RATE

        # ── Step 1: Underlying price ──────────────────────────────────────────
        try:
            underlying = get_underlying_data(ticker)
            price = underlying['price']
        except Exception as e:
            return self._error_result(ticker, opt_type, target_dte, str(e))

        # ── Step 2: Historical vol ────────────────────────────────────────────
        hv_data = get_historical_volatility(ticker)
        hv20    = hv_data.get('hv20', 0.25) or 0.25

        # ── Step 3: Options chain ─────────────────────────────────────────────
        chain = get_options_chain(ticker, target_dte=target_dte)
        if chain is None or chain.empty:
            return self._error_result(ticker, opt_type, target_dte, "No options chain available.")

        actual_dte = int(chain['dte'].iloc[0])
        expiry     = str(chain['expiration'].iloc[0])
        T          = dte_to_years(actual_dte)

        # ── Step 4: ATM implied vol ───────────────────────────────────────────
        current_iv = self._get_atm_iv(chain, price, hv20)

        # ── Step 5: IV rank ───────────────────────────────────────────────────
        iv_rank_data  = get_iv_rank(ticker, current_iv)
        iv_rank       = iv_rank_data.get('iv_rank',       50.0)
        iv_percentile = iv_rank_data.get('iv_percentile', 50.0)
        vrp_ratio     = round(current_iv / hv20, 2) if hv20 > 0 else 1.0

        # ── Step 6: IV assessment ─────────────────────────────────────────────
        iv_assessment, iv_note = self._assess_iv(iv_rank)

        # ── Step 7: Expected move ─────────────────────────────────────────────
        em_1sd     = round(price * current_iv * math.sqrt(T), 2)
        em_2sd     = round(em_1sd * 2, 2)
        em_pct_1sd = round(em_1sd / price * 100, 1)
        upper_1sd  = round(price + em_1sd, 2)
        lower_1sd  = round(price - em_1sd, 2)

        # ── Step 7b: Straddle expected move (ATM call mid + ATM put mid) ──────
        straddle_move = 0.0
        try:
            atm_diff = abs(chain['strike'] - price)
            atm_strike = float(chain.loc[atm_diff.idxmin(), 'strike'])
            atm_call_row = chain[
                (chain['option_type'] == 'call') &
                (chain['strike'] == atm_strike)
            ]
            atm_put_row = chain[
                (chain['option_type'] == 'put') &
                (chain['strike'] == atm_strike)
            ]
            if not atm_call_row.empty and not atm_put_row.empty:
                c_mid = float(atm_call_row['mid'].iloc[0])
                p_mid = float(atm_put_row['mid'].iloc[0])
                if c_mid > 0 and p_mid > 0:
                    straddle_move = round(c_mid + p_mid, 2)
        except Exception:
            pass
        straddle_upper = round(price + straddle_move, 2) if straddle_move else 0.0
        straddle_lower = round(price - straddle_move, 2) if straddle_move else 0.0

        # ── Step 8: Strike candidates ─────────────────────────────────────────
        side       = chain[chain['option_type'] == opt_type].copy()
        candidates = []
        for label, abs_delta in self.DELTA_TARGETS:
            target_delta = abs_delta if opt_type == 'call' else -abs_delta
            cand = self._build_candidate(
                label, target_delta, side, price, T, r, current_iv, opt_type
            )
            if cand is not None:
                candidates.append(cand)

        # ── Step 9: Debit spreads ─────────────────────────────────────────────
        debit_spreads = []
        if len(candidates) >= 2:
            ds = self._build_spread(
                candidates[0], candidates[1], opt_type, price, T, r
            )
            if ds:
                debit_spreads.append(ds)
        if len(candidates) >= 3:
            ds2 = self._build_spread(
                candidates[1], candidates[2], opt_type, price, T, r
            )
            if ds2:
                debit_spreads.append(ds2)

        # ── Step 10: Earnings & macro warnings ───────────────────────────────
        warnings             = []
        next_earnings        = None
        days_until_earnings  = None
        upcoming_macro       = []

        try:
            e_info = get_earnings_date(ticker)
            next_earnings       = e_info.get('next_earnings_date')
            days_until_earnings = e_info.get('days_until_earnings')
            if e_info.get('has_earnings_in_window'):
                days = days_until_earnings
                warnings.append(
                    f"EARNINGS in {days} days ({next_earnings}): IV will CRUSH after the "
                    f"announcement — even a big gap can produce a loss if IV drops 30-50%. "
                    f"If you want to play earnings, buy before the IV run-up and SELL before "
                    f"the announcement, not after."
                )
        except Exception:
            pass

        try:
            blocked, imminent = check_macro_blackout(days_before=2)
            if blocked:
                evts = ', '.join(f"{e['type']} {e['date_str']}" for e in imminent)
                warnings.append(
                    f"HR-010 Macro blackout: {evts} is within 2 days. "
                    f"Gap risk is elevated — consider waiting."
                )
            from .macro_calendar import get_upcoming_events
            upcoming_macro = [
                {'type': e['type'], 'date': e['date_str'], 'days_away': e['days_away']}
                for e in get_upcoming_events(days_ahead=actual_dte)
            ]
        except Exception:
            pass

        # ── Short-DTE specific warnings ───────────────────────────────────────
        if actual_dte <= 1:
            theta_pct = (candidates[0].theta_pct_daily if candidates else 0)
            warnings.insert(0,
                f"0-1 DTE: This is essentially a same-day directional bet. "
                f"Theta decay is near-total overnight. The option loses ~100% of "
                f"its remaining value by end of day if the stock does not move "
                f"immediately and significantly. Suitable only for intraday plays "
                f"with a clear, immediate catalyst."
            )
        elif actual_dte <= 3:
            warnings.insert(0,
                f"{actual_dte} DTE: Very short-term — theta destroys value over "
                f"the weekend and overnight. You need the move to happen within "
                f"1-3 days. OTM strikes at this DTE are near-zero delta lottery tickets."
            )
        elif actual_dte <= 10:
            warnings.insert(0,
                f"{actual_dte} DTE: Short-term option — theta decay is rapid "
                f"(accelerates sharply inside 10 DTE). Any move must happen soon "
                f"or theta will erase most of the premium paid. Debit spreads are "
                f"especially effective at this DTE to cap the theta bleed."
            )

        if iv_assessment == 'expensive':
            warnings.append(
                f"IV rank {iv_rank:.0f} is very high — this is a SELLING environment. "
                f"Buying options here means paying a significant premium. "
                f"A debit spread will cut your vega and theta exposure."
            )
        elif iv_assessment == 'elevated':
            warnings.append(
                f"IV is elevated (rank {iv_rank:.0f}). A debit spread reduces "
                f"your vega exposure and lowers the break-even."
            )

        return LongOptionEval(
            ticker=ticker,
            opt_type=opt_type,
            target_dte=target_dte,
            expiry=expiry,
            actual_dte=actual_dte,
            price=price,
            iv_rank=iv_rank,
            iv_percentile=iv_percentile,
            current_iv=current_iv,
            hv20=hv20,
            vrp_ratio=vrp_ratio,
            iv_assessment=iv_assessment,
            iv_note=iv_note,
            expected_move_1sd=em_1sd,
            expected_move_2sd=em_2sd,
            expected_move_pct_1sd=em_pct_1sd,
            upper_1sd=upper_1sd,
            lower_1sd=lower_1sd,
            candidates=candidates,
            debit_spreads=debit_spreads,
            warnings=warnings,
            next_earnings_date=next_earnings,
            days_until_earnings=days_until_earnings,
            upcoming_macro_events=upcoming_macro,
            straddle_move=straddle_move,
            straddle_upper=straddle_upper,
            straddle_lower=straddle_lower,
            trade_score=self._compute_trade_score(
                iv_rank, iv_assessment, candidates, actual_dte, upcoming_macro
            ),
            trade_score_label=self._score_label(self._compute_trade_score(
                iv_rank, iv_assessment, candidates, actual_dte, upcoming_macro
            )),
            plain_english=self._plain_english(
                ticker, opt_type, price, actual_dte, expiry,
                iv_assessment, iv_rank, straddle_move, em_1sd, candidates,
                debit_spreads, upcoming_macro
            ),
        )

    # ── Quality score helpers ─────────────────────────────────────────────────

    def _compute_trade_score(
        self, iv_rank: float, iv_assessment: str,
        candidates: list, dte: int, macro_events: list
    ) -> float:
        """0-100 trade quality score for buying options (low IV = good)."""
        score = 0.0

        # IV rank (30 pts) — for buying, LOW IV is better
        if iv_rank < 20:   score += 30
        elif iv_rank < 30: score += 24
        elif iv_rank < 40: score += 16
        elif iv_rank < 50: score += 8

        # Break-even feasibility (30 pts) — is ATM break-even inside 1-SD?
        if candidates:
            atm = candidates[0]
            move_needed_pct = atm.move_needed_pct
            if move_needed_pct < 5:    score += 30
            elif move_needed_pct < 8:  score += 22
            elif move_needed_pct < 12: score += 12
            elif move_needed_pct < 18: score += 5

        # DTE (20 pts) — more time = better for buyers
        if dte >= 60:   score += 20
        elif dte >= 45: score += 16
        elif dte >= 30: score += 10
        elif dte >= 14: score += 4

        # Macro event risk (20 pts) — fewer events = less gap risk
        n = len(macro_events)
        if n == 0:   score += 20
        elif n == 1: score += 13
        elif n == 2: score += 6

        return round(min(score, 100.0), 1)

    @staticmethod
    def _score_label(score: float) -> str:
        if score >= 80: return 'Strong'
        if score >= 60: return 'Good'
        if score >= 40: return 'Marginal'
        return 'Avoid'

    def _plain_english(
        self, ticker: str, opt_type: str, price: float, dte: int, expiry: str,
        iv_assessment: str, iv_rank: float, straddle_move: float, em_1sd: float,
        candidates: list, debit_spreads: list, macro_events: list
    ) -> str:
        """Plain-language summary for a long option position."""
        direction = 'rises' if opt_type == 'call' else 'falls'
        direction2 = 'above' if opt_type == 'call' else 'below'
        opt_word = 'call' if opt_type == 'call' else 'put'

        atm = candidates[0] if candidates else None
        move_ref = straddle_move if straddle_move > 0 else em_1sd

        lines = []
        lines.append("WHAT IS THIS TRADE?")
        lines.append(
            f"You are buying a {opt_word} option on {ticker} (currently ${price:.2f}). "
            f"This is a bet that {ticker} {direction} significantly in the next {dte} days "
            f"(expiry: {expiry})."
        )

        lines.append('')
        lines.append("HOW OPTIONS WORK:")
        lines.append(
            f"You pay a premium upfront. If {ticker} {direction} {direction2} your break-even "
            f"price by expiry, you make money. If it does not, you lose the premium you paid. "
            f"Unlike selling options, your maximum loss is limited to what you paid."
        )

        if atm:
            lines.append('')
            lines.append("THE NUMBERS (ATM option):")
            lines.append(f"  Cost to buy: ${atm.premium:.2f} per share (${atm.premium*100:.0f} per contract)")
            lines.append(f"  Break-even price: ${atm.break_even:.2f} — {ticker} needs to move {atm.move_needed_pct:.1f}% from here")
            lines.append(f"  Probability of reaching break-even: ~{atm.pop_to_be*100:.0f}%")
            lines.append(f"  Daily time cost: ${abs(atm.daily_theta):.2f}/day ({atm.theta_pct_daily:.1f}% of your investment per day)")

        lines.append('')
        lines.append("WHAT THE MARKET EXPECTS:")
        em_val = straddle_move if straddle_move > 0 else em_1sd
        lines.append(
            f"The options market is currently pricing in a move of roughly "
            f"${em_val:.2f} (up or down) over the next {dte} days. "
            f"Your break-even requires only the move in your direction."
        )

        lines.append('')
        iv_plain = {
            'cheap':    "OPTIONS ARE CHEAP right now — this is a good time to buy.",
            'fair':     "Options are fairly priced. Acceptable if you have strong conviction.",
            'elevated': "OPTIONS ARE EXPENSIVE. You are paying above-average premium.",
            'expensive':"OPTIONS ARE VERY EXPENSIVE. This is typically a better time to sell, not buy.",
        }
        lines.append(f"CURRENT OPTIONS COST: {iv_plain.get(iv_assessment, iv_assessment)}")
        lines.append(f"(IV Rank: {iv_rank:.0f}/100 — the higher this number, the more expensive options are)")

        if debit_spreads:
            ds = debit_spreads[0]
            lines.append('')
            lines.append("CHEAPER ALTERNATIVE — DEBIT SPREAD:")
            lines.append(
                f"Instead of buying one option for ${atm.premium:.2f}, you can buy a "
                f"{ds.label} for just ${ds.net_debit:.2f}. "
                f"Maximum return: {ds.max_return_pct:.0f}%. "
                f"This limits your upside but significantly reduces your daily time cost."
            )

        if macro_events:
            next_e = macro_events[0]
            lines.append('')
            lines.append(
                f"RISK EVENT: {next_e['type']} in {next_e['days_away']} days ({next_e['date']}). "
                f"This could cause a big move — good or bad for your position. "
                f"Make sure your expiry is AFTER this date to benefit from the potential move."
            )

        return '\n'.join(lines)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_atm_iv(self, chain, price: float, fallback: float) -> float:
        """Extract IV of the nearest-ATM option."""
        try:
            diffs = abs(chain['strike'] - price)
            atm_rows = chain[diffs == diffs.min()]
            iv = float(atm_rows['impliedVolatility'].mean())
            return iv if iv > 0.01 else fallback
        except Exception:
            return fallback

    def _assess_iv(self, iv_rank: float) -> tuple[str, str]:
        """Return (assessment, note) for the IV rank."""
        if iv_rank < 25:
            return ('cheap',
                    f"IV rank {iv_rank:.0f} -- options are CHEAP. This is a good "
                    f"environment to buy. You're paying below-average premium.")
        elif iv_rank < 40:
            return ('fair',
                    f"IV rank {iv_rank:.0f} -- options are fairly priced. "
                    f"Acceptable to buy if you have strong directional conviction.")
        elif iv_rank < 60:
            return ('elevated',
                    f"IV rank {iv_rank:.0f} -- options are ELEVATED. You're paying "
                    f"above-average premium. A debit spread reduces this risk.")
        else:
            return ('expensive',
                    f"IV rank {iv_rank:.0f} -- options are EXPENSIVE. This is a "
                    f"premium-SELLING environment. Buying here is costly.")

    def _find_strike_by_delta(
        self, side, target_delta: float, price: float, T: float, r: float
    ):
        """
        Find the row in the option chain closest to target_delta.
        Uses each option's own impliedVolatility (to capture skew correctly).
        Returns the chain row or None.
        """
        best_row  = None
        best_diff = float('inf')

        for _, row in side.iterrows():
            iv_row = float(row.get('impliedVolatility', 0) or 0)
            if iv_row < 0.01:
                continue
            try:
                strike = float(row['strike'])
                opt_type = str(row['option_type'])
                g = calculate_greeks(opt_type, price, strike, T, r, iv_row)
                diff = abs(g.delta - target_delta)
                if diff < best_diff:
                    best_diff = diff
                    best_row  = row
            except Exception:
                continue

        return best_row

    def _mid(self, row) -> float:
        """Best mid price from a chain row."""
        bid = float(row.get('bid', 0) or 0)
        ask = float(row.get('ask', 0) or 0)
        if bid > 0 and ask > 0:
            return round((bid + ask) / 2, 2)
        last = float(row.get('lastPrice', 0) or 0)
        return round(last, 2) if last > 0 else 0.0

    def _build_candidate(
        self, label: str, target_delta: float,
        side, price: float, T: float, r: float,
        atm_iv: float, opt_type: str
    ) -> Optional[StrikeCandidate]:
        """Build a StrikeCandidate for a given delta target."""
        try:
            row = self._find_strike_by_delta(side, target_delta, price, T, r)
            if row is None:
                return None

            strike   = float(row['strike'])
            iv_row   = float(row.get('impliedVolatility', atm_iv) or atm_iv)
            premium  = self._mid(row)
            if premium <= 0:
                return None

            g = calculate_greeks(opt_type, price, strike, T, r, iv_row)

            # Break-even and move needed
            if opt_type == 'call':
                break_even = round(strike + premium, 2)
            else:
                break_even = round(strike - premium, 2)
            move_needed     = round(abs(break_even - price), 2)
            move_needed_pct = round(move_needed / price * 100, 1)

            # Daily theta (theta is per year in BS; divide by 365 for daily)
            # calculate_greeks already returns daily theta
            daily_theta     = round(g.theta, 4)    # already $/day
            theta_pct_daily = round(abs(daily_theta) / premium * 100, 2) if premium > 0 else 0.0

            # POP to break-even: probability stock ends above/below break-even
            # Use delta of a hypothetical option at the break-even strike
            try:
                be_g = calculate_greeks('call', price, break_even, T, r, iv_row)
                pop_to_be = round(be_g.delta, 3) if opt_type == 'call' else round(1 - be_g.delta, 3)
            except Exception:
                pop_to_be = round(max(0.0, abs(g.delta) - 0.05), 3)

            return StrikeCandidate(
                label=label,
                strike=strike,
                delta=round(g.delta, 3),
                premium=premium,
                break_even=break_even,
                move_needed=move_needed,
                move_needed_pct=move_needed_pct,
                daily_theta=daily_theta,
                theta_pct_daily=theta_pct_daily,
                pop_to_be=pop_to_be,
                vega=round(g.vega, 4),
                gamma=round(g.gamma, 6),
                implied_vol=round(iv_row * 100, 1),
            )
        except Exception:
            return None

    def _build_spread(
        self,
        long_leg: StrikeCandidate,
        short_leg: StrikeCandidate,
        opt_type: str,
        price: float,
        T: float,
        r: float,
    ) -> Optional[DebitSpread]:
        """
        Build a vertical debit spread from two StrikeCandidates.
        For calls: buy lower strike, sell higher strike.
        For puts:  buy higher strike, sell lower strike.
        """
        try:
            if opt_type == 'call':
                long_s  = min(long_leg.strike, short_leg.strike)
                short_s = max(long_leg.strike, short_leg.strike)
                # Long is the cheaper/lower strike for calls
                if long_leg.strike == long_s:
                    lp = long_leg.premium
                    sp = short_leg.premium
                    lt = long_leg.daily_theta
                    st_theta = short_leg.daily_theta
                else:
                    lp = short_leg.premium
                    sp = long_leg.premium
                    lt = short_leg.daily_theta
                    st_theta = long_leg.daily_theta
            else:
                long_s  = max(long_leg.strike, short_leg.strike)
                short_s = min(long_leg.strike, short_leg.strike)
                if long_leg.strike == long_s:
                    lp = long_leg.premium
                    sp = short_leg.premium
                    lt = long_leg.daily_theta
                    st_theta = short_leg.daily_theta
                else:
                    lp = short_leg.premium
                    sp = long_leg.premium
                    lt = short_leg.daily_theta
                    st_theta = long_leg.daily_theta

            width      = round(abs(short_s - long_s), 2)
            net_debit  = round(lp - sp, 2)
            if net_debit <= 0.05:
                return None

            max_profit = round(width - net_debit, 2)
            if max_profit <= 0:
                return None

            if opt_type == 'call':
                break_even = round(long_s + net_debit, 2)
            else:
                break_even = round(long_s - net_debit, 2)

            max_return_pct   = round(max_profit / net_debit * 100, 1)
            net_theta        = round(lt + st_theta, 4)    # both are negative; short offsets
            naked_theta      = long_leg.daily_theta

            # How much of theta we eliminated (as a %)
            if abs(naked_theta) > 0:
                theta_vs_naked = round((1 - abs(net_theta) / abs(naked_theta)) * 100, 1)
            else:
                theta_vs_naked = 0.0

            label = f"Buy ${long_s:.0f} / Sell ${short_s:.0f} {opt_type.upper()} spread"

            return DebitSpread(
                label=label,
                long_strike=long_s,
                short_strike=short_s,
                net_debit=net_debit,
                max_profit=max_profit,
                break_even=break_even,
                max_return_pct=max_return_pct,
                theta_per_day=net_theta,
                theta_vs_naked_pct=theta_vs_naked,
                long_premium=lp,
                short_premium=sp,
                width=width,
            )
        except Exception:
            return None

    def _error_result(
        self, ticker: str, opt_type: str, target_dte: int, msg: str
    ) -> LongOptionEval:
        return LongOptionEval(
            ticker=ticker, opt_type=opt_type, target_dte=target_dte,
            expiry='', actual_dte=0, price=0,
            iv_rank=0, iv_percentile=0, current_iv=0, hv20=0, vrp_ratio=0,
            iv_assessment='unknown', iv_note='',
            expected_move_1sd=0, expected_move_2sd=0,
            expected_move_pct_1sd=0, upper_1sd=0, lower_1sd=0,
            error=msg,
        )
