"""
Paper Trader -- Automated paper trading mode.

On each run() cycle:
  1. Re-prices all open paper positions using the live options chain
  2. Auto-closes any position that hits: 50% profit, 2x credit stop, 21 DTE
  3. Scans the watchlist via DecisionEngine
  4. Auto-enters any [OPEN TRADE] that passes all entry rules

Paper trades are stored in paper_portfolio.json, separate from portfolio.json.
All the same risk rules apply (earnings blackout, IV rank, position sizing, etc.).
"""

from datetime import datetime, date
from typing import Optional

from .decision_engine import DecisionEngine
from .portfolio import Portfolio, Position
from .market_data import get_options_chain


# Exit constants -- mirrors backtester / knowledge base rules
PROFIT_TARGET      = 0.50   # Close at 50% of credit collected
STOP_LOSS_MULTIPLE = 2.0    # Close when loss reaches 2x entry credit
EXIT_DTE           = 21     # Forced exit at 21 DTE

DEFAULT_WATCHLIST = [
    'SPY', 'QQQ', 'IWM', 'GLD', 'TLT',
    'AAPL', 'MSFT', 'TSLA', 'AMZN', 'NVDA',
    'AMD', 'META', 'GOOGL',
]


class PaperTrader:
    """
    Automated paper trading engine.

    Usage:
        pt = PaperTrader(portfolio_value=25000)
        summary = pt.run()                  # full cycle: exits + entries
        exits   = pt.check_exits_only()     # just re-price and close
        entries = pt.scan_entries_only()    # just scan for new trades
        mtm     = pt.get_positions_with_pnl()  # live mark-to-market
    """

    def __init__(self, portfolio_value: float = 25000, watchlist: list = None):
        self.portfolio_value = portfolio_value
        self.watchlist = watchlist or DEFAULT_WATCHLIST
        self.portfolio = Portfolio(value=portfolio_value, paper=True)
        self.engine = DecisionEngine(portfolio_value=portfolio_value)

    # ── Public interface ───────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Full paper trading cycle.
        First processes exits on open positions, then scans for new entries.
        Returns a summary dict with what happened.
        """
        exits   = self._process_exits()
        entries = self._scan_entries()

        return {
            'timestamp':   datetime.now().strftime('%Y-%m-%d %H:%M'),
            'exits':       exits,
            'entered':     [e for e in entries if e['action'] == 'entered'],
            'skipped':     [e for e in entries if e['action'] != 'entered'],
            'open_count':  len(self.portfolio.open_positions()),
            'closed_count': len(self.portfolio.closed_positions()),
        }

    def scan_entries_only(self) -> dict:
        """Scan for new entries only, without touching existing positions."""
        entries = self._scan_entries()
        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'entered': [e for e in entries if e['action'] == 'entered'],
            'skipped': [e for e in entries if e['action'] != 'entered'],
        }

    def check_exits_only(self) -> list:
        """Re-price open positions and close any that hit exit conditions."""
        return self._process_exits()

    def get_positions_with_pnl(self) -> list[dict]:
        """
        Return all open paper positions with live mark-to-market P&L.
        Tries to re-price each position against the current options chain.
        """
        results = []
        today = date.today()

        for pos in self.portfolio.open_positions():
            try:
                expiry        = date.fromisoformat(pos.expiration)
            except Exception:
                expiry        = today
            dte_remaining = max((expiry - today).days, 0)

            current_debit = self._price_position(pos)
            if current_debit is not None:
                pnl_per_share = pos.entry_credit - current_debit
                pnl_dollars   = round(pnl_per_share * 100 * pos.contracts, 2)
                pnl_pct       = round(pnl_per_share / pos.entry_credit * 100, 1) if pos.entry_credit else 0.0
                # Proximity to exit triggers
                at_profit = pnl_per_share >= pos.entry_credit * PROFIT_TARGET
                at_stop   = pnl_per_share <= -(STOP_LOSS_MULTIPLE * pos.entry_credit)
                at_dte    = dte_remaining <= EXIT_DTE
            else:
                pnl_dollars = pnl_pct = None
                current_debit = None
                at_profit = at_stop = at_dte = False

            results.append({
                'ticker':        pos.ticker,
                'strategy':      pos.strategy,
                'entry_date':    pos.entry_date,
                'expiration':    pos.expiration,
                'dte_remaining': dte_remaining,
                'entry_credit':  pos.entry_credit,
                'current_debit': current_debit,
                'contracts':     pos.contracts,
                'unrealized_pnl': pnl_dollars,
                'pnl_pct':       pnl_pct,
                'at_profit_target': at_profit,
                'at_stop_loss':     at_stop,
                'at_dte_exit':      at_dte,
                'legs':          pos.legs,
            })

        return results

    # ── Internal: entries ─────────────────────────────────────────────────────

    def _scan_entries(self) -> list[dict]:
        """Scan the watchlist and enter any qualifying trades."""
        results = []
        for ticker in self.watchlist:
            try:
                results.append(self._try_enter(ticker))
            except Exception as e:
                results.append({
                    'ticker': ticker,
                    'action': 'error',
                    'reason': str(e),
                })
        return results

    def _try_enter(self, ticker: str) -> dict:
        """Analyze a single ticker. If [OPEN TRADE], add to paper portfolio."""
        # HR-006: one position per underlying
        if not self.portfolio.check_position_limit(ticker):
            return {
                'ticker': ticker,
                'action': 'skip',
                'reason': 'HR-006: already have an open position',
            }

        rec = self.engine.analyze(ticker)

        if rec.action != 'open':
            # Surface the first FAIL reason for clarity
            fail_msg = next(
                (r for r in rec.rationale if r.startswith('FAIL')),
                rec.rationale[-1] if rec.rationale else 'No entry signal',
            )
            return {'ticker': ticker, 'action': 'skip', 'reason': fail_msg}

        # Fetch the expiration date from the live chain
        chain = get_options_chain(ticker, target_dte=45)
        exp   = chain['expiration'].iloc[0] if chain is not None and not chain.empty else 'unknown'
        dte   = int(chain['dte'].iloc[0])   if chain is not None and not chain.empty else 45

        pos = Position(
            ticker=ticker,
            strategy=rec.strategy,
            entry_date=datetime.now().strftime('%Y-%m-%d'),
            expiration=exp,
            dte_at_entry=dte,
            legs=rec.legs,
            entry_credit=rec.entry_credit,
            contracts=rec.suggested_contracts,
            paper=True,
        )
        self.portfolio.add_position(pos)

        return {
            'ticker':     ticker,
            'action':     'entered',
            'strategy':   rec.strategy,
            'credit':     rec.entry_credit,
            'expiration': exp,
            'dte':        dte,
            'legs':       rec.legs,
        }

    # ── Internal: exits ───────────────────────────────────────────────────────

    def _process_exits(self) -> list[dict]:
        """Check every open position for exit conditions."""
        exits = []
        for pos in list(self.portfolio.open_positions()):  # snapshot — list may shrink
            try:
                result = self._check_exit(pos)
                if result:
                    exits.append(result)
            except Exception as e:
                exits.append({'ticker': pos.ticker, 'reason': f'pricing error: {e}', 'pnl': None})
        return exits

    def _check_exit(self, pos: Position) -> Optional[dict]:
        """
        Re-price one position and close it if an exit rule fires.
        Returns a dict describing the exit, or None if no exit is triggered.
        """
        today = date.today()
        try:
            expiry = date.fromisoformat(pos.expiration)
        except Exception:
            return None

        dte_remaining = (expiry - today).days

        # ── EX-003: 21 DTE forced exit ──
        if dte_remaining <= EXIT_DTE:
            current_debit = self._price_position(pos)
            if current_debit is None:
                current_debit = pos.entry_credit * 0.50  # conservative fallback
            self.portfolio.close_position(pos.ticker, current_debit)
            return {
                'ticker':       pos.ticker,
                'reason':       f'21 DTE forced exit ({dte_remaining}d left)',
                'entry_credit': pos.entry_credit,
                'close_debit':  round(current_debit, 4),
                'pnl':          round((pos.entry_credit - current_debit) * 100 * pos.contracts, 2),
            }

        # Need current market price to check profit / stop
        current_debit = self._price_position(pos)
        if current_debit is None:
            return None  # Can't price today — leave it open

        pnl_per_share = pos.entry_credit - current_debit

        # ── EX-001: 50% profit target ──
        if pnl_per_share >= pos.entry_credit * PROFIT_TARGET:
            self.portfolio.close_position(pos.ticker, current_debit)
            return {
                'ticker':       pos.ticker,
                'reason':       '50% profit target',
                'entry_credit': pos.entry_credit,
                'close_debit':  round(current_debit, 4),
                'pnl':          round(pnl_per_share * 100 * pos.contracts, 2),
            }

        # ── EX-002 / HR-003: 2x credit stop loss ──
        if pnl_per_share <= -(STOP_LOSS_MULTIPLE * pos.entry_credit):
            self.portfolio.close_position(pos.ticker, current_debit)
            return {
                'ticker':       pos.ticker,
                'reason':       '2x credit stop loss',
                'entry_credit': pos.entry_credit,
                'close_debit':  round(current_debit, 4),
                'pnl':          round(pnl_per_share * 100 * pos.contracts, 2),
            }

        return None  # No exit triggered

    # ── Internal: position pricing ────────────────────────────────────────────

    def _price_position(self, pos: Position) -> Optional[float]:
        """
        Re-price an open position using the current live options chain.

        Fetches the exact expiry stored on the position. For each leg, finds the
        matching strike (or nearest if the exact strike has no quotes) and
        computes the mid price. Returns the total cost-to-close per share,
        or None if the chain is unavailable.

        SELL legs cost money to close (we buy them back).
        BUY  legs return money when closed (we sell them).
        """
        try:
            chain = get_options_chain(pos.ticker, expiry=pos.expiration)
            if chain is None or chain.empty:
                return None

            total_debit = 0.0

            for leg in pos.legs:
                strike   = float(leg['strike'])
                opt_type = leg['type'].lower()   # 'call' or 'put'
                action   = leg['action']         # 'SELL' or 'BUY'

                # Match exact strike first
                matches = chain[
                    (abs(chain['strike'] - strike) < 0.01) &
                    (chain['option_type'] == opt_type)
                ]

                if matches.empty:
                    # Nearest available strike as fallback
                    same_type = chain[chain['option_type'] == opt_type].copy()
                    if same_type.empty:
                        return None
                    same_type['_diff'] = abs(same_type['strike'] - strike)
                    matches = same_type.nsmallest(1, '_diff')

                row = matches.iloc[0]
                mid = float(row.get('mid', row.get('lastPrice', 0)) or 0)

                if action == 'SELL':
                    total_debit += mid   # buying back a short costs us money
                else:
                    total_debit -= mid   # selling a long returns money

            return round(max(total_debit, 0.0), 4)

        except Exception:
            return None
