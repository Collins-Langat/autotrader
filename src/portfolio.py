"""
Portfolio Tracker — tracks open positions, P&L, and portfolio-level Greeks.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


PORTFOLIO_FILE       = Path(__file__).parent.parent / 'portfolio.json'
PAPER_PORTFOLIO_FILE = Path(__file__).parent.parent / 'paper_portfolio.json'


@dataclass
class Position:
    ticker: str
    strategy: str
    entry_date: str
    expiration: str
    dte_at_entry: int
    legs: list[dict]
    entry_credit: float
    contracts: int
    status: str = 'open'     # open, closed
    close_date: Optional[str] = None
    close_debit: Optional[float] = None
    pnl: Optional[float] = None
    notes: str = ''
    paper: bool = False      # True = paper trade, False = live trade


class Portfolio:
    def __init__(self, value: float = 25000, paper: bool = False):
        self.value = value
        self.paper = paper
        self._file = PAPER_PORTFOLIO_FILE if paper else PORTFOLIO_FILE
        self.positions: list[Position] = []
        self._load()

    def _load(self):
        if self._file.exists():
            with open(self._file, 'r') as f:
                data = json.load(f)
                self.value = data.get('portfolio_value', self.value)
                raw = data.get('positions', [])
                # Graceful load: ignore unknown keys so old files still load
                valid_fields = {f.name for f in Position.__dataclass_fields__.values()}
                self.positions = [
                    Position(**{k: v for k, v in p.items() if k in valid_fields})
                    for p in raw
                ]

    def _save(self):
        data = {
            'portfolio_value': self.value,
            'paper': self.paper,
            'last_updated': datetime.now().isoformat(),
            'positions': [asdict(p) for p in self.positions],
        }
        with open(self._file, 'w') as f:
            json.dump(data, f, indent=2)

    def add_position(self, position: Position):
        self.positions.append(position)
        self._save()
        print(f"[+] Position added: {position.ticker} {position.strategy}")

    def close_position(self, ticker: str, close_debit: float):
        for pos in self.positions:
            if pos.ticker == ticker and pos.status == 'open':
                pos.status = 'closed'
                pos.close_date = datetime.now().strftime('%Y-%m-%d')
                pos.close_debit = close_debit
                pos.pnl = round(
                    (pos.entry_credit - close_debit) * 100 * pos.contracts, 2
                )
                self._save()
                print(f"[+] Position closed: {ticker} | P&L: ${pos.pnl:+.2f}")
                return pos
        print(f"No open position found for {ticker}")
        return None

    def open_positions(self) -> list[Position]:
        return [p for p in self.positions if p.status == 'open']

    def closed_positions(self) -> list[Position]:
        return [p for p in self.positions if p.status == 'closed']

    def total_pnl(self) -> float:
        return sum(p.pnl for p in self.closed_positions() if p.pnl)

    def win_rate(self) -> float:
        closed = self.closed_positions()
        if not closed:
            return 0.0
        winners = [p for p in closed if p.pnl and p.pnl > 0]
        return round(len(winners) / len(closed), 4)

    def check_position_limit(self, ticker: str) -> bool:
        """HR-006: One position per underlying."""
        for pos in self.open_positions():
            if pos.ticker == ticker:
                return False  # Already have a position
        return True

    def buying_power_used(self) -> float:
        """Rough estimate of buying power used by open positions."""
        total = 0.0
        for pos in self.open_positions():
            total += pos.entry_credit * 100 * pos.contracts * 20  # rough BP estimate
        return total

    def buying_power_available(self) -> float:
        used = self.buying_power_used()
        return self.value - used

    def summary(self) -> dict:
        open_pos = self.open_positions()
        closed_pos = self.closed_positions()
        return {
            'portfolio_value': self.value,
            'open_positions': len(open_pos),
            'closed_positions': len(closed_pos),
            'total_pnl': self.total_pnl(),
            'win_rate': f"{self.win_rate()*100:.1f}%",
            'open_tickers': [p.ticker for p in open_pos],
        }

    def display_summary(self):
        s = self.summary()
        print(f"\n{'='*50}")
        print(f"  PORTFOLIO SUMMARY")
        print(f"{'='*50}")
        print(f"  Value:          ${s['portfolio_value']:,.0f}")
        print(f"  Open positions: {s['open_positions']}")
        print(f"  Closed trades:  {s['closed_positions']}")
        print(f"  Total P&L:      ${s['total_pnl']:+,.2f}")
        print(f"  Win Rate:       {s['win_rate']}")
        if s['open_tickers']:
            print(f"  Open tickers:   {', '.join(s['open_tickers'])}")
        print(f"{'='*50}\n")
