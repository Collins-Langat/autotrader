#!/usr/bin/env python3
"""
Options Trader — Main Entry Point
Run: python main.py

Usage:
  python main.py                                         # Scan default watchlist
  python main.py --ticker AAPL                           # Analyze single ticker
  python main.py --ticker SPY --strategy "Iron Condor"
  python main.py --portfolio                             # Show portfolio summary
  python main.py --kb                                    # Show knowledge base summary
  python main.py --backtest --ticker SPY                 # Run backtest (default dates)
  python main.py --backtest --ticker SPY --start 2022-01-01 --end 2024-01-01
  python main.py --backtest --ticker SPY --strategy "Iron Condor" --start 2021-01-01
  python main.py --paper-trade                           # Run one paper trading cycle
  python main.py --paper-trade --exits-only              # Check exits only
  python main.py --paper-trade --entries-only            # Scan entries only
  python main.py --paper-trade --positions               # Show open paper positions (MTM)
  python main.py --paper-portfolio                       # Show paper portfolio summary
  python main.py --long-options NVDA                     # Evaluate buying NVDA options (60 DTE, call)
  python main.py --long-options NVDA --type put          # Evaluate puts instead
  python main.py --long-options NVDA --type call --dte 30  # Custom DTE
  python main.py --report                                # Scan watchlist and generate PDF report
  python main.py --report --output ~/Documents           # Save PDF to a specific folder
  python main.py --undervalued                           # Screen watchlist for undervalued stocks
  python main.py --undervalued --watchlist AAPL MSFT AMD # Custom watchlist
  python main.py --undervalued --min-score 55            # Only show Watch List or better
"""

import argparse
import sys
from src.knowledge_base import KnowledgeBase
from src.decision_engine import DecisionEngine
from src.portfolio import Portfolio
from src.market_data import get_vix

# Default watchlist — liquid ETFs and high-IV stocks favored by TastyTrade
DEFAULT_WATCHLIST = [
    'SPY', 'QQQ', 'IWM', 'GLD', 'TLT',
    'AAPL', 'MSFT', 'TSLA', 'AMZN', 'NVDA',
    'AMD', 'META', 'GOOGL',
]


def scan_watchlist(portfolio_value: float, watchlist: list):
    """Scan the full watchlist and print recommendations."""
    engine = DecisionEngine(portfolio_value=portfolio_value)
    vix = get_vix()

    print(f"\n{'='*60}")
    print(f"  OPTIONS TRADER — WATCHLIST SCAN")
    print(f"  VIX: {vix:.1f}  |  Portfolio: ${portfolio_value:,.0f}")
    print(f"  Scanning {len(watchlist)} tickers...")
    print(f"{'='*60}\n")

    recommendations = []
    passes = []

    for ticker in watchlist:
        print(f"  Analyzing {ticker}...", end='\r')
        try:
            rec = engine.analyze(ticker)
            if rec.action == 'open':
                recommendations.append(rec)
            else:
                passes.append((ticker, rec.rationale[0] if rec.rationale else 'No reason'))
        except Exception as e:
            passes.append((ticker, f"Error: {e}"))

    print(f"  Scan complete. {len(recommendations)} opportunities found.\n")

    if recommendations:
        print(f"  [+] TRADE OPPORTUNITIES ({len(recommendations)}):")
        for rec in recommendations:
            print(engine.format_recommendation(rec))
    else:
        print("  No trade opportunities meet criteria right now.")
        print("  Top reasons for passing:\n")
        for ticker, reason in passes[:5]:
            print(f"    {ticker:8} — {reason}")

    print(f"\n  All tickers scanned: {', '.join(passes[i][0] for i in range(len(passes)))}")
    print(f"  Passed: {len(passes)}")


def analyze_single(ticker: str, strategy: str, portfolio_value: float):
    """Deep analysis of a single ticker."""
    engine = DecisionEngine(portfolio_value=portfolio_value)
    print(f"\nAnalyzing {ticker}...")

    try:
        rec = engine.analyze(ticker, strategy_name=strategy)
        print(engine.format_recommendation(rec))
    except Exception as e:
        print(f"\nError analyzing {ticker}: {e}")
        import traceback
        traceback.print_exc()


def show_knowledge_base():
    """Print knowledge base summary."""
    kb = KnowledgeBase()
    s = kb.summary()

    print(f"\n{'='*60}")
    print(f"  KNOWLEDGE BASE SUMMARY")
    print(f"{'='*60}")
    print(f"  Strategies loaded: {s['strategies_loaded']}")
    for name in s['strategy_names']:
        strategy = kb.get_strategy(name)
        entry = strategy.get('entry_conditions', {})
        print(f"    -{name}")
        print(f"      Entry: DTE {entry.get('dte_target', '?')} | "
              f"IV Rank >{entry.get('iv_rank_minimum', '?')} | "
              f"Delta {strategy.get('legs', [{}])[0].get('delta_target', '?')}")

    print(f"\n  Hard risk rules: {s['hard_risk_rules']}")
    print(f"  Entry rules:     {s['entry_rules']}")
    print(f"  Exit rules:      {s['exit_rules']}")
    print(f"  Market regimes:  {s['market_regimes']}")
    print(f"\n  Sources:")
    for src in s['sources']:
        print(f"    -{src.replace('_', ' ').title()}")
    print(f"{'='*60}\n")


def show_portfolio(portfolio_value: float):
    portfolio = Portfolio(value=portfolio_value)
    portfolio.display_summary()


def run_paper_trade(
    portfolio_value: float,
    watchlist: list,
    exits_only: bool = False,
    entries_only: bool = False,
    show_positions: bool = False,
):
    """Run one paper trading cycle and print a formatted summary."""
    from src.paper_trader import PaperTrader

    pt = PaperTrader(portfolio_value=portfolio_value, watchlist=watchlist)

    print(f"\n{'='*60}")
    print(f"  PAPER TRADER")
    print(f"  Portfolio: ${portfolio_value:,.0f}  |  Watchlist: {len(watchlist)} tickers")

    # ── Positions MTM only ──
    if show_positions:
        print(f"  Mode: Live mark-to-market")
        print(f"{'='*60}\n")
        positions = pt.get_positions_with_pnl()
        if not positions:
            print("  No open paper positions.")
        else:
            print(f"  {'Ticker':<8} {'Strategy':<18} {'Entry':>8} {'Current':>8} "
                  f"{'P&L $':>8} {'P&L %':>7} {'DTE':>4}  Flags")
            print(f"  {'-'*76}")
            for p in positions:
                ec   = f"${p['entry_credit']:.2f}" if p['entry_credit'] else '  n/a'
                cd   = f"${p['current_debit']:.2f}" if p['current_debit'] is not None else '   n/a'
                pnl  = f"${p['unrealized_pnl']:+.2f}" if p['unrealized_pnl'] is not None else '   n/a'
                pct  = f"{p['pnl_pct']:+.1f}%" if p['pnl_pct'] is not None else '   n/a'
                flags = []
                if p['at_profit_target']: flags.append('PROFIT')
                if p['at_stop_loss']:     flags.append('STOP')
                if p['at_dte_exit']:      flags.append('21DTE')
                flag_str = ' '.join(flags) if flags else ''
                print(
                    f"  {p['ticker']:<8} {p['strategy']:<18} "
                    f"{ec:>8} {cd:>8} {pnl:>8} {pct:>7} "
                    f"{p['dte_remaining']:>4}  {flag_str}"
                )
        print(f"\n{'='*60}\n")
        return

    # ── Exits only ──
    if exits_only:
        print(f"  Mode: Check exits only")
        print(f"{'='*60}\n")
        exits = pt.check_exits_only()
        if not exits:
            print("  No exits triggered.")
        else:
            print(f"  {len(exits)} position(s) closed:\n")
            for e in exits:
                pnl_str = f"${e['pnl']:+.2f}" if e['pnl'] is not None else 'n/a'
                print(f"  [-] {e['ticker']:<8}  {e['reason']:<35}  P&L: {pnl_str}")
        print(f"\n{'='*60}\n")
        return

    # ── Entries only ──
    if entries_only:
        print(f"  Mode: Scan for new entries")
        print(f"{'='*60}\n")
        result = pt.scan_entries_only()
        _print_entry_summary(result['entered'], result['skipped'])
        print(f"{'='*60}\n")
        return

    # ── Full cycle ──
    print(f"  Mode: Full cycle (exits then entries)")
    print(f"{'='*60}\n")
    result = pt.run()

    # Exits section
    if result['exits']:
        print(f"  EXITS ({len(result['exits'])}):")
        for e in result['exits']:
            pnl_str = f"${e['pnl']:+.2f}" if e['pnl'] is not None else 'n/a'
            print(f"  [-] {e['ticker']:<8}  {e['reason']:<35}  P&L: {pnl_str}")
        print()
    else:
        print("  No exits triggered.\n")

    # Entries section
    _print_entry_summary(result['entered'], result['skipped'])

    print(f"  Open positions: {result['open_count']}  |  Closed: {result['closed_count']}")
    print(f"  Timestamp: {result['timestamp']}")
    print(f"\n  Tip: Run 'streamlit run app.py' for the full Paper Trading dashboard.")
    print(f"{'='*60}\n")


def _print_entry_summary(entered: list, skipped: list):
    """Helper: print entry results."""
    if entered:
        print(f"  ENTRIES ({len(entered)}):")
        for e in entered:
            print(
                f"  [+] {e['ticker']:<8}  {e['strategy']:<18}  "
                f"Credit: ${e['credit']:.2f}  "
                f"Exp: {e['expiration']}  DTE: {e['dte']}"
            )
        print()
    else:
        print("  No new entries.\n")

    if skipped:
        print(f"  SKIPPED ({len(skipped)}):")
        for s in skipped[:8]:  # cap at 8 lines to avoid noise
            print(f"  [ ] {s['ticker']:<8}  {s['reason']}")
        if len(skipped) > 8:
            print(f"      ... and {len(skipped)-8} more")
        print()


def run_backtest(ticker: str, strategy: str, start_date: str, end_date: str, portfolio_value: float):
    """Run a backtest and print results to the terminal."""
    from src.backtester import BacktestEngine

    print(f"\n{'='*60}")
    print(f"  BACKTEST")
    print(f"  Ticker:    {ticker}")
    print(f"  Strategy:  {strategy}")
    print(f"  Period:    {start_date}  to  {end_date}")
    print(f"  Portfolio: ${portfolio_value:,.0f}")
    print(f"{'='*60}")
    print(f"  Fetching historical data and running simulation...")
    print(f"  (This may take 10-30 seconds)\n")

    engine = BacktestEngine(portfolio_value=portfolio_value)

    try:
        result = engine.run(ticker, strategy, start_date, end_date)
    except ValueError as e:
        print(f"  [ERROR] {e}")
        return

    if result.total_trades == 0:
        print("  No trades were generated in this period.")
        print("  Try a longer date range or a different ticker/strategy.")
        return

    # ── Summary header ──
    win_pct = result.win_rate * 100
    pnl_sign = '+' if result.total_pnl >= 0 else ''

    print(f"  RESULTS SUMMARY")
    print(f"  {'-'*54}")
    print(f"  Total trades:       {result.total_trades}")
    print(f"  Win rate:           {win_pct:.1f}%  "
          f"({result.winning_trades}W / {result.losing_trades}L / {result.breakeven_trades}BE)")
    print(f"  Total P&L:          {pnl_sign}${result.total_pnl:,.2f}")
    print(f"  Avg P&L per trade:  {'+' if result.avg_pnl_per_trade >= 0 else ''}${result.avg_pnl_per_trade:,.2f}")
    print(f"  Avg winner:         +${result.avg_winner:,.2f}")
    print(f"  Avg loser:          -${abs(result.avg_loser):,.2f}")
    print(f"  Avg credit:         ${result.avg_credit:.4f}/share")
    print(f"  Avg DTE held:       {result.avg_dte_held:.1f} days")
    print(f"  {'-'*54}")
    print(f"  Max drawdown:       -${result.max_drawdown:,.2f}")
    print(f"  Sharpe ratio:       {result.sharpe_ratio:.2f}")
    print(f"  Profit factor:      {result.profit_factor:.2f}")
    print(f"  {'-'*54}")

    # ── Exit reason breakdown ──
    print(f"  Exit reasons:")
    reason_labels = {
        'profit_target':    '50% profit target',
        'stop_loss':        '2x credit stop loss',
        'dte_exit':         '21 DTE forced exit',
        'expired':          'Expired',
        'end_of_backtest':  'End of backtest (open at cutoff)',
    }
    total = result.total_trades
    for reason, count in sorted(result.close_reason_breakdown.items(), key=lambda x: -x[1]):
        label = reason_labels.get(reason, reason)
        pct = count / total * 100 if total > 0 else 0
        print(f"    {label:<35} {count:3d}  ({pct:.0f}%)")

    # ── Trade log (last 10 trades) ──
    print(f"\n  RECENT TRADES (last 10 of {result.total_trades}):")
    print(f"  {'Entry':<12} {'Expiry':<12} {'DTE':>4} {'Credit':>8} "
          f"{'Close':>8} {'P&L':>8}  Reason")
    print(f"  {'-'*76}")

    recent = result.trades[-10:]
    for t in recent:
        pnl_str = f"+${t.pnl_dollars:,.0f}" if (t.pnl_dollars or 0) >= 0 else f"-${abs(t.pnl_dollars or 0):,.0f}"
        print(
            f"  {t.entry_date:<12} {t.expiry_date:<12} "
            f"{t.dte_at_entry:>4}  "
            f"${t.entry_credit:>6.4f}  "
            f"${t.close_debit or 0:>6.4f}  "
            f"{pnl_str:>8}  {t.close_reason}"
        )

    print(f"\n  Equity curve: {result.equity_curve[0][0]} -> {result.equity_curve[-1][0]}")
    print(f"  Final cumulative P&L: {pnl_sign}${result.total_pnl:,.2f}")
    print(f"\n  Tip: Run 'streamlit run app.py' for charts and full trade history.")
    print(f"{'='*60}\n")


def run_long_options(ticker: str, opt_type: str, target_dte: int):
    """Evaluate buying a call or put — print full long-option analysis."""
    from src.long_evaluator import LongEvaluator

    print(f"\n{'='*60}")
    print(f"  LONG OPTIONS EVALUATOR")
    print(f"  Ticker: {ticker}  |  Type: {opt_type.upper()}  |  Target DTE: {target_dte}")
    print(f"{'='*60}")
    print(f"  Fetching data...")

    ev = LongEvaluator()
    try:
        r = ev.evaluate(ticker, opt_type, target_dte)
    except Exception as e:
        print(f"\n  [ERROR] {e}")
        import traceback
        traceback.print_exc()
        return

    # ── Header ──────────────────────────────────────────────────
    iv_labels = {
        'cheap':    'CHEAP (good time to buy)',
        'fair':     'FAIR',
        'elevated': 'ELEVATED (buying is expensive)',
        'expensive':'EXPENSIVE (avoid buying)',
    }
    print(f"\n  {ticker} @ ${r.price:.2f}   Expiry: {r.expiry} ({r.actual_dte} DTE)")
    print(f"  ATM IV:   {r.current_iv*100:.1f}%   HV20: {r.hv20*100:.1f}%   "
          f"VRP: {r.current_iv/r.hv20:.2f}x")
    print(f"  IV Rank:  {r.iv_rank:.0f}  ({iv_labels.get(r.iv_assessment, r.iv_assessment)})")

    # ── Expected move ────────────────────────────────────────────
    print(f"\n  EXPECTED MOVE (1-SD = ~68% of outcomes)")
    print(f"  1-SD range: ${r.lower_1sd:.2f}  to  ${r.upper_1sd:.2f}  "
          f"(+/-${r.expected_move_1sd:.2f}, {r.expected_move_1sd/r.price*100:.1f}%)")
    print(f"  2-SD range: ${r.lower_1sd - r.expected_move_1sd:.2f}  to  "
          f"${r.upper_1sd + r.expected_move_1sd:.2f}  "
          f"(+/-${r.expected_move_2sd:.2f}, {r.expected_move_2sd/r.price*100:.1f}%)")

    # ── Strike candidates ────────────────────────────────────────
    print(f"\n  STRIKE CANDIDATES  ({opt_type.upper()})")
    print(f"  {'Label':<20} {'Strike':>8} {'Premium':>9} {'Break-even':>11} "
          f"{'Move Req':>9} {'Theta/day':>10} {'POP-to-BE':>10}")
    print(f"  {'-'*82}")
    for c in r.candidates:
        print(
            f"  {c.label:<20} "
            f"${c.strike:>7.2f} "
            f"${c.premium:>8.2f} "
            f"${c.break_even:>10.2f} "
            f"{c.move_needed_pct:>8.1f}% "
            f"${c.daily_theta:>9.3f} "
            f"{c.pop_to_be*100:>9.1f}%"
        )

    # ── Debit spreads ────────────────────────────────────────────
    if r.debit_spreads:
        print(f"\n  DEBIT SPREAD ALTERNATIVES")
        print(f"  {'Spread':<28} {'Debit':>7} {'Max Ret':>8} {'Break-even':>11} {'Theta/day':>10}")
        print(f"  {'-'*68}")
        for s in r.debit_spreads:
            print(
                f"  {s.label:<28} "
                f"${s.net_debit:>6.2f} "
                f"{s.max_return_pct:>7.0f}% "
                f"${s.break_even:>10.2f} "
                f"${s.theta_per_day:>9.3f}"
            )

    # ── Warnings ─────────────────────────────────────────────────
    if r.warnings:
        print(f"\n  WARNINGS:")
        for w in r.warnings:
            print(f"  [!] {w}")

    # ── Verdict ──────────────────────────────────────────────────
    print(f"\n  VERDICT")
    print(f"  {'-'*54}")
    if r.iv_assessment == 'cheap':
        verdict = "IV is cheap - reasonable time to buy options."
    elif r.iv_assessment == 'fair':
        verdict = "IV is fair - options are reasonably priced."
    elif r.iv_assessment == 'elevated':
        verdict = "IV is elevated - consider a debit spread to cut cost."
    else:
        verdict = "IV is expensive - debit spread strongly preferred over naked long."

    atm = next((c for c in r.candidates if c.label == 'ATM (50d)'), r.candidates[0])
    print(f"  {verdict}")
    print(f"  ATM break-even requires a {atm.move_needed_pct:.1f}% move "
          f"in {r.actual_dte} days.")
    print(f"  ATM theta: ${atm.daily_theta:.3f}/day  "
          f"({atm.theta_pct_daily:.2f}% of premium per day)")
    print(f"  ATM probability of profit at expiry: {atm.pop_to_be*100:.1f}%")

    print(f"\n  Tip: Run 'streamlit run app.py' -> Long Options for charts and full analysis.")
    print(f"{'='*60}\n")


def run_report(portfolio_value: float, watchlist: list, output_dir: str = None):
    """
    Scan the watchlist with the full decision engine, then generate a
    plain-English PDF report with trade cards, quality scores, and a
    'skipped' table explaining why other tickers were passed over.
    """
    import os
    from src.report_generator import generate_report

    if output_dir is None:
        # Default: save to Desktop so it is easy to find
        output_dir = os.path.join(os.path.expanduser('~'), 'Desktop')

    engine = DecisionEngine(portfolio_value=portfolio_value)
    vix = get_vix()

    print(f"\n{'='*60}")
    print(f"  OPTIONS TRADER -- REPORT GENERATOR")
    print(f"  VIX: {vix:.1f}  |  Portfolio: ${portfolio_value:,.0f}")
    print(f"  Scanning {len(watchlist)} tickers...")
    print(f"{'='*60}\n")

    recommendations = []
    prices = {}

    for ticker in watchlist:
        print(f"  Analyzing {ticker}...", end='\r')
        try:
            rec = engine.analyze(ticker)
            recommendations.append(rec)
            # Store price for the report (set by engine if data fetched OK)
            if rec.current_price:
                prices[ticker] = rec.current_price
        except Exception as e:
            print(f"\n  [skip] {ticker}: {e}")

    n_open = len([r for r in recommendations if r.action == 'open'])
    print(f"  Scan complete. {n_open} trade opportunity(s) found.   ")
    print(f"  Generating PDF...\n")

    try:
        path = generate_report(
            recommendations=recommendations,
            vix=vix,
            portfolio_value=portfolio_value,
            prices=prices,
            output_dir=output_dir,
        )
        print(f"  Report saved to:")
        print(f"  {path}")
        print(f"\n  Open the PDF to read plain-English trade recommendations.")
        # Try to auto-open on Windows
        try:
            os.startfile(path)
        except Exception:
            pass
    except Exception as e:
        print(f"\n  [ERROR] Could not generate PDF: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*60}\n")


def run_undervalued(portfolio_value: float, watchlist: list, min_score: float = 0):
    """
    Screen the watchlist for undervalued stocks using fundamentals + technicals.
    Prints a ranked table and plain-English breakdown of top picks.
    """
    from src.stock_screener import StockScreener

    screener = StockScreener()

    print(f"\n{'='*60}")
    print(f"  VALUE SCREENER")
    print(f"  Screening {len(watchlist)} tickers for undervaluation...")
    print(f"  (Combining analyst targets, P/E, earnings growth, RSI, 52-week range)")
    print(f"{'='*60}\n")

    results = []
    for ticker in watchlist:
        print(f"  Screening {ticker}...", end='\r')
        ev = screener.screen(ticker)
        results.append(ev)

    print(f"  Done.                                    \n")

    # Filter by min score
    shown = [r for r in results if not r.error and r.total_score >= min_score]
    errors = [r for r in results if r.error]

    if not shown and not errors:
        print("  No results above the minimum score threshold.")
        return

    # ── Results table ──
    print(f"  {'Ticker':<8} {'Score':>6} {'Label':<22} {'Price':>8} "
          f"{'Target':>8} {'Disc%':>6} {'FwdPE':>6} {'RSI':>5} {'IVRank':>7} {'Action'}")
    print(f"  {'-'*100}")

    action_labels = {
        'buy_call':     'BUY CALL',
        'debit_spread': 'DEBIT SPREAD',
        'wait':         'Wait',
    }

    for r in shown:
        target_str = f"${r.analyst_target:.0f}" if r.analyst_target else '  n/a'
        disc_str   = f"{r.discount_to_target_pct:.0f}%" if r.analyst_target else '  n/a'
        pe_str     = f"{r.forward_pe:.1f}" if r.forward_pe else '  n/a'
        iv_str     = f"{r.iv_rank:.0f}" if r.iv_rank else '  n/a'
        action     = action_labels.get(r.options_action, r.options_action)
        print(
            f"  {r.ticker:<8} {r.total_score:>6.0f} {r.score_label:<22} "
            f"${r.price:>7.2f} {target_str:>8} {disc_str:>6} {pe_str:>6} "
            f"{r.rsi_14:>5.1f} {iv_str:>7}  {action}"
        )

    if errors:
        print(f"\n  Could not screen: {', '.join(r.ticker for r in errors)}")

    # ── Plain-English breakdown of top picks ──
    top = [r for r in shown if r.total_score >= 55][:3]  # top 3 with Watch List or better
    if top:
        print(f"\n{'='*60}")
        print(f"  DETAILED BREAKDOWN (Watch List or better)\n")
        for r in top:
            print(f"  -- {r.ticker} {'--'*20}")
            for line in r.plain_english.split('\n'):
                print(f"  {line}")
            print()

    print(f"  Tip: Run 'streamlit run app.py' -> Value Screener for charts and full analysis.")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Options Trader — Systematic premium-selling system'
    )
    parser.add_argument('--ticker', type=str, help='Analyze a specific ticker')
    parser.add_argument('--strategy', type=str, default='auto',
                        help='Strategy name: "auto", "Short Strangle", "Iron Condor", "Short Put"')
    parser.add_argument('--portfolio-value', type=float, default=25000,
                        help='Portfolio value in USD (default: 25000)')
    parser.add_argument('--portfolio', action='store_true', help='Show portfolio summary')
    parser.add_argument('--kb', action='store_true', help='Show knowledge base summary')
    parser.add_argument('--watchlist', nargs='+', help='Custom watchlist of tickers')
    parser.add_argument('--backtest', action='store_true',
                        help='Run a historical backtest (requires --ticker)')
    parser.add_argument('--start', type=str, default='2022-01-01',
                        help='Backtest start date YYYY-MM-DD (default: 2022-01-01)')
    parser.add_argument('--end', type=str, default='2024-12-31',
                        help='Backtest end date YYYY-MM-DD (default: 2024-12-31)')

    # Paper trading
    parser.add_argument('--paper-trade', action='store_true',
                        help='Run a paper trading cycle (scan + auto-enter + auto-exit)')
    parser.add_argument('--exits-only', action='store_true',
                        help='Paper trade: check and process exits only (no new entries)')
    parser.add_argument('--entries-only', action='store_true',
                        help='Paper trade: scan and enter new positions only (no exit check)')
    parser.add_argument('--positions', action='store_true',
                        help='Paper trade: show live mark-to-market for all open paper positions')
    parser.add_argument('--paper-portfolio', action='store_true',
                        help='Show paper portfolio summary (positions, P&L, win rate)')

    # Long options evaluator
    parser.add_argument('--long-options', type=str, metavar='TICKER',
                        help='Evaluate buying a call or put on TICKER')
    parser.add_argument('--type', type=str, default='call', choices=['call', 'put'],
                        help='Option type: call or put (default: call)')
    parser.add_argument('--dte', type=int, default=60,
                        help='Target days-to-expiry (default: 60)')

    # PDF report generator
    parser.add_argument('--report', action='store_true',
                        help='Scan watchlist and generate a plain-English PDF report')
    parser.add_argument('--output', type=str, default=None,
                        help='Folder to save the PDF report (default: Desktop)')

    # Value screener
    parser.add_argument('--undervalued', action='store_true',
                        help='Screen watchlist for potentially undervalued stocks')
    parser.add_argument('--min-score', type=float, default=0,
                        help='Minimum value score to display (0-100, default: 0 = show all)')

    args = parser.parse_args()

    if args.kb:
        show_knowledge_base()
        return

    if args.portfolio:
        show_portfolio(args.portfolio_value)
        return

    if args.backtest:
        if not args.ticker:
            print("\n[ERROR] --backtest requires --ticker. Example:")
            print("  python main.py --backtest --ticker SPY")
            print("  python main.py --backtest --ticker SPY --strategy 'Iron Condor' --start 2021-01-01 --end 2023-12-31\n")
            return
        strategy = args.strategy if args.strategy != 'auto' else 'Short Strangle'
        run_backtest(
            ticker=args.ticker.upper(),
            strategy=strategy,
            start_date=args.start,
            end_date=args.end,
            portfolio_value=args.portfolio_value,
        )
        return

    if args.paper_portfolio:
        paper_portfolio = Portfolio(value=args.portfolio_value, paper=True)
        paper_portfolio.display_summary()
        return

    if args.paper_trade or args.exits_only or args.entries_only or args.positions:
        watchlist = args.watchlist if args.watchlist else DEFAULT_WATCHLIST
        run_paper_trade(
            portfolio_value=args.portfolio_value,
            watchlist=[t.upper() for t in watchlist],
            exits_only=args.exits_only,
            entries_only=args.entries_only,
            show_positions=args.positions,
        )
        return

    if args.long_options:
        run_long_options(
            ticker=args.long_options.upper(),
            opt_type=args.type,
            target_dte=args.dte,
        )
        return

    if args.report:
        watchlist = args.watchlist if args.watchlist else DEFAULT_WATCHLIST
        run_report(
            portfolio_value=args.portfolio_value,
            watchlist=[t.upper() for t in watchlist],
            output_dir=args.output,
        )
        return

    if args.undervalued:
        watchlist = args.watchlist if args.watchlist else DEFAULT_WATCHLIST
        run_undervalued(
            portfolio_value=args.portfolio_value,
            watchlist=[t.upper() for t in watchlist],
            min_score=args.min_score,
        )
        return

    if args.ticker:
        analyze_single(args.ticker.upper(), args.strategy, args.portfolio_value)
        return

    # Default: scan watchlist
    watchlist = args.watchlist if args.watchlist else DEFAULT_WATCHLIST
    scan_watchlist(args.portfolio_value, [t.upper() for t in watchlist])


if __name__ == '__main__':
    main()
