"""
Options Trader — Streamlit Dashboard
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page config — must be first Streamlit call
st.set_page_config(
    page_title="Options Trader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports from our engine ──────────────────────────────────────────────────
from src.knowledge_base import KnowledgeBase
from src.decision_engine import DecisionEngine
from src.portfolio import Portfolio, Position
from src.market_data import get_vix, get_underlying_data, get_historical_volatility, get_iv_rank
from src.greeks import calculate_greeks, dte_to_years
from src.backtester import BacktestEngine

# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource
def load_kb():
    return KnowledgeBase()

@st.cache_data(ttl=300)  # cache 5 minutes
def fetch_vix():
    return get_vix()

DEFAULT_WATCHLIST = [
    'SPY', 'QQQ', 'IWM', 'GLD', 'TLT',
    'AAPL', 'MSFT', 'TSLA', 'AMZN', 'NVDA',
    'AMD', 'META', 'GOOGL',
]

STRATEGY_OPTIONS = ['auto', 'Short Strangle', 'Iron Condor', 'Short Put', 'Jade Lizard', 'Covered Call']

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Options Trader")
    st.caption("Systematic premium-selling system")
    st.divider()

    page = st.radio(
        "Navigation",
        ["Analyze Ticker", "Watchlist Scan", "Portfolio", "Paper Trading", "Backtester", "Long Options", "Value Screener", "Knowledge Base", "Greeks Calculator"],
        index=0,
    )

    st.divider()
    portfolio_value = st.number_input(
        "Portfolio Value ($)",
        min_value=5000,
        max_value=10_000_000,
        value=25000,
        step=5000,
    )

    try:
        vix = fetch_vix()
        vix_color = "red" if vix > 25 else "orange" if vix > 18 else "green"
        st.metric("VIX", f"{vix:.1f}")
    except Exception:
        vix = 20.0
        st.metric("VIX", "N/A")

    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

    # ── Upcoming macro events ──
    try:
        from src.macro_calendar import get_upcoming_events, check_macro_blackout
        blocked, imminent = check_macro_blackout(days_before=2)
        upcoming = get_upcoming_events(days_ahead=14)
        if blocked:
            st.error("HR-010 ACTIVE — Macro blackout")
            for e in imminent:
                st.caption(f"{e['type']}: {e['date_str']} ({e['days_away']}d)")
        elif upcoming:
            st.warning(f"Next: {upcoming[0]['type']} in {upcoming[0]['days_away']}d")
            for e in upcoming[:3]:
                st.caption(f"{e['type']}: {e['date_str']} ({e['days_away']}d)")
        else:
            st.success("No events in 14 days")
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYZE TICKER
# ══════════════════════════════════════════════════════════════════════════════
if page == "Analyze Ticker":
    st.header("Analyze a Ticker")
    st.caption("Runs the full decision engine against a single underlying. Checks all entry rules, constructs the trade, and gives a recommendation.")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        ticker_input = st.text_input("Ticker Symbol", value="SPY", max_chars=10).upper()
    with col2:
        strategy_input = st.selectbox("Strategy", STRATEGY_OPTIONS)
    with col3:
        st.write("")
        st.write("")
        analyze_btn = st.button("Analyze", type="primary", use_container_width=True)

    if analyze_btn:
        with st.spinner(f"Analyzing {ticker_input}..."):
            try:
                engine = DecisionEngine(portfolio_value=portfolio_value)
                rec = engine.analyze(ticker_input, strategy_name=strategy_input)

                # ── Action banner ──
                if rec.action == 'open':
                    st.success(f"OPEN TRADE — {rec.strategy} on {ticker_input}")
                else:
                    st.warning(f"PASS — {rec.strategy} on {ticker_input}")

                # ── Top metrics row ──
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Strategy", rec.strategy)
                m2.metric("IV Rank", f"{rec.iv_rank:.0f}",
                          help="Where current IV sits in its 52-week range (0=low, 100=high)")
                m3.metric("IV Pct", f"{rec.iv_percentile:.0f}",
                          help="% of trading days in the past year where IV was lower than today")
                m4.metric("VIX", f"{rec.vix:.1f}")
                m5.metric("Regime", rec.regime.replace('_', ' ').title().replace('Iv', 'IV'))

                # Earnings metric — color-code by proximity
                if rec.days_until_earnings is not None:
                    days_e = rec.days_until_earnings
                    earnings_label = f"{rec.next_earnings_date}"
                    earnings_delta = f"{days_e}d away"
                    if days_e <= 7:
                        m6.metric("Earnings", earnings_label, delta=f"BLACKOUT {earnings_delta}", delta_color="inverse")
                    elif days_e <= 45:
                        m6.metric("Earnings", earnings_label, delta=f"In window: {earnings_delta}", delta_color="inverse")
                    else:
                        m6.metric("Earnings", earnings_label, delta=f"Clear: {earnings_delta}", delta_color="normal")
                else:
                    m6.metric("Earnings", "Unknown")

                # ── Macro events row ──
                if rec.upcoming_macro_events:
                    st.caption("Upcoming Macro Events (45-day window)")
                    macro_cols = st.columns(min(len(rec.upcoming_macro_events), 4))
                    for i, e in enumerate(rec.upcoming_macro_events[:4]):
                        days = e['days_away']
                        label = e['type']
                        val   = e['date']
                        if days <= 2:
                            delta_str   = f"BLACKOUT ({days}d)"
                            delta_color = "inverse"
                        elif days <= 7:
                            delta_str   = f"SOON: {days}d away"
                            delta_color = "inverse"
                        else:
                            delta_str   = f"{days}d away"
                            delta_color = "normal"
                        macro_cols[i].metric(label, val, delta=delta_str, delta_color=delta_color,
                                             help=e.get('description', ''))

                # ── Volatility breakdown row ──
                st.caption("Volatility Analysis")
                v1, v2, v3, v4 = st.columns(4)
                atm_iv_str = f"{rec.current_iv*100:.1f}%" if rec.current_iv else "n/a"
                hv20_str   = f"{rec.current_hv*100:.1f}%" if rec.current_hv else "n/a"
                vrp_str    = f"{rec.vrp_ratio:.2f}x" if rec.vrp_ratio else "n/a"
                has_edge   = rec.vrp_ratio and rec.vrp_ratio >= 1.0
                vrp_delta  = "IV > HV (edge)" if has_edge else ("IV <= HV (thin)" if rec.vrp_ratio else None)
                v1.metric("ATM IV", atm_iv_str,
                          help="Implied volatility of near-ATM options — the premium we are selling")
                v2.metric("HV20", hv20_str,
                          help="20-day realized (historical) volatility — what the market has actually done")
                v3.metric("VRP Ratio", vrp_str, delta=vrp_delta,
                          delta_color="normal" if has_edge else "inverse",
                          help="IV / HV — the volatility risk premium. >1.0 means IV > realized vol (the edge we capture by selling premium)")
                v4.metric("POP", f"{rec.probability_of_profit*100:.0f}%" if rec.action == 'open' else "—",
                          help="Probability of profit at expiration (derived from short strike deltas)")

                # ── Trade construction ──
                if rec.action == 'open':
                    st.subheader("Trade Construction")
                    c1, c2 = st.columns(2)
                    with c1:
                        legs_data = []
                        for leg in rec.legs:
                            legs_data.append({
                                'Action': leg['action'],
                                'Type': leg['type'],
                                'Strike': f"${leg['strike']:.1f}",
                                'Mid Price': f"${leg['mid']:.2f}",
                                'Delta': leg.get('delta', '—'),
                            })
                        st.dataframe(
                            pd.DataFrame(legs_data),
                            hide_index=True,
                            use_container_width=True,
                        )

                    with c2:
                        st.metric("Net Credit", f"${rec.entry_credit:.2f}/contract")
                        st.metric("Max Profit", f"${rec.max_profit:.2f}")
                        st.metric("Max Loss", "Undefined (2x stop)" if rec.max_loss is None else f"${rec.max_loss:.2f}")
                        st.metric("Contracts", rec.suggested_contracts)

                    st.info(
                        f"**Target Exit:** Close at 50% profit (${rec.max_profit*0.5:.2f}) or 21 DTE — whichever comes first.  \n"
                        f"**Stop Loss:** Close if loss reaches 2x credit = "
                        f"${rec.entry_credit * 2 * 100 * rec.suggested_contracts:.2f}"
                    )

                    # ── Add to portfolio button ──
                    if st.button("Add to Portfolio", type="secondary"):
                        portfolio = Portfolio(value=portfolio_value)
                        if not portfolio.check_position_limit(ticker_input):
                            st.error(f"HR-006: Already have an open position in {ticker_input}.")
                        else:
                            from src.market_data import get_options_chain
                            chain = get_options_chain(ticker_input, target_dte=45)
                            exp = chain['expiration'].iloc[0] if chain is not None and not chain.empty else "unknown"
                            dte = int(chain['dte'].iloc[0]) if chain is not None and not chain.empty else 45
                            pos = Position(
                                ticker=ticker_input,
                                strategy=rec.strategy,
                                entry_date=datetime.now().strftime('%Y-%m-%d'),
                                expiration=exp,
                                dte_at_entry=dte,
                                legs=rec.legs,
                                entry_credit=rec.entry_credit,
                                contracts=rec.suggested_contracts,
                            )
                            portfolio.add_position(pos)
                            st.success(f"Position added: {ticker_input} {rec.strategy}")

                # ── Trade Score + Plain-English Summary ──
                if rec.trade_score or rec.plain_english:
                    st.subheader("Plain-English Summary")
                    score_cols = st.columns([1, 1, 4])
                    if rec.trade_score:
                        _label = rec.trade_score_label or ''
                        _delta_color = (
                            'normal'  if _label == 'Strong' else
                            'normal'  if _label == 'Good'   else
                            'inverse' if _label in ('Marginal', 'Avoid') else 'off'
                        )
                        score_cols[0].metric(
                            "Trade Score",
                            f"{rec.trade_score:.0f} / 100",
                            delta=_label,
                            delta_color=_delta_color,
                        )
                    if rec.plain_english:
                        with st.container():
                            st.markdown(rec.plain_english.replace('\n', '  \n'))

                # ── Rationale ──
                st.subheader("Decision Rationale")
                for r in rec.rationale:
                    icon = "✅" if r.startswith("PASS") else "❌" if r.startswith("FAIL") else "ℹ️"
                    st.write(f"{icon} {r}")

                if rec.warnings:
                    st.subheader("Warnings")
                    for w in rec.warnings:
                        st.warning(w)

            except Exception as e:
                st.error(f"Error analyzing {ticker_input}: {e}")
                import traceback
                with st.expander("Error details"):
                    st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: WATCHLIST SCAN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Watchlist Scan":
    st.header("Watchlist Scanner")
    st.caption("Scan multiple tickers against the full rule set. Surfaces trade opportunities.")

    custom_tickers = st.text_input(
        "Tickers (comma-separated)",
        value=", ".join(DEFAULT_WATCHLIST),
    )

    scan_btn = st.button("Run Scan", type="primary")

    if scan_btn:
        tickers = [t.strip().upper() for t in custom_tickers.split(',') if t.strip()]
        engine = DecisionEngine(portfolio_value=portfolio_value)

        progress = st.progress(0, text="Starting scan...")
        results = []

        for i, ticker in enumerate(tickers):
            progress.progress((i + 1) / len(tickers), text=f"Analyzing {ticker}...")
            try:
                rec = engine.analyze(ticker)
                # Earnings status for scan table
                if rec.days_until_earnings is not None:
                    days_e = rec.days_until_earnings
                    earnings_str = f"{rec.next_earnings_date} ({days_e}d)"
                    if days_e <= 45:
                        earnings_str = f"[BLACKOUT] {earnings_str}" if days_e <= 7 else f"[IN WINDOW] {earnings_str}"
                else:
                    earnings_str = "Unknown"

                results.append({
                    'Ticker': ticker,
                    'Action': rec.action.upper(),
                    'Strategy': rec.strategy,
                    'IV Rank': rec.iv_rank,
                    'IV Pct': f"{rec.iv_percentile:.0f}",
                    'ATM IV': f"{rec.current_iv*100:.1f}%" if rec.current_iv else '—',
                    'HV20': f"{rec.current_hv*100:.1f}%" if rec.current_hv else '—',
                    'VRP': f"{rec.vrp_ratio:.2f}x" if rec.vrp_ratio else '—',
                    'VIX': rec.vix,
                    'Regime': rec.regime.replace('_', ' ').replace('environment', '').strip().title(),
                    'Earnings': earnings_str,
                    'Credit': f"${rec.entry_credit:.2f}" if rec.action == 'open' else '—',
                    'POP': f"{rec.probability_of_profit*100:.0f}%" if rec.action == 'open' else '—',
                    'Contracts': rec.suggested_contracts if rec.action == 'open' else '—',
                    'Reason': rec.rationale[0] if rec.rationale else '',
                    '_rec': rec,
                })
            except Exception as e:
                results.append({
                    'Ticker': ticker, 'Action': 'ERROR', 'Strategy': '—',
                    'IV Rank': 0, 'VIX': 0, 'Regime': '—', 'Earnings': '—',
                    'Credit': '—', 'POP': '—', 'Contracts': '—',
                    'Reason': str(e), '_rec': None,
                })

        progress.empty()

        opens = [r for r in results if r['Action'] == 'OPEN']
        passes = [r for r in results if r['Action'] != 'OPEN']

        # ── Summary metrics ──
        c1, c2, c3 = st.columns(3)
        c1.metric("Tickers Scanned", len(results))
        c2.metric("Opportunities Found", len(opens))
        c3.metric("Passed", len(passes))

        # ── Opportunities ──
        if opens:
            st.subheader(f"Trade Opportunities ({len(opens)})")
            df_open = pd.DataFrame([{k: v for k, v in r.items() if k != '_rec'} for r in opens])
            st.dataframe(df_open, hide_index=True, use_container_width=True)

            # IV Rank bar chart
            fig = px.bar(
                df_open, x='Ticker', y='IV Rank',
                color='Strategy', title='IV Rank by Ticker (Opportunities)',
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.add_hline(y=30, line_dash="dash", annotation_text="Strangle threshold (30)")
            fig.add_hline(y=20, line_dash="dot", annotation_text="Condor/Put threshold (20)", line_color="orange")
            st.plotly_chart(fig, use_container_width=True)

        # ── Full results table ──
        st.subheader("Full Scan Results")
        df_all = pd.DataFrame([{k: v for k, v in r.items() if k != '_rec'} for r in results])

        def color_action(val):
            if val == 'OPEN':
                return 'background-color: #1a5c2a; color: white'
            elif val == 'ERROR':
                return 'background-color: #5c1a1a; color: white'
            return ''

        st.dataframe(
            df_all.style.map(color_action, subset=['Action']),
            hide_index=True,
            use_container_width=True,
        )

        # ── Trade Score cards (opportunities only) ──
        if opens:
            st.subheader("Plain-English Trade Summaries")
            st.caption("What each trade is, how much you earn, and when to exit — explained simply.")
            for r in opens:
                rec_obj = r.get('_rec')
                if rec_obj is None:
                    continue
                score = getattr(rec_obj, 'trade_score', 0)
                label = getattr(rec_obj, 'trade_score_label', '')
                pe    = getattr(rec_obj, 'plain_english', '')
                score_color = {'Strong': 'green', 'Good': 'blue', 'Marginal': 'orange', 'Avoid': 'red'}.get(label, 'gray')
                with st.expander(
                    f"{r['Ticker']}  —  {r['Strategy']}  |  Score: {score:.0f}/100 ({label})",
                    expanded=(score >= 60),
                ):
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Trade Score", f"{score:.0f} / 100", delta=label)
                    col_b.metric("Win Probability", r.get('POP', '—'))
                    col_c.metric("Income / Contract", r.get('Credit', '—'))
                    if pe:
                        st.markdown(pe.replace('\n', '  \n'))

        # ── PDF Report Download ──
        st.divider()
        st.subheader("Download Report")
        st.caption(
            "Generate a plain-English PDF summarising today's scan: "
            "what to trade, how much you earn, and what was skipped."
        )
        if st.button("Generate PDF Report", type="secondary"):
            with st.spinner("Building PDF..."):
                try:
                    import tempfile
                    from src.report_generator import generate_report
                    recs_for_pdf = [r['_rec'] for r in results if r['_rec'] is not None]
                    prices_for_pdf = {
                        r['_rec'].ticker: r['_rec'].current_price
                        for r in results
                        if r['_rec'] is not None and r['_rec'].current_price
                    }
                    with tempfile.TemporaryDirectory() as tmpdir:
                        pdf_path = generate_report(
                            recommendations=recs_for_pdf,
                            vix=vix,
                            portfolio_value=portfolio_value,
                            prices=prices_for_pdf,
                            output_dir=tmpdir,
                        )
                        with open(pdf_path, 'rb') as f:
                            pdf_bytes = f.read()

                    from datetime import datetime as _dt
                    fname = f"options_report_{_dt.now().strftime('%Y%m%d_%H%M')}.pdf"
                    st.download_button(
                        label="Save PDF Report",
                        data=pdf_bytes,
                        file_name=fname,
                        mime="application/pdf",
                        type="primary",
                    )
                    st.success(f"PDF ready — {len(pdf_bytes)//1024} KB. Click 'Save PDF Report' above.")
                except ImportError:
                    st.error("reportlab not installed. Run: pip install reportlab")
                except Exception as e:
                    st.error(f"Could not generate PDF: {e}")
                    import traceback
                    with st.expander("Error details"):
                        st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PORTFOLIO
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Portfolio":
    st.header("Portfolio")
    portfolio = Portfolio(value=portfolio_value)
    summary = portfolio.summary()

    # ── Summary metrics ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portfolio Value", f"${summary['portfolio_value']:,.0f}")
    c2.metric("Open Positions", summary['open_positions'])
    c3.metric("Total P&L", f"${summary['total_pnl']:+,.2f}")
    c4.metric("Win Rate", summary['win_rate'])

    # ── Open positions ──
    st.subheader("Open Positions")
    open_pos = portfolio.open_positions()
    if open_pos:
        open_data = []
        for pos in open_pos:
            open_data.append({
                'Ticker': pos.ticker,
                'Strategy': pos.strategy,
                'Entry Date': pos.entry_date,
                'Expiration': pos.expiration,
                'DTE at Entry': pos.dte_at_entry,
                'Credit': f"${pos.entry_credit:.2f}",
                'Contracts': pos.contracts,
                'Max Profit': f"${pos.entry_credit * 100 * pos.contracts:.2f}",
                '50% Target': f"${pos.entry_credit * 50 * pos.contracts:.2f}",
            })
        st.dataframe(pd.DataFrame(open_data), hide_index=True, use_container_width=True)

        st.subheader("Close a Position")
        close_ticker = st.selectbox("Select position to close", [p.ticker for p in open_pos])
        close_debit = st.number_input("Closing debit (price paid to close)", min_value=0.0, step=0.01)
        if st.button("Close Position", type="primary"):
            portfolio.close_position(close_ticker, close_debit)
            st.success(f"Position closed. Refresh to see updated P&L.")
            st.rerun()
    else:
        st.info("No open positions. Use 'Analyze Ticker' to find and open trades.")

    # ── Closed positions ──
    st.subheader("Trade History")
    closed_pos = portfolio.closed_positions()
    if closed_pos:
        closed_data = []
        for pos in closed_pos:
            closed_data.append({
                'Ticker': pos.ticker,
                'Strategy': pos.strategy,
                'Entry': pos.entry_date,
                'Close': pos.close_date,
                'Credit': f"${pos.entry_credit:.2f}",
                'Debit': f"${pos.close_debit:.2f}" if pos.close_debit else '—',
                'P&L': f"${pos.pnl:+.2f}" if pos.pnl is not None else '—',
                'Contracts': pos.contracts,
            })

        df_closed = pd.DataFrame(closed_data)
        st.dataframe(df_closed, hide_index=True, use_container_width=True)

        # ── P&L chart ──
        pnl_vals = [p.pnl for p in closed_pos if p.pnl is not None]
        if pnl_vals:
            cumulative = [sum(pnl_vals[:i+1]) for i in range(len(pnl_vals))]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=cumulative,
                mode='lines+markers',
                name='Cumulative P&L',
                line=dict(color='#00cc88', width=2),
                fill='tozeroy',
                fillcolor='rgba(0,204,136,0.1)',
            ))
            fig.update_layout(
                title='Cumulative P&L',
                yaxis_title='P&L ($)',
                xaxis_title='Trade #',
                template='plotly_dark',
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No closed trades yet.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Knowledge Base":
    st.header("Knowledge Base")
    st.caption("All trading rules, strategies, and source material encoded in the system.")

    kb = load_kb()

    tab1, tab2, tab3, tab4 = st.tabs(["Strategies", "Rules", "Market Regimes", "Sources"])

    with tab1:
        st.subheader("Strategies")
        for name, strategy in kb.strategies.items():
            with st.expander(f"{name} — {strategy.get('type','').replace('_',' ').title()}"):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Description:** {strategy.get('description','')}")
                    entry = strategy.get('entry_conditions', {})
                    st.markdown("**Entry Conditions:**")
                    st.json({
                        'DTE Target': entry.get('dte_target'),
                        'DTE Range': entry.get('dte_range'),
                        'IV Rank Min': entry.get('iv_rank_minimum'),
                        'IV Rank Preferred': entry.get('iv_rank_preferred'),
                    })
                with c2:
                    exit_cond = strategy.get('exit_conditions', {})
                    st.markdown("**Exit Conditions:**")
                    st.json({
                        'Profit Target': f"{exit_cond.get('profit_target_pct',0)*100:.0f}%",
                        'Stop Loss': f"{exit_cond.get('stop_loss_multiple','2')}x credit",
                        'Forced Exit DTE': exit_cond.get('forced_exit_dte'),
                    })

                if strategy.get('key_insights'):
                    st.markdown("**Key Insights:**")
                    for insight in strategy['key_insights']:
                        st.markdown(f"- {insight}")

    with tab2:
        st.subheader("Hard Risk Rules (Never Violated)")
        for rule in kb.hard_risk_rules:
            with st.expander(f"[{rule['id']}] {rule['name'].replace('_',' ').title()}"):
                st.markdown(f"**Rule:** `{rule.get('rule','')}`")
                st.markdown(f"**Source:** {rule.get('source','')}")
                if rule.get('rationale'):
                    st.info(rule['rationale'])

        st.subheader("Entry Rules")
        for rule in kb.entry_rules:
            with st.expander(f"[{rule['id']}] {rule['name'].replace('_',' ').title()}"):
                st.markdown(f"**Source:** {rule.get('source','')}")
                if rule.get('rationale'):
                    st.info(rule['rationale'])

        st.subheader("Exit Rules")
        for rule in kb.exit_rules:
            with st.expander(f"[{rule['id']}] {rule['name'].replace('_',' ').title()}"):
                st.markdown(f"**Source:** {rule.get('source','')}")
                if rule.get('rationale'):
                    st.info(rule['rationale'])

    with tab3:
        st.subheader("Market Regime Playbooks")
        for regime_name, regime in kb.market_regimes.items():
            with st.expander(regime_name.replace('_', ' ').title()):
                st.markdown(f"**Description:** {regime.get('description','').strip()}")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Preferred Strategies:**")
                    for s in regime.get('preferred_strategies', []):
                        st.markdown(f"- **{s['name']}** — {s['reason']}")
                with col2:
                    st.markdown("**Avoid:**")
                    for s in regime.get('avoid_strategies', []):
                        st.markdown(f"- **{s['name']}** — {s['reason']}")

                if regime.get('key_insights'):
                    st.markdown("**Key Insights:**")
                    for insight in regime['key_insights']:
                        st.markdown(f"- {insight}")

    with tab4:
        st.subheader("Source Material")
        for src_name, content in kb.sources.items():
            with st.expander(src_name.replace('_', ' ').title()):
                st.markdown(content)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: GREEKS CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Greeks Calculator":
    st.header("Greeks Calculator")
    st.caption("Black-Scholes Greeks for any option. Useful for evaluating specific strikes.")

    c1, c2, c3 = st.columns(3)
    with c1:
        opt_type = st.selectbox("Option Type", ["call", "put"])
        underlying_price = st.number_input("Underlying Price ($)", value=500.0, step=1.0)
        strike = st.number_input("Strike Price ($)", value=510.0, step=1.0)
    with c2:
        dte_val = st.number_input("Days to Expiration", value=45, step=1, min_value=1)
        iv_input = st.number_input("Implied Volatility (%)", value=25.0, step=0.5, min_value=0.1)
        rfr = st.number_input("Risk-Free Rate (%)", value=5.0, step=0.25)
    with c3:
        market_price = st.number_input("Market Price (optional, for IV calc)", value=0.0, step=0.01)

    if st.button("Calculate Greeks", type="primary"):
        T = dte_to_years(dte_val)
        sigma = iv_input / 100
        r = rfr / 100
        mp = market_price if market_price > 0 else None

        g = calculate_greeks(opt_type, underlying_price, strike, T, r, sigma, market_price=mp)

        st.subheader("Results")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Delta", f"{g.delta:.4f}", help="Rate of price change vs underlying")
        col2.metric("Gamma", f"{g.gamma:.6f}", help="Rate of delta change vs underlying")
        col3.metric("Theta (daily)", f"${g.theta:.4f}", help="Daily time decay (negative = you lose this per day as LONG)")
        col4.metric("Vega (per 1% IV)", f"${g.vega:.4f}", help="P&L change per 1% IV move")

        col5, col6, col7 = st.columns(3)
        col5.metric("Theoretical Value", f"${g.theoretical_value:.4f}")
        col6.metric("Intrinsic Value", f"${g.intrinsic_value:.4f}")
        col7.metric("Time Value", f"${g.time_value:.4f}")

        # ── Delta profile chart ──
        st.subheader("Delta Profile (across strikes)")
        import numpy as np
        strikes_range = [underlying_price * x for x in np.arange(0.70, 1.30, 0.02)]
        deltas = [
            calculate_greeks(opt_type, underlying_price, k, T, r, sigma).delta
            for k in strikes_range
        ]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=strikes_range, y=deltas,
            mode='lines', name='Delta',
            line=dict(color='#00cc88', width=2),
        ))
        fig.add_vline(x=underlying_price, line_dash="dash", annotation_text="Current Price")
        fig.add_vline(x=strike, line_dash="dot", line_color="orange", annotation_text=f"Strike ${strike:.0f}")
        fig.update_layout(
            title=f'Delta Profile — {opt_type.upper()}',
            xaxis_title='Strike Price',
            yaxis_title='Delta',
            template='plotly_dark',
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Theta decay chart ──
        st.subheader("Theta Decay (time to expiration)")
        dtes = list(range(dte_val, 0, -1))
        thetas = [
            calculate_greeks(opt_type, underlying_price, strike, dte_to_years(d), r, sigma).theta
            for d in dtes
        ]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=dtes[::-1], y=thetas[::-1],
            mode='lines', name='Theta/day',
            line=dict(color='#ff6666', width=2),
            fill='tozeroy',
            fillcolor='rgba(255,102,102,0.1)',
        ))
        fig2.add_vline(x=21, line_dash="dash", annotation_text="21 DTE exit rule")
        fig2.update_layout(
            title='Daily Theta Decay',
            xaxis_title='Days to Expiration',
            yaxis_title='Theta ($/day)',
            xaxis_autorange='reversed',
            template='plotly_dark',
        )
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: BACKTESTER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Backtester":
    st.header("Strategy Backtester")
    st.caption(
        "Simulates how a strategy would have performed on historical price data. "
        "Uses Black-Scholes with rolling 20-day HV as IV proxy. "
        "Fills at theoretical mid — no slippage modeled."
    )

    with st.expander("Methodology & Limitations", expanded=False):
        st.markdown("""
        **What this does:**
        - Fetches 1-5 years of historical daily close prices from Yahoo Finance (free)
        - Computes 20-day rolling historical volatility (HV) as a proxy for implied volatility
        - For each potential entry day, analytically finds the 16-delta strikes using Black-Scholes
        - Simulates the position day-by-day, re-pricing the options with current price and HV
        - Applies the three exit rules: 50% profit target, 2x credit stop loss, 21 DTE forced exit

        **Known limitations:**
        - HV understates real IV (real IV includes a volatility risk premium). Credits in the backtest are slightly understated vs live trading — this means the backtest is **conservative**
        - Fills at Black-Scholes theoretical mid — no bid/ask spread or slippage
        - No IV skew modeled (real puts are more expensive than calls at same delta)
        - No earnings avoidance (historical earnings calendar not available for free)
        """)

    # ── Inputs ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        bt_ticker = st.text_input("Ticker", value="SPY", max_chars=10).upper()
    with c2:
        bt_strategy = st.selectbox(
            "Strategy",
            ["Short Strangle", "Iron Condor", "Short Put", "Jade Lizard", "Covered Call"],
        )
    with c3:
        bt_start = st.date_input(
            "Start Date",
            value=datetime(2022, 1, 1).date(),
            min_value=datetime(2015, 1, 1).date(),
        )
    with c4:
        bt_end = st.date_input(
            "End Date",
            value=datetime(2024, 12, 31).date(),
        )

    run_bt = st.button("Run Backtest", type="primary")

    if run_bt:
        if bt_start >= bt_end:
            st.error("Start date must be before end date.")
        else:
            with st.spinner(f"Running {bt_strategy} backtest on {bt_ticker} ({bt_start} to {bt_end})..."):
                try:
                    engine = BacktestEngine(portfolio_value=portfolio_value)
                    result = engine.run(
                        ticker=bt_ticker,
                        strategy=bt_strategy,
                        start_date=bt_start.strftime('%Y-%m-%d'),
                        end_date=bt_end.strftime('%Y-%m-%d'),
                    )

                    if result.total_trades == 0:
                        st.warning("No trades were opened in this period. Try a wider date range or different strategy.")
                    else:
                        # ── Summary metrics ──
                        st.subheader("Performance Summary")
                        m1, m2, m3, m4, m5, m6 = st.columns(6)
                        m1.metric("Total Trades", result.total_trades)
                        m2.metric("Win Rate", f"{result.win_rate*100:.1f}%")
                        m3.metric("Total P&L", f"${result.total_pnl:+,.2f}")
                        m4.metric("Avg P&L / Trade", f"${result.avg_pnl_per_trade:+.2f}")
                        m5.metric("Max Drawdown", f"${result.max_drawdown:,.2f}")
                        m6.metric("Profit Factor", f"{result.profit_factor:.2f}")

                        m7, m8, m9, m10 = st.columns(4)
                        m7.metric("Avg Credit", f"${result.avg_credit:.2f}/share")
                        m8.metric("Avg Winner", f"${result.avg_winner:+.2f}")
                        m9.metric("Avg Loser", f"${result.avg_loser:+.2f}")
                        m10.metric("Avg DTE Held", f"{result.avg_dte_held:.0f} days")

                        # ── Equity curve ──
                        st.subheader("Equity Curve (Cumulative P&L)")
                        eq_dates = [d for d, _ in result.equity_curve]
                        eq_vals = [v for _, v in result.equity_curve]

                        fig_eq = go.Figure()
                        fig_eq.add_trace(go.Scatter(
                            x=eq_dates, y=eq_vals,
                            mode='lines',
                            name='Cumulative P&L',
                            line=dict(color='#00cc88', width=2),
                            fill='tozeroy',
                            fillcolor='rgba(0,204,136,0.08)',
                        ))
                        fig_eq.add_hline(y=0, line_color='rgba(255,255,255,0.3)', line_dash='dot')
                        fig_eq.update_layout(
                            yaxis_title='Cumulative P&L ($)',
                            xaxis_title='Date',
                            template='plotly_dark',
                            hovermode='x unified',
                        )
                        st.plotly_chart(fig_eq, use_container_width=True)

                        # ── Two charts side by side ──
                        col_l, col_r = st.columns(2)

                        with col_l:
                            # Close reason breakdown
                            if result.close_reason_breakdown:
                                labels = list(result.close_reason_breakdown.keys())
                                values = list(result.close_reason_breakdown.values())
                                readable = {
                                    'profit_target': '50% Profit Target',
                                    'stop_loss': '2x Stop Loss',
                                    'dte_exit': '21 DTE Exit',
                                    'expired': 'Expired',
                                    'end_of_backtest': 'End of Period',
                                }
                                labels = [readable.get(l, l) for l in labels]
                                fig_pie = px.pie(
                                    names=labels, values=values,
                                    title='Exit Reason Breakdown',
                                    color_discrete_sequence=px.colors.qualitative.Set2,
                                )
                                fig_pie.update_layout(template='plotly_dark')
                                st.plotly_chart(fig_pie, use_container_width=True)

                        with col_r:
                            # P&L distribution histogram
                            pnls = [t.pnl_dollars for t in result.trades if t.pnl_dollars is not None]
                            fig_hist = go.Figure()
                            fig_hist.add_trace(go.Histogram(
                                x=pnls,
                                nbinsx=30,
                                name='P&L per Trade',
                                marker_color=['#00cc88' if p > 0 else '#ff6666' for p in pnls],
                            ))
                            fig_hist.add_vline(x=0, line_color='white', line_dash='dash')
                            fig_hist.update_layout(
                                title='P&L Distribution per Trade ($)',
                                xaxis_title='P&L ($)',
                                yaxis_title='Count',
                                template='plotly_dark',
                            )
                            st.plotly_chart(fig_hist, use_container_width=True)

                        # ── Trade table ──
                        st.subheader(f"Trade History ({result.total_trades} trades)")
                        trade_rows = []
                        for t in result.trades:
                            row = {
                                'Entry': t.entry_date,
                                'Expiry': t.expiry_date,
                                'DTE In': t.dte_at_entry,
                                'Price': f"${t.entry_price:.2f}",
                                'Call Strike': f"${t.call_strike:.1f}" if t.call_strike else '—',
                                'Put Strike': f"${t.put_strike:.1f}" if t.put_strike else '—',
                                'Credit': f"${t.entry_credit:.2f}",
                                'HV at Entry': f"{t.entry_hv*100:.1f}%",
                                'Close': t.close_date or '—',
                                'DTE Out': t.dte_at_close if t.dte_at_close is not None else '—',
                                'Debit': f"${t.close_debit:.2f}" if t.close_debit is not None else '—',
                                'P&L': f"${t.pnl_dollars:+.2f}" if t.pnl_dollars is not None else '—',
                                'Contracts': t.contracts,
                                'Exit Reason': t.close_reason.replace('_', ' ').title(),
                            }
                            trade_rows.append(row)

                        df_trades = pd.DataFrame(trade_rows)

                        def color_pnl(val):
                            if isinstance(val, str) and val.startswith('$'):
                                try:
                                    n = float(val.replace('$', '').replace('+', '').replace(',', ''))
                                    if n > 0:
                                        return 'color: #00cc88'
                                    elif n < 0:
                                        return 'color: #ff6666'
                                except Exception:
                                    pass
                            return ''

                        st.dataframe(
                            df_trades.style.map(color_pnl, subset=['P&L']),
                            hide_index=True,
                            use_container_width=True,
                        )

                        # ── Benchmark note ──
                        st.info(
                            f"**Note:** Credits are understated vs live trading because HV is used as IV proxy. "
                            f"Real IV typically exceeds HV by 3-7 percentage points (the volatility risk premium). "
                            f"This makes the backtest **conservative** — live results with real IV data would likely show higher credits and better P&L."
                        )

                except Exception as e:
                    st.error(f"Backtest failed: {e}")
                    import traceback
                    with st.expander("Error details"):
                        st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PAPER TRADING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Paper Trading":
    st.header("Paper Trading")
    st.caption(
        "Automated paper trading mode — scans the watchlist, auto-enters qualifying trades, "
        "and monitors open positions for 50% profit / 2x stop / 21 DTE exits. "
        "All trades stored in paper_portfolio.json, separate from your live portfolio."
    )

    from src.paper_trader import PaperTrader
    from src.portfolio import Portfolio

    # ── Watchlist config ──
    pt_tickers = st.text_input(
        "Watchlist (comma-separated)",
        value=", ".join(DEFAULT_WATCHLIST),
        key="pt_watchlist",
    )
    pt_watchlist = [t.strip().upper() for t in pt_tickers.split(',') if t.strip()]

    # ── Action buttons ──
    b1, b2, b3, b4 = st.columns(4)
    run_cycle_btn    = b1.button("Run Full Cycle",    type="primary",   use_container_width=True,
                                 help="Process exits then scan for new entries")
    scan_entries_btn = b2.button("Scan Entries Only", type="secondary", use_container_width=True,
                                 help="Scan for new entries without touching open positions")
    check_exits_btn  = b3.button("Check Exits Only",  type="secondary", use_container_width=True,
                                 help="Re-price and close positions that hit exit rules")
    refresh_btn      = b4.button("Refresh MTM",       type="secondary", use_container_width=True,
                                 help="Reload the mark-to-market table without running a cycle")

    # ── Session state for cycle results ──
    if 'pt_result' not in st.session_state:
        st.session_state.pt_result = None
    if 'pt_error' not in st.session_state:
        st.session_state.pt_error = None

    # ── Run actions ──
    if run_cycle_btn:
        with st.spinner("Running full paper trading cycle..."):
            try:
                pt = PaperTrader(portfolio_value=portfolio_value, watchlist=pt_watchlist)
                result = pt.run()
                st.session_state.pt_result = ('cycle', result)
                st.session_state.pt_error = None
            except Exception as e:
                st.session_state.pt_error = str(e)

    elif scan_entries_btn:
        with st.spinner("Scanning for new entries..."):
            try:
                pt = PaperTrader(portfolio_value=portfolio_value, watchlist=pt_watchlist)
                result = pt.scan_entries_only()
                st.session_state.pt_result = ('entries', result)
                st.session_state.pt_error = None
            except Exception as e:
                st.session_state.pt_error = str(e)

    elif check_exits_btn:
        with st.spinner("Checking exits on open positions..."):
            try:
                pt = PaperTrader(portfolio_value=portfolio_value, watchlist=pt_watchlist)
                exits = pt.check_exits_only()
                st.session_state.pt_result = ('exits', exits)
                st.session_state.pt_error = None
            except Exception as e:
                st.session_state.pt_error = str(e)

    if st.session_state.pt_error:
        st.error(f"Error: {st.session_state.pt_error}")

    # ── Show last cycle result ──
    if st.session_state.pt_result:
        mode, data = st.session_state.pt_result
        st.divider()

        if mode == 'cycle':
            c1, c2, c3 = st.columns(3)
            c1.metric("Exits Triggered",  len(data['exits']))
            c2.metric("New Entries",       len(data['entered']))
            c3.metric("Open Positions",    data['open_count'])

            if data['exits']:
                st.subheader("Positions Closed This Cycle")
                exit_rows = []
                for e in data['exits']:
                    exit_rows.append({
                        'Ticker': e['ticker'],
                        'Exit Reason': e['reason'],
                        'Entry Credit': f"${e['entry_credit']:.4f}" if e.get('entry_credit') else '—',
                        'Close Debit':  f"${e['close_debit']:.4f}" if e.get('close_debit') else '—',
                        'P&L': f"${e['pnl']:+.2f}" if e.get('pnl') is not None else '—',
                    })
                st.dataframe(pd.DataFrame(exit_rows), hide_index=True, use_container_width=True)

            if data['entered']:
                st.subheader("Positions Entered This Cycle")
                entry_rows = []
                for e in data['entered']:
                    entry_rows.append({
                        'Ticker': e['ticker'],
                        'Strategy': e['strategy'],
                        'Credit': f"${e['credit']:.2f}",
                        'Expiration': e['expiration'],
                        'DTE': e['dte'],
                    })
                st.dataframe(pd.DataFrame(entry_rows), hide_index=True, use_container_width=True)

            if data['skipped']:
                with st.expander(f"Skipped ({len(data['skipped'])})"):
                    skip_rows = [{'Ticker': s['ticker'], 'Reason': s['reason']} for s in data['skipped']]
                    st.dataframe(pd.DataFrame(skip_rows), hide_index=True, use_container_width=True)

        elif mode == 'entries':
            st.subheader(f"Entry Scan — {data['timestamp']}")
            if data['entered']:
                entry_rows = []
                for e in data['entered']:
                    entry_rows.append({
                        'Ticker': e['ticker'],
                        'Strategy': e['strategy'],
                        'Credit': f"${e['credit']:.2f}",
                        'Expiration': e['expiration'],
                        'DTE': e['dte'],
                    })
                st.success(f"{len(data['entered'])} new position(s) entered.")
                st.dataframe(pd.DataFrame(entry_rows), hide_index=True, use_container_width=True)
            else:
                st.info("No new entries. All tickers failed entry criteria.")

            if data['skipped']:
                with st.expander(f"Skipped ({len(data['skipped'])})"):
                    skip_rows = [{'Ticker': s['ticker'], 'Reason': s['reason']} for s in data['skipped']]
                    st.dataframe(pd.DataFrame(skip_rows), hide_index=True, use_container_width=True)

        elif mode == 'exits':
            st.subheader("Exit Check")
            if data:
                st.warning(f"{len(data)} position(s) closed.")
                exit_rows = []
                for e in data:
                    exit_rows.append({
                        'Ticker': e['ticker'],
                        'Exit Reason': e['reason'],
                        'Entry Credit': f"${e['entry_credit']:.4f}" if e.get('entry_credit') else '—',
                        'Close Debit':  f"${e['close_debit']:.4f}" if e.get('close_debit') else '—',
                        'P&L': f"${e['pnl']:+.2f}" if e.get('pnl') is not None else '—',
                    })
                st.dataframe(pd.DataFrame(exit_rows), hide_index=True, use_container_width=True)
            else:
                st.success("No exits triggered — all positions within parameters.")

    # ══ Open Positions (live MTM) ══
    st.divider()
    st.subheader("Open Paper Positions — Live Mark-to-Market")

    paper_portfolio = Portfolio(value=portfolio_value, paper=True)
    open_positions  = paper_portfolio.open_positions()

    if not open_positions:
        st.info("No open paper positions. Run a full cycle or entry scan to add some.")
    else:
        with st.spinner("Re-pricing positions..."):
            try:
                pt_mtm = PaperTrader(portfolio_value=portfolio_value, watchlist=pt_watchlist)
                mtm_data = pt_mtm.get_positions_with_pnl()
            except Exception as e:
                st.error(f"Could not load live MTM: {e}")
                mtm_data = []

        if mtm_data:
            mtm_rows = []
            for p in mtm_data:
                flags = []
                if p['at_profit_target']: flags.append('TARGET')
                if p['at_stop_loss']:     flags.append('STOP')
                if p['at_dte_exit']:      flags.append('21DTE')
                mtm_rows.append({
                    'Ticker':     p['ticker'],
                    'Strategy':   p['strategy'],
                    'Entry Date': p['entry_date'],
                    'Expiration': p['expiration'],
                    'DTE Left':   p['dte_remaining'],
                    'Entry Credit': f"${p['entry_credit']:.2f}" if p['entry_credit'] else '—',
                    'Current Debit': f"${p['current_debit']:.2f}" if p['current_debit'] is not None else 'n/a',
                    'Unrealized P&L': f"${p['unrealized_pnl']:+.2f}" if p['unrealized_pnl'] is not None else 'n/a',
                    'P&L %':     f"{p['pnl_pct']:+.1f}%" if p['pnl_pct'] is not None else 'n/a',
                    'Flags':     ' | '.join(flags) if flags else '',
                })

            df_mtm = pd.DataFrame(mtm_rows)

            def color_pnl_cell(val):
                if isinstance(val, str) and val.startswith('$'):
                    try:
                        n = float(val.replace('$', '').replace('+', '').replace(',', ''))
                        if n > 0:   return 'color: #00cc88'
                        elif n < 0: return 'color: #ff6666'
                    except Exception:
                        pass
                return ''

            def color_flags(val):
                if 'STOP' in str(val) or 'TARGET' in str(val) or '21DTE' in str(val):
                    return 'background-color: #4a3300; color: #ffcc00'
                return ''

            st.dataframe(
                df_mtm.style
                    .map(color_pnl_cell, subset=['Unrealized P&L'])
                    .map(color_flags,    subset=['Flags']),
                hide_index=True,
                use_container_width=True,
            )

            # ── Unrealized P&L bar chart ──
            pnl_vals = [p['unrealized_pnl'] for p in mtm_data if p['unrealized_pnl'] is not None]
            tickers  = [p['ticker']         for p in mtm_data if p['unrealized_pnl'] is not None]
            if pnl_vals:
                colors = ['#00cc88' if v >= 0 else '#ff6666' for v in pnl_vals]
                fig_mtm = go.Figure(go.Bar(
                    x=tickers, y=pnl_vals,
                    marker_color=colors,
                    text=[f"${v:+.2f}" for v in pnl_vals],
                    textposition='outside',
                ))
                fig_mtm.add_hline(y=0, line_color='rgba(255,255,255,0.3)')
                fig_mtm.update_layout(
                    title='Unrealized P&L by Position ($)',
                    yaxis_title='Unrealized P&L ($)',
                    template='plotly_dark',
                    showlegend=False,
                )
                st.plotly_chart(fig_mtm, use_container_width=True)

    # ══ Paper Trade History ══
    st.divider()
    st.subheader("Paper Trade History")

    closed_paper = paper_portfolio.closed_positions()
    if not closed_paper:
        st.info("No closed paper trades yet.")
    else:
        hist_rows = []
        for pos in closed_paper:
            hist_rows.append({
                'Ticker':   pos.ticker,
                'Strategy': pos.strategy,
                'Entry':    pos.entry_date,
                'Close':    pos.close_date or '—',
                'Credit':   f"${pos.entry_credit:.2f}",
                'Debit':    f"${pos.close_debit:.2f}" if pos.close_debit is not None else '—',
                'P&L':      f"${pos.pnl:+.2f}" if pos.pnl is not None else '—',
                'Contracts': pos.contracts,
            })

        df_hist = pd.DataFrame(hist_rows)

        def color_hist_pnl(val):
            if isinstance(val, str) and val.startswith('$'):
                try:
                    n = float(val.replace('$', '').replace('+', '').replace(',', ''))
                    if n > 0:   return 'color: #00cc88'
                    elif n < 0: return 'color: #ff6666'
                except Exception:
                    pass
            return ''

        st.dataframe(
            df_hist.style.map(color_hist_pnl, subset=['P&L']),
            hide_index=True,
            use_container_width=True,
        )

        # ── Summary metrics ──
        pnl_vals = [pos.pnl for pos in closed_paper if pos.pnl is not None]
        if pnl_vals:
            total_pnl = sum(pnl_vals)
            wins      = sum(1 for v in pnl_vals if v > 0)
            win_rate  = wins / len(pnl_vals) * 100

            h1, h2, h3, h4 = st.columns(4)
            h1.metric("Total Trades",  len(pnl_vals))
            h2.metric("Win Rate",      f"{win_rate:.1f}%")
            h3.metric("Total P&L",     f"${total_pnl:+,.2f}")
            h4.metric("Avg P&L",       f"${total_pnl/len(pnl_vals):+.2f}")

            # ── Cumulative P&L chart ──
            cumulative = [sum(pnl_vals[:i+1]) for i in range(len(pnl_vals))]
            fig_hist_pnl = go.Figure()
            fig_hist_pnl.add_trace(go.Scatter(
                y=cumulative,
                mode='lines+markers',
                name='Cumulative P&L',
                line=dict(color='#00cc88', width=2),
                fill='tozeroy',
                fillcolor='rgba(0,204,136,0.1)',
            ))
            fig_hist_pnl.add_hline(y=0, line_color='rgba(255,255,255,0.3)', line_dash='dot')
            fig_hist_pnl.update_layout(
                title='Paper Trade Cumulative P&L',
                yaxis_title='P&L ($)',
                xaxis_title='Trade #',
                template='plotly_dark',
            )
            st.plotly_chart(fig_hist_pnl, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: LONG OPTIONS EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Long Options":
    st.header("Long Options Evaluator")
    st.caption(
        "Evaluate buying a call or put. "
        "Key difference from selling: you want LOW IV, and theta works AGAINST you every day."
    )

    from src.long_evaluator import LongEvaluator

    # ── Inputs ──
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        lo_ticker = st.text_input("Ticker", value="NVDA", max_chars=10, key="lo_ticker").upper()
    with c2:
        lo_type = st.selectbox("Option Type", ["call", "put"], key="lo_type")
    with c3:
        _DTE_OPTIONS = [1, 3, 6, 8, 10, 17, 23, 30, 45, 60, 90, 120, 180]
        _DTE_LABELS  = {
            1:   "1d  — same day",
            3:   "3d  — short term",
            6:   "6d",
            8:   "8d",
            10:  "10d",
            17:  "17d",
            23:  "23d",
            30:  "30d  — ~1 month",
            45:  "45d  — 6 weeks",
            60:  "60d  — 2 months",
            90:  "90d  — 3 months",
            120: "120d — 4 months",
            180: "180d — 6 months",
        }
        lo_dte = st.selectbox(
            "Target DTE",
            options=_DTE_OPTIONS,
            index=7,            # default: 30d
            format_func=lambda d: _DTE_LABELS.get(d, str(d)),
            key="lo_dte",
        )
    with c4:
        st.write("")
        st.write("")
        lo_btn = st.button("Evaluate", type="primary", use_container_width=True)

    if lo_btn:
        with st.spinner(f"Evaluating {lo_ticker} {lo_dte}-DTE {lo_type.upper()}..."):
            try:
                ev     = LongEvaluator()
                result = ev.evaluate(lo_ticker, lo_type, target_dte=lo_dte)
            except Exception as e:
                st.error(f"Evaluation failed: {e}")
                import traceback
                with st.expander("Error details"):
                    st.code(traceback.format_exc())
                result = None

        if result and not result.error:

            # ── Warnings banner ──
            for w in result.warnings:
                st.warning(w)

            # ── IV Environment ──────────────────────────────────────────────
            st.subheader("IV Environment")
            assessment_colors = {
                'cheap':    ('green',  "CHEAP — Good time to buy"),
                'fair':     ('blue',   "FAIR — Acceptable"),
                'elevated': ('orange', "ELEVATED — You are overpaying"),
                'expensive':('red',    "EXPENSIVE — Selling environment"),
            }
            color, verdict = assessment_colors.get(
                result.iv_assessment, ('grey', result.iv_assessment.upper())
            )
            st.markdown(
                f"<div style='background:{'#1a4d1a' if color=='green' else '#1a2d4d' if color=='blue' else '#4d2e00' if color=='orange' else '#4d1a1a'};"
                f"padding:12px;border-radius:8px;font-size:1.1em;font-weight:bold;'>"
                f"{verdict}</div>",
                unsafe_allow_html=True,
            )
            st.caption(result.iv_note)
            st.write("")

            iv1, iv2, iv3, iv4, iv5 = st.columns(5)
            iv1.metric("IV Rank",     f"{result.iv_rank:.0f}",
                       help="0 = cheapest IV has been in a year, 100 = most expensive. Want <30 to buy.")
            iv2.metric("IV Pct",      f"{result.iv_percentile:.0f}",
                       help="% of days in the past year where IV was lower than today")
            iv3.metric("ATM IV",      f"{result.current_iv*100:.1f}%",
                       help="Implied volatility of near-ATM options")
            iv4.metric("HV20",        f"{result.hv20*100:.1f}%",
                       help="20-day realized volatility")
            iv5.metric("VRP Ratio",   f"{result.vrp_ratio:.2f}x",
                       delta="Options expensive vs realized" if result.vrp_ratio > 1.3 else "IV near realized vol",
                       delta_color="inverse" if result.vrp_ratio > 1.3 else "normal",
                       help="IV/HV. >1.3 means you're paying a big premium over what the market has actually moved")

            # ── Expected Move ────────────────────────────────────────────────
            st.subheader(f"Expected Move — {result.actual_dte} DTE  ({result.expiry})")
            em1, em2, em3, em4 = st.columns(4)
            em1.metric("Current Price",  f"${result.price:,.2f}")
            em2.metric(f"1-SD Move",     f"±${result.expected_move_1sd:,.2f}  ({result.expected_move_pct_1sd:.1f}%)",
                       help="1 standard deviation move over the holding period. ~68% chance price stays within this range.")
            em3.metric("Upper 1-SD",     f"${result.upper_1sd:,.2f}")
            em4.metric("Lower 1-SD",     f"${result.lower_1sd:,.2f}")

            st.caption(
                f"2-SD range: ${result.lower_1sd - result.expected_move_1sd:,.2f} — "
                f"${result.upper_1sd + result.expected_move_1sd:,.2f}  "
                f"(±${result.expected_move_2sd:,.2f}, ~95% probability)  |  "
                f"Formula: Price x IV x sqrt(DTE/365)"
            )

            # Visual: price line with expected move ranges
            import numpy as np
            strikes_plot = [c.strike for c in result.candidates if c is not None]
            bes_plot     = [c.break_even for c in result.candidates if c is not None]

            fig_em = go.Figure()
            # Shaded 1-SD band
            fig_em.add_shape(type="rect",
                x0=-0.5, x1=3.5,
                y0=result.lower_1sd, y1=result.upper_1sd,
                fillcolor="rgba(0,204,136,0.1)", line_width=0,
            )
            fig_em.add_shape(type="rect",
                x0=-0.5, x1=3.5,
                y0=result.lower_1sd - result.expected_move_1sd,
                y1=result.lower_1sd,
                fillcolor="rgba(255,102,102,0.05)", line_width=0,
            )
            fig_em.add_shape(type="rect",
                x0=-0.5, x1=3.5,
                y0=result.upper_1sd,
                y1=result.upper_1sd + result.expected_move_1sd,
                fillcolor="rgba(255,102,102,0.05)", line_width=0,
            )
            fig_em.add_hline(y=result.price, line_dash="solid", line_color="white",
                             annotation_text=f"Current ${result.price:,.2f}")
            fig_em.add_hline(y=result.upper_1sd, line_dash="dash", line_color="#00cc88",
                             annotation_text=f"+1SD ${result.upper_1sd:,.2f}")
            fig_em.add_hline(y=result.lower_1sd, line_dash="dash", line_color="#00cc88",
                             annotation_text=f"-1SD ${result.lower_1sd:,.2f}")
            for c in result.candidates:
                if c:
                    color_be = "#ff6666" if (
                        (lo_type == 'call' and c.break_even > result.upper_1sd) or
                        (lo_type == 'put'  and c.break_even < result.lower_1sd)
                    ) else "#ffcc00"
                    fig_em.add_hline(
                        y=c.break_even,
                        line_dash="dot", line_color=color_be,
                        annotation_text=f"BE {c.label}: ${c.break_even:,.2f}",
                        annotation_position="right",
                    )
            fig_em.update_layout(
                title=f"{lo_ticker} Price Levels vs Break-evens (1-SD band = green)",
                yaxis_title="Price ($)",
                xaxis_visible=False,
                template="plotly_dark",
                height=380,
            )
            st.plotly_chart(fig_em, use_container_width=True)

            # ── Strike Analysis Table ────────────────────────────────────────
            st.subheader("Strike Analysis")
            st.caption(
                "Break-even = price the stock must reach at expiry to profit. "
                "POP = probability stock ends beyond break-even. "
                "Red = break-even is outside the 1-SD expected move."
            )

            if result.candidates:
                rows = []
                for c in result.candidates:
                    beyond_1sd = (
                        (lo_type == 'call' and c.break_even > result.upper_1sd) or
                        (lo_type == 'put'  and c.break_even < result.lower_1sd)
                    )
                    rows.append({
                        'Strike Label':     c.label,
                        'Strike':           f"${c.strike:,.2f}",
                        'Delta':            f"{c.delta:+.2f}",
                        'Option IV':        f"{c.implied_vol:.1f}%",
                        'Premium':          f"${c.premium:.2f}",
                        'Break-even':       f"${c.break_even:,.2f}",
                        'Move Needed':      f"${c.move_needed:.2f}  ({c.move_needed_pct:.1f}%)",
                        'Daily Theta':      f"${c.daily_theta:.3f}",
                        'Theta %/day':      f"{c.theta_pct_daily:.2f}%",
                        'POP to BE':        f"{c.pop_to_be*100:.0f}%",
                        'Beyond 1-SD':      "YES - risky" if beyond_1sd else "within range",
                    })

                df_strikes = pd.DataFrame(rows)

                def color_beyond(val):
                    if val == "YES - risky":
                        return 'color: #ff6666'
                    return 'color: #00cc88'

                def color_theta(val):
                    try:
                        n = float(val.replace('%','').replace('$',''))
                        return 'color: #ff9900' if n < 0 else ''
                    except Exception:
                        return ''

                st.dataframe(
                    df_strikes.style
                        .map(color_beyond, subset=['Beyond 1-SD'])
                        .map(color_theta, subset=['Daily Theta']),
                    hide_index=True,
                    use_container_width=True,
                )

                # ── Theta decay chart ──
                st.subheader("Daily Theta Cost — How Much You Lose Per Day While Waiting")
                labels_c  = [c.label for c in result.candidates]
                thetas_c  = [abs(c.daily_theta) for c in result.candidates]
                prems_c   = [c.premium for c in result.candidates]
                theta_pcts = [c.theta_pct_daily for c in result.candidates]

                fig_th = go.Figure()
                fig_th.add_trace(go.Bar(
                    name="Daily Theta ($)",
                    x=labels_c, y=thetas_c,
                    marker_color="#ff9900",
                    text=[f"${v:.3f}" for v in thetas_c],
                    textposition="outside",
                ))
                fig_th.update_layout(
                    title="Daily Theta Cost by Strike ($ per share)",
                    yaxis_title="$/day lost to time decay",
                    template="plotly_dark",
                    showlegend=False,
                )
                st.plotly_chart(fig_th, use_container_width=True)

            # ── Debit Spreads ────────────────────────────────────────────────
            if result.debit_spreads:
                st.subheader("Debit Spread Alternatives")
                st.caption(
                    "A debit spread buys one strike and sells a higher/lower one. "
                    "It reduces your premium cost, daily theta, and vega — but caps your max profit."
                )

                for ds in result.debit_spreads:
                    with st.expander(f"{ds.label}  |  Net cost: ${ds.net_debit:.2f}  |  Max profit: ${ds.max_profit:.2f}  |  Max return: {ds.max_return_pct:.0f}%"):
                        dc1, dc2, dc3, dc4, dc5 = st.columns(5)
                        dc1.metric("Net Debit",     f"${ds.net_debit:.2f}",
                                   help="What you pay to open — max loss")
                        dc2.metric("Max Profit",    f"${ds.max_profit:.2f}",
                                   help=f"Achieved if stock goes past ${ds.short_strike:.0f} at expiry")
                        dc3.metric("Break-even",    f"${ds.break_even:.2f}",
                                   help="Stock price where you start making money at expiry")
                        dc4.metric("Max Return",    f"{ds.max_return_pct:.0f}%",
                                   help="Max profit / net debit — your return if you're right")
                        dc5.metric("Daily Theta",   f"${ds.theta_per_day:.3f}",
                                   delta=f"vs naked: {ds.theta_vs_naked_pct:.0f}% less decay",
                                   delta_color="normal",
                                   help="Net daily theta on the spread vs buying the naked option")

                        st.caption(
                            f"Buy ${ds.long_strike:.0f} (${ds.long_premium:.2f}) + "
                            f"Sell ${ds.short_strike:.0f} (-${ds.short_premium:.2f}) = "
                            f"Net ${ds.net_debit:.2f} debit.  "
                            f"Width: ${ds.width:.0f}.  "
                            f"Risk/reward: risk ${ds.net_debit:.2f} to make ${ds.max_profit:.2f} "
                            f"({ds.max_return_pct:.0f}% return)."
                        )

            # ── Upcoming macro events ────────────────────────────────────────
            if result.upcoming_macro_events:
                st.subheader(f"Macro Events in the {result.actual_dte}-Day Window")
                macro_rows = [
                    {'Event': e['type'], 'Date': e['date'], 'Days Away': e['days_away']}
                    for e in result.upcoming_macro_events
                ]
                st.dataframe(pd.DataFrame(macro_rows), hide_index=True, use_container_width=True)
                st.caption(
                    "Each FOMC / CPI / NFP event can cause a sudden move that "
                    "benefits OR hurts your position. Make sure your DTE covers you past the event."
                )

            # ── Quick summary ────────────────────────────────────────────────
            st.divider()
            st.subheader("Quick Verdict")
            if result.candidates:
                atm = result.candidates[0]
                verdict_lines = [
                    f"**{lo_ticker} ${atm.strike:.0f} {lo_type.upper()} ({result.expiry})**",
                    f"- Premium: **${atm.premium:.2f}** per share  (${atm.premium*100:.0f} per contract)",
                    f"- Break-even at expiry: **${atm.break_even:.2f}** — stock must move **{atm.move_needed_pct:.1f}%** from here",
                    f"- Daily theta cost: **${abs(atm.daily_theta):.3f}/day** ({atm.theta_pct_daily:.2f}% of premium per day)",
                    f"- Probability of beating break-even: **{atm.pop_to_be*100:.0f}%**",
                    f"- IV assessment: **{result.iv_assessment.upper()}** (rank {result.iv_rank:.0f})",
                ]
                st.markdown("\n".join(verdict_lines))

        elif result and result.error:
            st.error(f"Could not evaluate {lo_ticker}: {result.error}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: VALUE SCREENER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Value Screener":
    st.header("Value Screener")
    st.caption(
        "Identifies potentially undervalued stocks by combining analyst price targets, "
        "P/E ratios, earnings growth, RSI, and 52-week low proximity. "
        "When a stock scores well AND options are cheap, it auto-suggests a call or debit spread."
    )

    # ── How it works expander ──
    with st.expander("How the scoring works", expanded=False):
        st.markdown("""
**Total score: 0 to 100** — split equally between fundamentals and technicals.

| Fundamental (50 pts) | What it checks |
|---|---|
| Analyst price target discount | How far below Wall Street's consensus target the stock trades |
| Forward P/E ratio | Whether the stock is cheap or expensive relative to next year's expected earnings |
| Earnings growth | Whether the company is still growing its profits |

| Technical (50 pts) | What it checks |
|---|---|
| 52-week low proximity | How close the stock is to its annual low (beaten-down = higher score) |
| RSI-14 | Whether the stock is "oversold" — heavily sold off and possibly due for a bounce |

**Labels:** Strong Value Signal (75+) · Watch List (55-74) · Neutral (35-54) · No Signal (<35)

**Options action:** If score ≥ 55 AND IV rank ≤ 35 → Buy Call · IV rank 36-50 → Debit Spread · Otherwise → Wait
        """)

    # ── Inputs ──
    vs_col1, vs_col2, vs_col3 = st.columns([3, 1, 1])
    with vs_col1:
        vs_tickers_raw = st.text_input(
            "Tickers to screen (comma-separated)",
            value=", ".join(DEFAULT_WATCHLIST),
            key="vs_tickers",
        )
    with vs_col2:
        vs_min_score = st.slider("Min score", min_value=0, max_value=100, value=0, step=5, key="vs_min")
    with vs_col3:
        st.write("")
        st.write("")
        vs_btn = st.button("Screen", type="primary", use_container_width=True, key="vs_btn")

    if vs_btn:
        vs_tickers = [t.strip().upper() for t in vs_tickers_raw.split(',') if t.strip()]
        from src.stock_screener import StockScreener
        screener = StockScreener()

        vs_progress = st.progress(0, text="Starting screen...")
        vs_results = []
        for i, ticker in enumerate(vs_tickers):
            vs_progress.progress((i + 1) / len(vs_tickers), text=f"Screening {ticker}...")
            ev = screener.screen(ticker)
            vs_results.append(ev)

        vs_progress.empty()

        # Filter and sort
        valid = [r for r in vs_results if not r.error and r.total_score >= vs_min_score]
        errored = [r for r in vs_results if r.error]

        # ── Summary metrics ──
        strong = [r for r in valid if r.score_label == 'Strong Value Signal']
        watchlist_hits = [r for r in valid if r.score_label == 'Watch List']
        buy_call_hits = [r for r in valid if r.options_action == 'buy_call']

        sm1, sm2, sm3, sm4 = st.columns(4)
        sm1.metric("Screened", len(vs_results))
        sm2.metric("Strong Value Signals", len(strong))
        sm3.metric("Watch List", len(watchlist_hits))
        sm4.metric("Buy Call Signals", len(buy_call_hits))

        if not valid:
            if errored:
                st.warning(f"No results above score {vs_min_score}. Data unavailable for: {', '.join(r.ticker for r in errored)}")
            else:
                st.info(f"No tickers scored above {vs_min_score}. Try lowering the minimum score.")
        else:
            # ── Summary table ──
            st.subheader("Ranked Results")

            def _action_badge(action):
                if action == 'buy_call':      return 'BUY CALL'
                if action == 'debit_spread':  return 'DEBIT SPREAD'
                return 'Wait'

            table_rows = []
            for r in valid:
                table_rows.append({
                    'Ticker':         r.ticker,
                    'Score':          f"{r.total_score:.0f}",
                    'Rating':         r.score_label,
                    'Price':          f"${r.price:.2f}" if r.price else '—',
                    'Analyst Target': f"${r.analyst_target:.0f}" if r.analyst_target else '—',
                    'Discount %':     f"{r.discount_to_target_pct:.0f}%" if r.analyst_target else '—',
                    'Fwd P/E':        f"{r.forward_pe:.1f}" if r.forward_pe else '—',
                    'RSI':            f"{r.rsi_14:.0f}",
                    'IV Rank':        f"{r.iv_rank:.0f}" if r.iv_rank else '—',
                    'Options Action': _action_badge(r.options_action),
                    '_ev':            r,
                })

            def _color_rating(val):
                if val == 'Strong Value Signal': return 'background-color: #1a5c2a; color: white'
                if val == 'Watch List':          return 'background-color: #1a3a5c; color: white'
                if val == 'Neutral':             return 'color: #aaaaaa'
                return ''

            def _color_action(val):
                if val == 'BUY CALL':     return 'background-color: #1a5c2a; color: white; font-weight: bold'
                if val == 'DEBIT SPREAD': return 'background-color: #1a3a5c; color: white'
                return ''

            df_vs = pd.DataFrame([{k: v for k, v in r.items() if k != '_ev'} for r in table_rows])
            st.dataframe(
                df_vs.style
                    .map(_color_rating, subset=['Rating'])
                    .map(_color_action, subset=['Options Action']),
                hide_index=True,
                use_container_width=True,
            )

            # ── Detail cards for Watch List or better ──
            top_hits = [r for r in table_rows if r['_ev'].score_label in ('Strong Value Signal', 'Watch List')]
            if top_hits:
                st.subheader("Plain-English Breakdown")
                st.caption("Expanded details for your Watch List and Strong Signal stocks.")

                for row in top_hits:
                    ev = row['_ev']
                    action_color = {'buy_call': 'green', 'debit_spread': 'blue'}.get(ev.options_action, 'grey')
                    expanded = ev.score_label == 'Strong Value Signal' or ev.options_action == 'buy_call'

                    with st.expander(
                        f"{ev.ticker}  —  {ev.score_label}  ({ev.total_score:.0f}/100)  |  Options: {_action_badge(ev.options_action)}",
                        expanded=expanded,
                    ):
                        # Metrics row
                        ec1, ec2, ec3, ec4, ec5, ec6 = st.columns(6)
                        ec1.metric("Price",          f"${ev.price:.2f}")
                        ec2.metric("Analyst Target", f"${ev.analyst_target:.0f}" if ev.analyst_target else '—',
                                   delta=f"{ev.discount_to_target_pct:.0f}% upside" if ev.discount_to_target_pct > 0 else None,
                                   delta_color='normal')
                        ec3.metric("Fwd P/E",        f"{ev.forward_pe:.1f}" if ev.forward_pe else '—')
                        ec4.metric("RSI",            f"{ev.rsi_14:.0f}",
                                   delta="Oversold" if ev.rsi_14 <= 30 else ("Weak" if ev.rsi_14 <= 40 else None),
                                   delta_color='normal' if ev.rsi_14 <= 40 else 'off')
                        ec5.metric("52w Low",        f"${ev.week52_low:.2f}",
                                   delta=f"{ev.pct_above_52w_low:.0f}% above low",
                                   delta_color='inverse' if ev.pct_above_52w_low <= 20 else 'off')
                        ec6.metric("IV Rank",        f"{ev.iv_rank:.0f}" if ev.iv_rank else '—')

                        # Score bar chart (fund vs tech split)
                        fig_score = go.Figure(go.Bar(
                            x=[ev.fundamental_score, ev.technical_score],
                            y=['Fundamental', 'Technical'],
                            orientation='h',
                            marker_color=['#4f9ef8', '#00c278'],
                            text=[f"{ev.fundamental_score:.0f}/50", f"{ev.technical_score:.0f}/50"],
                            textposition='inside',
                        ))
                        fig_score.update_layout(
                            title=f"Score Breakdown — {ev.total_score:.0f}/100",
                            xaxis=dict(range=[0, 50], title='Score'),
                            height=140,
                            margin=dict(l=10, r=10, t=30, b=10),
                            template='plotly_dark',
                        )
                        st.plotly_chart(fig_score, use_container_width=True)

                        # Plain-English text
                        if ev.plain_english:
                            st.markdown(ev.plain_english.replace('\n', '  \n'))

                        # Evaluate Options button — pre-loads Long Options page
                        if ev.options_action in ('buy_call', 'debit_spread'):
                            st.divider()
                            btn_cols = st.columns([1, 2])
                            with btn_cols[0]:
                                if st.button(
                                    f"Evaluate {ev.ticker} Options",
                                    key=f"eval_opts_{ev.ticker}",
                                    type="secondary",
                                ):
                                    st.session_state['lo_ticker_prefill'] = ev.ticker
                                    st.session_state['nav_page'] = 'Long Options'
                                    st.rerun()

            # ── 60-day price charts for top picks ──
            chart_picks = [r['_ev'] for r in top_hits[:4]]  # max 4 charts
            if chart_picks:
                st.subheader("60-Day Price Charts")
                st.caption("Price history with 52-week low/high reference lines.")
                chart_cols = st.columns(min(len(chart_picks), 2))
                for idx, ev in enumerate(chart_picks):
                    with chart_cols[idx % 2]:
                        try:
                            import yfinance as _yf
                            _hist = _yf.Ticker(ev.ticker).history(period='3mo')
                            if not _hist.empty:
                                fig_c = go.Figure()
                                fig_c.add_trace(go.Scatter(
                                    x=_hist.index, y=_hist['Close'],
                                    name=ev.ticker, line=dict(color='#4f9ef8', width=2),
                                ))
                                if ev.week52_low:
                                    fig_c.add_hline(y=ev.week52_low, line_dash='dash',
                                                    line_color='#ef4444',
                                                    annotation_text=f"52w Low ${ev.week52_low:.0f}")
                                if ev.week52_high:
                                    fig_c.add_hline(y=ev.week52_high, line_dash='dot',
                                                    line_color='#00c278',
                                                    annotation_text=f"52w High ${ev.week52_high:.0f}")
                                if ev.analyst_target:
                                    fig_c.add_hline(y=ev.analyst_target, line_dash='dash',
                                                    line_color='#f59e0b',
                                                    annotation_text=f"Target ${ev.analyst_target:.0f}")
                                fig_c.update_layout(
                                    title=f"{ev.ticker} — Score {ev.total_score:.0f}/100",
                                    template='plotly_dark', height=260,
                                    margin=dict(l=10, r=10, t=40, b=10),
                                    showlegend=False,
                                )
                                st.plotly_chart(fig_c, use_container_width=True)
                        except Exception:
                            st.caption(f"Chart unavailable for {ev.ticker}")

        if errored:
            with st.expander(f"Data errors ({len(errored)} tickers)"):
                for r in errored:
                    st.caption(f"{r.ticker}: {r.error}")

    # ── Pre-fill hint from Value Screener ──
    if 'lo_ticker_prefill' in st.session_state and page == 'Long Options':
        st.info(f"Pre-loaded from Value Screener: {st.session_state.pop('lo_ticker_prefill')}")
