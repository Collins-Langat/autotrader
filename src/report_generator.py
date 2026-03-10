"""
PDF Report Generator
Produces a plain-English options trading brief from watchlist scan results.

Usage:
    from src.report_generator import generate_report
    path = generate_report(recommendations, vix=22.9, portfolio_value=25000)
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

# ── Colour palette ────────────────────────────────────────────────────────────
DARK_BG    = colors.HexColor('#1a1a2e')
ACCENT     = colors.HexColor('#4f9ef8')
GREEN      = colors.HexColor('#00c278')
ORANGE     = colors.HexColor('#f59e0b')
RED        = colors.HexColor('#ef4444')
LIGHT_GREY = colors.HexColor('#f0f4f8')
MID_GREY   = colors.HexColor('#94a3b8')
DARK_TEXT  = colors.HexColor('#1e293b')

# Score colours
SCORE_COLORS = {
    'Strong':   GREEN,
    'Good':     ACCENT,
    'Marginal': ORANGE,
    'Avoid':    RED,
}


def _styles():
    """Return a dict of named paragraph styles."""
    base = getSampleStyleSheet()
    s = {}

    s['title'] = ParagraphStyle(
        'title', parent=base['Normal'],
        fontSize=22, textColor=colors.white, fontName='Helvetica-Bold',
        spaceAfter=2, leading=26,
    )
    s['subtitle'] = ParagraphStyle(
        'subtitle', parent=base['Normal'],
        fontSize=10, textColor=MID_GREY, fontName='Helvetica',
        spaceAfter=0,
    )
    s['section'] = ParagraphStyle(
        'section', parent=base['Normal'],
        fontSize=13, textColor=DARK_TEXT, fontName='Helvetica-Bold',
        spaceBefore=14, spaceAfter=4,
    )
    s['ticker'] = ParagraphStyle(
        'ticker', parent=base['Normal'],
        fontSize=16, textColor=DARK_TEXT, fontName='Helvetica-Bold',
        spaceBefore=0, spaceAfter=2,
    )
    s['label'] = ParagraphStyle(
        'label', parent=base['Normal'],
        fontSize=8, textColor=MID_GREY, fontName='Helvetica',
        spaceAfter=1,
    )
    s['value'] = ParagraphStyle(
        'value', parent=base['Normal'],
        fontSize=11, textColor=DARK_TEXT, fontName='Helvetica-Bold',
        spaceAfter=4,
    )
    s['body'] = ParagraphStyle(
        'body', parent=base['Normal'],
        fontSize=9.5, textColor=DARK_TEXT, fontName='Helvetica',
        leading=14, spaceAfter=3,
    )
    s['plain_line'] = ParagraphStyle(
        'plain_line', parent=base['Normal'],
        fontSize=9, textColor=DARK_TEXT, fontName='Helvetica',
        leading=13, spaceAfter=2, leftIndent=8,
    )
    s['plain_head'] = ParagraphStyle(
        'plain_head', parent=base['Normal'],
        fontSize=9, textColor=ACCENT, fontName='Helvetica-Bold',
        leading=13, spaceAfter=1, spaceBefore=6,
    )
    s['pass_body'] = ParagraphStyle(
        'pass_body', parent=base['Normal'],
        fontSize=8.5, textColor=MID_GREY, fontName='Helvetica',
        leading=12, spaceAfter=2,
    )
    s['footer'] = ParagraphStyle(
        'footer', parent=base['Normal'],
        fontSize=7.5, textColor=MID_GREY, fontName='Helvetica',
        alignment=TA_CENTER,
    )
    return s


def _header_block(doc_elements, vix, portfolio_value, n_ops, n_pass, date_str, s):
    """Render the dark header banner."""
    # Title table (dark background)
    title_data = [[
        Paragraph('Options Trader Daily Brief', s['title']),
        Paragraph(
            f'<para align="right"><font color="#94a3b8">{date_str}</font></para>',
            ParagraphStyle('tr', fontSize=9, textColor=MID_GREY,
                           fontName='Helvetica', leading=14),
        ),
    ]]
    title_table = Table(title_data, colWidths=[4.5*inch, 2.5*inch])
    title_table.setStyle(TableStyle([
        ('BACKGROUND',   (0, 0), (-1, -1), DARK_BG),
        ('TOPPADDING',   (0, 0), (-1, -1), 14),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 10),
        ('LEFTPADDING',  (0, 0), (-1, -1), 16),
        ('RIGHTPADDING', (0, 0), (-1, -1), 16),
        ('VALIGN',       (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    doc_elements.append(title_table)

    # Stats bar
    stats_data = [[
        Paragraph('<font color="#94a3b8">VIX</font>', s['label']),
        Paragraph('<font color="#94a3b8">Portfolio</font>', s['label']),
        Paragraph('<font color="#94a3b8">Opportunities</font>', s['label']),
        Paragraph('<font color="#94a3b8">Passed</font>', s['label']),
    ], [
        Paragraph(f'{vix:.1f}', s['value']),
        Paragraph(f'${portfolio_value:,.0f}', s['value']),
        Paragraph(str(n_ops), s['value']),
        Paragraph(str(n_pass), s['value']),
    ]]
    stats_table = Table(stats_data, colWidths=[1.75*inch]*4)
    stats_table.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), LIGHT_GREY),
        ('TOPPADDING',    (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING',   (0, 0), (-1, -1), 14),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 14),
        ('BOX',           (0, 0), (-1, -1), 0.5, MID_GREY),
    ]))
    doc_elements.append(stats_table)
    doc_elements.append(Spacer(1, 10))


def _score_badge_table(score, label, s):
    """Return a small inline table showing the trade score."""
    badge_color = SCORE_COLORS.get(label, ACCENT)
    data = [[
        Paragraph(
            f'<font color="white"><b>Score: {score:.0f}/100 — {label}</b></font>',
            ParagraphStyle('badge', fontSize=9, fontName='Helvetica-Bold',
                           textColor=colors.white, leading=12),
        )
    ]]
    t = Table(data, colWidths=[2.2*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), badge_color),
        ('TOPPADDING',    (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING',   (0, 0), (-1, -1), 8),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 8),
        ('ROUNDEDCORNERS',(0, 0), (-1, -1), [4, 4, 4, 4]),
    ]))
    return t


def _metrics_row(label_vals: list, s):
    """Return a metrics table with alternating label/value pairs."""
    n = len(label_vals)
    col_w = 7.0 / n * inch
    labels = [Paragraph(lbl, s['label']) for lbl, _ in label_vals]
    values = [Paragraph(val, s['value']) for _, val in label_vals]
    t = Table([labels, values], colWidths=[col_w] * n)
    t.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), LIGHT_GREY),
        ('TOPPADDING',    (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING',   (0, 0), (-1, -1), 8),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 8),
        ('BOX',           (0, 0), (-1, -1), 0.3, MID_GREY),
    ]))
    return t


def _plain_english_block(plain_english: str, s):
    """Render plain-English paragraphs with section headers styled."""
    flowables = []
    for line in plain_english.split('\n'):
        line = line.strip()
        if not line:
            flowables.append(Spacer(1, 3))
        elif line.isupper() and len(line) < 60:
            # Section heading
            flowables.append(Paragraph(line, s['plain_head']))
        elif line.startswith('  ') or line.startswith('- '):
            flowables.append(Paragraph(line.strip(), s['plain_line']))
        else:
            flowables.append(Paragraph(line, s['body']))
    return flowables


def _trade_card(rec, price: float, s):
    """Return a list of flowables for one trade recommendation."""
    out = []

    # Ticker + strategy header row
    header_data = [[
        Paragraph(
            f'<b>{rec.ticker}</b>  '
            f'<font color="#4f9ef8" size="11">{rec.strategy}</font>',
            s['ticker'],
        ),
        _score_badge_table(rec.trade_score, rec.trade_score_label, s),
    ]]
    header_t = Table(header_data, colWidths=[4.8*inch, 2.2*inch])
    header_t.setStyle(TableStyle([
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING',    (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING',   (0, 0), (-1, -1), 0),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 0),
    ]))
    out.append(header_t)

    # Key metrics row
    expiry_info = ''
    dte_info = ''
    if rec.legs:
        exp_vals = [l.get('expiration', '') for l in rec.legs if l.get('expiration')]
        dte_vals = [l.get('dte', '') for l in rec.legs if l.get('dte')]
        if exp_vals: expiry_info = exp_vals[0]
        if dte_vals: dte_info = str(dte_vals[0])

    pop_pct = f"{rec.probability_of_profit*100:.0f}%"
    vrp_str = f"{rec.vrp_ratio:.2f}x" if rec.vrp_ratio else 'n/a'

    out.append(_metrics_row([
        ('Current Price',  f'${price:.2f}'),
        ('Income / Contract', f'${rec.entry_credit*100:.0f}'),
        ('IV Rank',        f'{rec.iv_rank:.0f}'),
        ('Win Probability', pop_pct),
        ('Expiry',         expiry_info or 'n/a'),
        ('DTE',            dte_info or 'n/a'),
        ('VRP Ratio',      vrp_str),
    ], s))
    out.append(Spacer(1, 6))

    # Plain-English explanation
    if rec.plain_english:
        out.extend(_plain_english_block(rec.plain_english, s))

    out.append(Spacer(1, 4))
    out.append(HRFlowable(width='100%', thickness=0.5, color=LIGHT_GREY))
    out.append(Spacer(1, 8))
    return out


def _passes_table(passes: list, s):
    """Compact table of tickers that were skipped and why."""
    out = []
    out.append(Paragraph('No Trade — Tickers Scanned But Skipped', s['section']))
    out.append(Paragraph(
        'These tickers were analyzed but did not meet the entry criteria today.',
        s['body'],
    ))
    out.append(Spacer(1, 4))

    rows = [['Ticker', 'Reason']]
    for ticker, reason in passes:
        # Strip technical jargon for the plain-English version
        plain_reason = reason
        if 'HR-010' in reason:
            plain_reason = 'Macro event (FOMC/CPI/NFP) within 2 days — too risky to enter'
        elif 'HR-001' in reason:
            plain_reason = 'Earnings announcement within the holding period — skip to avoid IV crush'
        elif 'EN-001' in reason or 'IV rank' in reason:
            plain_reason = 'Options are not expensive enough right now — better to wait'
        elif 'DTE' in reason:
            plain_reason = 'No suitable expiry date available in the 30-60 day window'
        elif 'credit' in reason.lower():
            plain_reason = 'Potential income too small to be worth the trade'
        elif 'Error' in reason:
            plain_reason = 'Data unavailable — check again later'
        rows.append([ticker, plain_reason])

    col_widths = [1.0*inch, 6.0*inch]
    t = Table(rows, colWidths=col_widths)
    t.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, 0), DARK_BG),
        ('TEXTCOLOR',     (0, 0), (-1, 0), colors.white),
        ('FONTNAME',      (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0, 0), (-1, -1), 8.5),
        ('TOPPADDING',    (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING',   (0, 0), (-1, -1), 8),
        ('ROWBACKGROUNDS',(0, 1), (-1, -1), [colors.white, LIGHT_GREY]),
        ('GRID',          (0, 0), (-1, -1), 0.3, MID_GREY),
        ('TEXTCOLOR',     (0, 1), (0, -1), DARK_TEXT),
        ('TEXTCOLOR',     (1, 1), (1, -1), MID_GREY),
    ]))
    out.append(t)
    return out


def _value_section(value_evals: list, s) -> list:
    """
    Render a compact 'Value Watch List' section for the PDF.
    value_evals: list of StockValueEval objects (from stock_screener.py).
    """
    out = []
    out.append(Paragraph('Value Watch List', s['section']))
    out.append(Paragraph(
        'Stocks identified as potentially undervalued based on analyst targets, '
        'P/E ratios, earnings growth, RSI, and 52-week low proximity. '
        'When options are also cheap, a call purchase may be warranted.',
        s['body'],
    ))
    out.append(Spacer(1, 4))

    action_labels = {
        'buy_call':    'BUY CALL',
        'debit_spread':'DEBIT SPREAD',
        'wait':        'Wait',
    }
    action_colors = {
        'buy_call':    GREEN,
        'debit_spread':ACCENT,
        'wait':        MID_GREY,
    }

    rows = [['Ticker', 'Score', 'Rating', 'Price', 'Target', 'Disc%', 'RSI', 'IV Rank', 'Options Action']]
    for ev in value_evals:
        target_str = f"${ev.analyst_target:.0f}" if ev.analyst_target else 'n/a'
        disc_str   = f"{ev.discount_to_target_pct:.0f}%" if ev.analyst_target else 'n/a'
        pe_str     = f"{ev.forward_pe:.1f}" if ev.forward_pe else 'n/a'
        iv_str     = f"{ev.iv_rank:.0f}" if ev.iv_rank else 'n/a'
        action     = action_labels.get(ev.options_action, ev.options_action)
        rows.append([
            ev.ticker,
            f"{ev.total_score:.0f}",
            ev.score_label,
            f"${ev.price:.2f}",
            target_str,
            disc_str,
            f"{ev.rsi_14:.0f}",
            iv_str,
            action,
        ])

    col_widths = [0.65*inch, 0.5*inch, 1.4*inch, 0.65*inch, 0.65*inch,
                  0.5*inch, 0.45*inch, 0.65*inch, 1.0*inch]
    t = Table(rows, colWidths=col_widths)

    # Base style
    ts = [
        ('BACKGROUND',    (0, 0), (-1, 0), DARK_BG),
        ('TEXTCOLOR',     (0, 0), (-1, 0), colors.white),
        ('FONTNAME',      (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0, 0), (-1, -1), 7.5),
        ('TOPPADDING',    (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING',   (0, 0), (-1, -1), 5),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 5),
        ('ROWBACKGROUNDS',(0, 1), (-1, -1), [colors.white, LIGHT_GREY]),
        ('GRID',          (0, 0), (-1, -1), 0.3, MID_GREY),
        ('TEXTCOLOR',     (0, 1), (0, -1), DARK_TEXT),
    ]
    # Color-code rating and action columns
    for row_idx, ev in enumerate(value_evals, start=1):
        rating_color = {
            'Strong Value Signal': GREEN,
            'Watch List':          ACCENT,
        }.get(ev.score_label, MID_GREY)
        ts.append(('TEXTCOLOR',  (2, row_idx), (2, row_idx), rating_color))
        ts.append(('FONTNAME',   (2, row_idx), (2, row_idx), 'Helvetica-Bold'))
        act_color = action_colors.get(ev.options_action, MID_GREY)
        ts.append(('TEXTCOLOR',  (8, row_idx), (8, row_idx), act_color))
        ts.append(('FONTNAME',   (8, row_idx), (8, row_idx), 'Helvetica-Bold'))

    t.setStyle(TableStyle(ts))
    out.append(t)
    out.append(Spacer(1, 6))

    # One-line plain-English for each Watch List or better result
    notable = [ev for ev in value_evals if ev.score_label in ('Strong Value Signal', 'Watch List')]
    for ev in notable[:4]:  # max 4 detail lines to keep PDF compact
        action_word = action_labels.get(ev.options_action, 'Wait')
        # Extract just the first two lines of the plain_english summary
        summary_lines = [ln.strip() for ln in ev.plain_english.split('\n') if ln.strip()]
        snippet = ' '.join(summary_lines[:3])[:200] + '...' if len(summary_lines) > 3 else ' '.join(summary_lines)
        out.append(Paragraph(
            f'<b>{ev.ticker}</b> ({ev.score_label}, {ev.total_score:.0f}/100): {snippet}',
            s['pass_body'],
        ))

    return out


def generate_report(
    recommendations,          # list of TradeRecommendation
    vix: float = 0.0,
    portfolio_value: float = 25000,
    prices: Optional[dict] = None,   # {ticker: price} — pass if available
    output_dir: str = '.',
    value_evals=None,                # optional list of StockValueEval from StockScreener
) -> str:
    """
    Generate a PDF report from a watchlist scan.
    Returns the file path of the saved PDF.
    """
    prices = prices or {}
    date_str = datetime.now().strftime('%B %d, %Y  %H:%M')
    fname = f"options_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    output_path = str(Path(output_dir) / fname)

    opportunities = [r for r in recommendations if r.action == 'open']
    passes = [
        (r.ticker, r.rationale[0] if r.rationale else 'No entry signal')
        for r in recommendations if r.action != 'open'
    ]

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.6*inch, leftMargin=0.6*inch,
        topMargin=0.5*inch,   bottomMargin=0.6*inch,
    )

    s = _styles()
    story = []

    # ── Header ──────────────────────────────────────────────────────────────
    _header_block(
        story, vix, portfolio_value,
        n_ops=len(opportunities), n_pass=len(passes),
        date_str=date_str, s=s,
    )

    # ── Intro blurb ─────────────────────────────────────────────────────────
    if opportunities:
        story.append(Paragraph(
            f'<b>Trade Opportunities ({len(opportunities)} found)</b>',
            s['section'],
        ))
        story.append(Paragraph(
            'The following trades meet all entry criteria. Each card explains '
            'what to do, how much you earn, and when to exit — in plain language.',
            s['body'],
        ))
        story.append(Spacer(1, 8))

        for rec in opportunities:
            price = prices.get(rec.ticker, 0.0)
            card = _trade_card(rec, price, s)
            story.append(KeepTogether(card[:6]))   # keep header + metrics together
            for elem in card[6:]:
                story.append(elem)
    else:
        story.append(Paragraph('No Trade Opportunities Today', s['section']))
        story.append(Paragraph(
            'The scanner did not find any tickers meeting all entry criteria. '
            'Common reasons: macro events within 2 days, IV rank too low, or '
            'earnings inside the holding window. Check back tomorrow.',
            s['body'],
        ))
        story.append(Spacer(1, 10))

    # ── Passes table ────────────────────────────────────────────────────────
    if passes:
        story.extend(_passes_table(passes, s))

    # ── Value Watch List section (optional) ─────────────────────────────────
    if value_evals:
        notable = [ev for ev in value_evals
                   if not ev.error and ev.score_label in ('Strong Value Signal', 'Watch List')]
        if notable:
            story.append(Spacer(1, 10))
            story.extend(_value_section(notable, s))

    # ── Footer ──────────────────────────────────────────────────────────────
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width='100%', thickness=0.5, color=MID_GREY))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        'This report is generated for educational and research purposes only. '
        'It does not constitute financial advice. All options trading involves '
        'risk of loss. Past performance does not guarantee future results. '
        'Data sourced from Yahoo Finance via yfinance. '
        'Always verify prices with your broker before placing trades.',
        s['footer'],
    ))

    doc.build(story)
    return output_path
