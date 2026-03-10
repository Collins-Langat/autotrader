# Euan Sinclair — Positional Option Trading & Volatility Trading: Key Principles
# Source: "Positional Option Trading" and "Volatility Trading" by Euan Sinclair

## The Core Thesis: Edge-Based Trading
- Every trade must have a **statistical edge** — a positive expected value.
- "If you can't explain the edge, you don't have one."
- The edge in options selling comes from: (1) volatility risk premium, (2) mean reversion of IV.
- Trade only when edge is present. Patience is the hardest but most important discipline.

## Quantifying the Edge Before Trading
- Before any trade, estimate: **E[profit] = (IV - forecast_RV) × vega**
- If IV = 25% and your forecast RV = 20%, the edge is 5% of vega per position.
- Edge must exceed transaction costs (bid-ask spread + commissions) plus a buffer.
- Rule: **Edge / (max_loss) >= 0.15** for the trade to be worth taking.

## Volatility Forecasting Methods
Sinclair's hierarchy of reliability:
1. **GARCH models** — best statistical forecast of near-term RV
2. **Historical volatility** (20-day, 30-day rolling) — simple but effective baseline
3. **IV itself** — contains market's aggregate forecast, includes risk premium
4. **Analyst forecasts, news** — least reliable, most subject to bias

### Practical application:
- If HV20 < IV by > 5 percentage points: sell premium
- If HV20 > IV by > 5 percentage points: buy premium or reduce exposure

## Volatility Regimes
- Markets exhibit **volatility clustering**: high vol begets high vol; low vol begets low vol.
- Use **GARCH or simple exponential smoothing** to identify current regime.
- Trade size DOWN in high-vol regimes (higher uncertainty), even if selling is attractive.
- Regime identification:
  - VIX < 15: Low vol regime → reduce selling
  - VIX 15-25: Normal regime → standard sizing
  - VIX > 25: High vol regime → favorable for selling but size conservatively

## Skew Trading
- **Put skew** exists because investors pay for downside protection systematically.
- The implied skew overstates the realized skew in most markets → edge in selling puts.
- Skew trade: sell OTM puts, buy OTM calls (or reverse with defined-risk spread).
- Monitor **skew ratio**: if put IV / call IV > 1.3, the skew is rich and puts are expensive.

## Position Sizing with Kelly Criterion
- Sinclair advocates a **fractional Kelly** approach to sizing.
- Full Kelly: f = (edge) / (variance) — theoretically optimal but too volatile in practice.
- Use **half Kelly** or **quarter Kelly** to reduce drawdowns.
- Practical rule: never risk more than the half-Kelly size, which typically works out to 1-3% per trade.

## The Role of Correlation
- When volatility spikes, correlations across assets spike too.
- "All your diversification disappears exactly when you need it most."
- **Sinclair's warning**: 10 uncorrelated 2% positions in normal markets become one 15% position in a crisis.
- Hedge the tail: consider a small long VIX position or long put spread as a portfolio hedge.

## Patience and Trade Selection
- Sinclair's data: most of the edge in options trading comes from **the top 20% of opportunities**.
- Wait for high IV rank situations — they are where the real edge lives.
- "Being in cash while waiting for a good trade is not a cost — it is the strategy."
- Track your edge per trade in a log. If average edge per trade is declining, you're overtrading.

## Managing Losing Trades
- "The question is never whether to take a loss but when."
- Define exit before entry. Never hope or average into a losing options position.
- Sinclair's framework: if the original reason for the trade is no longer valid, exit.
- **Don't roll a loser just to avoid taking the loss.** Rolling a bad trade makes it a bigger bad trade.
