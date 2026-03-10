# TastyTrade Research — Key Backtested Findings
# Source: tastytrade.com/research, Tom Sosnoff, Tony Battista, Tom Preston

## Core Philosophy
- The house (casino) wins not because every hand is won, but because the expected value per hand is positive and they play millions of hands.
- Options sellers are the casino. Sell premium consistently, manage risk mechanically, let the statistics work over time.
- **Mechanization over emotion**: Every rule below is derived from backtesting thousands of trades. Trust the system.

---

## The 45/21 Framework
- **Enter at 45 DTE**: Theta decay accelerates between 45 and 21 DTE. This window captures the sweet spot.
- **Exit at 21 DTE or 50% profit**: Whichever comes first.
- **Data backing**: In backtests across SPX, QQQ, IWM, GLD from 2005-2023, the 45-in/21-out framework outperformed holding to expiration by reducing max drawdown by ~30% while maintaining 85-90% of the profit.

## The 50% Profit Rule
- Closing at 50% max profit **increases win rate from ~65% to ~80%** in backtested strangles.
- Psychological benefit: consistent winners reinforce discipline.
- **Never hold for the last 10-20% of profit.** The incremental gain does not justify the risk.

## The 16-Delta Rule (1 Standard Deviation)
- 16-delta strikes = 1 standard deviation from current price.
- **~84% probability of expiring OTM** (one-tailed, from normal distribution).
- In backtests, 16-delta short strangles have an ~82-84% win rate over 5+ year periods.
- The 30-delta strike is more aggressive — ~70% probability of profit, more credit, but more losers.

## Position Sizing
- **2% of portfolio max per trade** for undefined-risk positions.
- **5% of portfolio max** for defined-risk positions.
- **50% buying power utilization cap** across all open positions.
- Small consistent positions + high number of trades = edge realization through diversification.

## The Number of Occurrences
- The core concept: the more uncorrelated trades you have open simultaneously, the closer your actual win rate approaches the theoretical win rate.
- **Target: 10-15 positions open simultaneously** across uncorrelated underlyings.
- Diversification across underlyings is more important than diversification across strategies.

## Implied vs. Realized Volatility (The Core Edge)
- **Implied volatility (IV) is consistently higher than realized volatility (RV)** on average.
- The difference (IV - RV) is known as the **volatility risk premium (VRP)**.
- TastyTrade data: IV overstated realized moves in the S&P 500 in ~70% of months from 2005-2023.
- **This is the statistical edge**: selling IV captures the VRP over time.

## IV Rank Thresholds
- IV rank > 50: Open premium-selling positions (strangles, condors, short puts)
- IV rank 30-50: Defined risk strategies only (iron condors, credit spreads)
- IV rank < 30: Reduce premium selling activity. Consider debit strategies.

## The Expected Value Framework
- Win rate × average win - Loss rate × average loss = Expected Value per trade
- Example strangle: 70% win rate × $200 win - 30% × $400 loss = $140 - $120 = **+$20 EV per trade**
- The goal is **consistent positive EV trades**, not home runs.

## Key Findings by Product
- **SPX/SPY**: Best liquid product for strangles. High liquidity, no assignment risk (cash settled).
- **QQQ, IWM, GLD, TLT**: Good secondary products. High liquidity, low gap risk.
- **Individual stocks**: Higher IV rank typically, but carry event risk (earnings, M&A). Size smaller.
- **Avoid**: Penny stocks, very low liquidity names, names with upcoming binary events.
