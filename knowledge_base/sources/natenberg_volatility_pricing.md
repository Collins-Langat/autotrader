# Natenberg — Option Volatility & Pricing: Key Principles
# Source: "Option Volatility & Pricing" by Sheldon Natenberg (2nd Ed.)

## Core Principle: Volatility is the Key Variable
- The price of an option is primarily a function of **implied volatility (IV)**.
- All other inputs (price, strike, DTE, interest rate) are observable. **Volatility is the only unknown**.
- An experienced options trader is primarily a **volatility trader**, not a direction trader.

## Theoretical Value and Edge
- Every option has a **theoretical value** calculated from a pricing model (Black-Scholes or binomial).
- An option is **"cheap"** if its IV is below your forecast of realized volatility.
- An option is **"expensive"** if its IV is above your forecast of realized volatility.
- **Buy cheap volatility. Sell expensive volatility.**

## Volatility Skew
- In equity markets, puts consistently trade at higher IV than calls (**put skew** or **volatility skew**).
- Reason: demand for downside protection pushes put prices above theoretical value.
- **Implication for premium sellers**: OTM puts often have more edge to sell than OTM calls.
- Skew is the trader's ally: you're selling overpriced downside protection.

## The Greeks Framework

### Delta (Δ)
- Rate of change of option price with respect to underlying price.
- 16-delta ≈ 16% chance of expiring in-the-money (roughly).
- Delta is also the **hedge ratio**: a 16-delta option requires selling 16 shares to delta-hedge.
- Delta changes as underlying moves (this is Gamma).

### Gamma (Γ)
- Rate of change of Delta with respect to underlying price.
- **Short gamma = risk near expiration**. The position can lose money rapidly if the underlying moves.
- Gamma is always negative for short options positions.
- Gamma risk peaks at ATM, near expiration. This is why we exit at 21 DTE.

### Theta (Θ)
- Rate of time decay — how much the option loses in value per day, all else equal.
- **Short options positions have positive theta**: you collect theta daily.
- Theta accelerates as expiration approaches (most decay in final 30 days).
- The "rent" you collect for having sold volatility.

### Vega (V)
- Sensitivity of option price to a 1% change in implied volatility.
- **Short options = short vega**: if IV rises, your position loses value.
- Long options = long vega: if IV rises, your position gains value.
- **Key insight**: Short premium strategies lose money when IV spikes, even if the underlying hasn't moved.

## Volatility Mean Reversion
- Volatility is **mean-reverting**. It spikes and then returns to its historical average.
- This is the statistical foundation for selling options in high IV environments.
- Natenberg's guidance: "The most important volatility forecasting tool is the historical distribution of volatility."
- Monitor IV vs. its own history (IV rank, IV percentile) to identify extremes.

## Pricing Models and Their Limitations
- Black-Scholes assumes constant volatility, normal distribution of returns, continuous trading.
- Reality: volatility is not constant (volatility smile/skew exists), returns have fat tails.
- **The model is a tool, not the truth.** Use it to identify relative value, not absolute value.
- Fat tails mean: OTM put selling carries more real risk than B-S implies. Respect the skew.

## Position Management Principles
- A position's risk profile changes continuously as time passes and the underlying moves.
- Monitor Greeks daily, especially delta (directional exposure) and gamma (acceleration risk).
- **Adjust positions when delta gets too large** — this is active risk management.
- Natenberg's rule of thumb: if delta exceeds 0.50 of your net short premium, adjust.

## The Volatility Risk Premium (VRP)
- IV consistently overstates actual realized volatility in most markets, most of the time.
- This overstatement is the VRP — the excess that options buyers pay for protection.
- **The VRP is the structural edge that premium sellers exploit.**
- VRP is larger in equities (fear of downside) than commodities or FX.
- The VRP has persisted for decades because the demand for portfolio protection is structural.
