##Simple strategy where you buy once index > ma and then sell once index < ma. Assuming long only positions. 

import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Download data
data = yf.download("SPY", start="2020-01-01", interval="1d", auto_adjust=True)

# Calculate the 200-day Moving Average
# 24 day sma generates sharpe: 0.56 cf market 0.42
sp_ma = data["Close"].rolling(window=24).mean()
# 97 day ema generates sharpe: 0.57 cf market 0.42
#sp_ma = data["Close"].ewm(span=97).mean()

# Calculate the signal: 1 if Price > MA, 0 otherwise
data['Signal'] = np.where(data['Close'] > sp_ma, 1, 0)

# Calculate daily log returns of the underlying (SPY)
data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))

# Calculate Strategy Returns
# Shift the signal by 1 day because we can only act on today's signal tomorrow
data['Strategy_Returns'] = data['Signal'].shift(1) * data['Log_Returns']

# Calculate Cumulative Returns for comparison
data['Cumulative_Market'] = data['Log_Returns'].cumsum().apply(np.exp)
data['Cumulative_Strategy'] = data['Strategy_Returns'].cumsum().apply(np.exp)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(data['Cumulative_Market'], label='Buy & Hold SPY', alpha=0.6)
plt.plot(data['Cumulative_Strategy'], label='MA Cross Strategy', linewidth=2)
plt.title("Strategy Comparison: Buy & Hold vs. 24-day MA Trend Following")

risk_free_rate = 0.04
daily_rf = risk_free_rate / 252

# Calculate Excess Returns 
data['Market_Excess'] = data['Log_Returns'] - daily_rf

# Strategy Excess Returns (Daily)
data['Strategy_Excess'] = data['Strategy_Returns'] - daily_rf

# Calculate Annualized Sharpe Ratios
sharpe_market = (data['Market_Excess'].mean() / data['Market_Excess'].std()) * np.sqrt(252)
sharpe_strategy = (data['Strategy_Excess'].mean() / data['Strategy_Excess'].std()) * np.sqrt(252)

print(f"Annualized Market Sharpe Ratio: {sharpe_market:.2f}")
print(f"Annualized Strategy Sharpe Ratio: {sharpe_strategy:.2f}")

# Check how many days you were In (1) vs Out (0)
print(data['Signal'].value_counts())

# Check if the signal ever actually changed
print(f"Number of signal flips: {data['Signal'].diff().abs().sum()}")

plt.legend()
plt.show()

