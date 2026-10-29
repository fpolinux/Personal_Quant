import yfinance as yf
import numpy as np

def historical_prices(stock):
    ticker = yf.download(stock, start = "2024-10-08", end="2025-10-08")
    return ticker   


stock_prices = historical_prices("ardt")
stock_returns = np.log(stock_prices/stock_prices.shift(1))
std_dev_stock = stock_returns.std()

spy_prices = historical_prices("spy")
spy_returns = np.log(spy_prices/spy_prices.shift(1))
std_dev_spy = spy_returns.std()
sigma_i = std_dev_stock*np.sqrt(252)
sigma_m = std_dev_spy * np.sqrt(252)
print(sigma_i, sigma_m)
beta = 0.237 * sigma_i/sigma_m
print(beta)