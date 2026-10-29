import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

def data(tickers, start):
    data = yf.download(tickers, start=start)
    return data

def return_series(tickers, start):
    raw_data = data(tickers, start)
    close_price_one = raw_data["Close"][tickers[0]]
    close_price_two = raw_data["Close"][tickers[1]]
    returns = pd.DataFrame()
    returns[tickers[0] + "_return"] = 100*np.log(close_price_one/close_price_one.shift()).dropna()
    returns[tickers[1] + "_return"] = 100*np.log(close_price_two/close_price_two.shift()).dropna()
    return returns.dropna()

def regression(tickers, start):
    returns = return_series(tickers, start)
    X = returns[tickers[1] + "_return"]
    Y = returns[tickers[0] + "_return"]
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    return model

def plot_regression(tickers, start):
    returns = return_series(tickers, start)
    model = regression(tickers, start)
    plt.scatter(returns[tickers[1] + "_return"], returns[tickers[0] + "_return"], color='blue', label='Data points')
    plt.plot(returns[tickers[1] + "_return"], model.predict(sm.add_constant(returns[tickers[1] + "_return"])), color='red', label='Regression line')
    plt.xlabel(tickers[1] + " Returns")
    plt.ylabel(tickers[0] + " Returns")
    plt.title("Regression of " + tickers[0] + " on " + tickers[1])
    plt.legend()
    plt.show()

regression_results = regression(['SPK.AX', 'SPK.NZ'], '2025-01-01')
plot_regression(['SPK.AX', 'SPK.NZ'], '2025-01-01')