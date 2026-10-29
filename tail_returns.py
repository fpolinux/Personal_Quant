import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Downloads data and returns log_returns
def returns(ticker):
    data = yf.download(ticker, period="max", interval= "1d", auto_adjust = True, keepna = False, progress = True)
    close = data["Close"]
    log_returns = np.log(close/close.shift()).dropna()
    return log_returns

## Filters through returns above/below a threshold % and returns a scatter plot
## Automatically goes 'above' +ve thresholds and 'below' for thresholds to only look at extremes
def slicer(ticker, threshold=-0.05):
    data = returns(ticker)
    ## Filter_ret only looks for extremes
    filter_ret = data[data < threshold].dropna() if threshold < 0 else data[data > threshold].dropna() 
    #print(filter_ret)
    plt.scatter(x=filter_ret.index, y=filter_ret*100, label="Log Daily Returns")
    plt.legend(loc="lower left")
    ## Adjust graph title
    plt.title(f"Daily returns of {ticker} {"above" if threshold > 0 else "less than"} {threshold*100}%")
    plt.ylabel("% Return")
    plt.tight_layout()
    plt.show()

slicer("SPY", -0.05)