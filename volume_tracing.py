import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def get_data(ticker):
    data = yf.download(ticker, start="2025-01-01", interval="1d")
    return data

def sep_plots():
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (12,5))
    axes[0].plot(np.log(data["Volume"]), label="Log volume", color="red")
    #axes[0].plot(data["Close"], color="black", label="Close price", alpha=0.3)
    axes[1].plot(np.log(data["Close"]/data["Close"].shift(1)), label="Log returns", color="blue")
    axes[0].set_ylabel("Log volume")
    axes[1].set_ylabel("Log returns")
    axes[0].legend()
    axes[1].legend()
    plt.show()

def combined_plot():
    fig, ax1 = plt.subplots(figsize = (10,5))
    ax1.plot(np.log(data["Volume"]), label="Log volume", color="red")
    ax1.set_ylabel("Log volume", color="red")
    ax2 = ax1.twinx()
    ax2.plot(np.log(data["Close"]/data["Close"].shift(1)), label="Log returns", color="blue")
    ax2.set_ylabel("Log returns", color="blue")
    plt.title("Log Volume and Log Returns")
    plt.show()

data = get_data("AMPX")

##Correlation calculation
#print(np.log(data[('Close', 'ASTS')]/data[("Close", "ASTS")].shift()).corr(data[('Volume', 'ASTS')]))

def vol_ma():
    fig, ax1 = plt.subplots(figsize = (10,5))
    data["Volume_MA20"] = data["Volume"].rolling(window=20).mean()
    data["Volume_MA50"] = data["Volume"].rolling(window=50).mean()
    #ax1.plot(data["Volume"], label="Volume", color="gray", alpha=0.5)
    ax1.plot(data["Volume_MA20"], label="20-day MA", color="orange")
    ax2 = ax1.twinx()
    ax2.plot(data["Close"], label="Close price", color="black")
    #plt.plot(data["Volume_MA50"], label="50-day MA", color="green")
    plt.title(f"SNAP 20-day-MA Volumes and Close Price")
    ax1.set_ylabel("Volume", color="black")
    ax2.set_ylabel("Close Price", color="black")
    plt.legend()
    plt.axhline(y=7, color='r', linestyle='--', label="$7")
    plt.axhline(y=8, color='r', linestyle='--', label="$8")
    plt.show()

vol_ma()

