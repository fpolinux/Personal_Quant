import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from arch import arch_model
from datetime import datetime, timedelta

def download_stock_data(ticker, start_date, end_date):
    """Download historical stock data from Yahoo Finance"""
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    return data['Close']

def calculate_daily_volatility(prices, window=21):
    """Calculate rolling historical volatility"""
    returns = np.log(prices/prices.shift(1))
    historical_vol = returns.rolling(window=window).std()
    return historical_vol.dropna()

def estimate_garch_volatility(returns):
    """Estimate GARCH(1,1) volatility (returns should be in decimal form)"""
    model = arch_model(returns, vol='Garch', p=1, q=1, dist='normal')
    results = model.fit(disp='off', show_warning=False)
    return results.conditional_volatility, results

def plot_volatility_comparison(prices, historical_vol, garch_vol, ticker):
    """Plot volatility comparison (both in daily terms)"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Plot price series
    ax1.plot(prices, label='Price', color='blue')
    ax1.set_title(f'{ticker} Price and Volatility Comparison (Daily)')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    # Plot volatility measures
    ax2.plot(historical_vol, label='Historical Volatility (21-day)', color='green', alpha=0.7)
    ax2.plot(garch_vol, label='GARCH(1,1) Volatility', color='red', alpha=0.7)
    ax2.set_ylabel('Daily Volatility')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Parameters
    ticker = "GOOGL"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3*365)  # 3 years of data
    
    try:
        # Download data
        print(f"Downloading data for {ticker}...")
        prices = download_stock_data(ticker, start_date, end_date)
        
        if prices.empty:
            raise ValueError("No data returned for the given ticker and date range")
            
        # Calculate simple historical volatility (daily)
        historical_vol = calculate_daily_volatility(prices)
        
        # Calculate returns for GARCH model (in decimal form)
        returns = prices.pct_change().dropna()
        
        if len(returns) < 100:
            raise ValueError("Insufficient data points for GARCH estimation")
            
        # Estimate GARCH volatility (daily)
        print("Estimating GARCH(1,1) model")
        garch_vol, garch_results = estimate_garch_volatility(returns)
        
        # Print model summary
        print("\nGARCH(1,1) Model Summary:")
        print(garch_results.summary())
        
        # Plot comparison (both in daily terms)
        plot_volatility_comparison(prices, historical_vol, garch_vol, ticker)
        
        # Forecast future volatility (next 5 days, daily)
        print("\nVolatility Forecast (Daily):")
        forecasts = garch_results.forecast(horizon=5)
        print(pd.DataFrame({
            'Day': range(1, 6),
            'Daily Volatility': np.sqrt(forecasts.variance.values[-1].flatten())
        }))
        
        # Optionally show annualized numbers for comparison
        print("\nAnnualized Volatilities:")
        print(f"Current annualised historical vol: {historical_vol.iloc[-1]*np.sqrt(252)}")
        print(f"Current annualised GARCH vol: {garch_vol.iloc[-1]*np.sqrt(252)}")
        print(f"3-day annualised GARCH forecast: {np.sqrt(forecasts.variance.values[-1][2])*np.sqrt(252)}")
        
    except Exception as e:
        print(f"Error: {str(e)}")