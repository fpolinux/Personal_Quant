##This program is supposed to be a simple implementation of the black-scholes pricing model as found in Hull's book on options trading.
import numpy as np
import math
from scipy.stats import norm  
import datetime
import yfinance as yf

class BlackScholes:
    def __init__(self, S, K, T, r, div=0,  sigma=0, price=0):
        self.S = S      # Current stock price
        self.K = K      # Strike price
        self.T = T      # Time to expiration in years
        self.r = r      # Risk-free interest rate
        self.sigma = sigma  # Volatility of the underlying stock
        self.price = price
        self.d = div

    def d1(self):
        return (np.log(self.S / self.K) + (self.r - self.d + 0.5 * (self.sigma ** 2)) * self.T) / (self.sigma * np.sqrt(self.T))

    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.T)

    def call_price(self):
        d1 = self.d1()
        d2 = self.d2()
        call_price = (self.S * np.exp(-self.d * self.T) * norm.cdf(d1)) - (self.K * np.exp(-self.r * self.T) * norm.cdf(d2))
        return call_price

    def put_price(self):
        d1 = self.d1()
        d2 = self.d2()
        put_price = (self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)) - (self.S * norm.cdf(-d1))
        return put_price
    def iv(self, tol=1e-5, max_iterations=100):
        # Find IV via Newton-Raphson 
        if self.price <= 0:
            raise ValueError("Option price must be positive to calculate implied volatility.")
        
        sigma = self.sigma  # Initial guess using input
        for i in range(max_iterations):
            self.sigma = sigma
            price = self.call_price()  # For a call option
            vega = (self.S * norm.pdf(self.d1()) * np.sqrt(self.T))  # Derivative of price with respect to sigma
            
            price_diff = price - self.price
            
            if abs(price_diff) < tol:
                return sigma
            
            sigma -= price_diff / vega  # Newton-Raphson step
            
        raise ValueError("Implied volatility not converged.")
    
## Finding the historical volatility of the stock
def vol(stock):
    start_date = datetime.datetime.today() - datetime.timedelta(days=(t*365))
    data = yf.download(stock, start=start_date, end=datetime.datetime.today(), auto_adjust=True)
    returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
    volatility = returns.std() * np.sqrt(252)  # Annualized volatility
    return volatility

def rf_rate(t):
    rf_ticker = yf.Ticker("^IRX")  # Risk-free interest rate   
    market_rf = rf_ticker.info["previousClose"]/100
    return market_rf


#print(black_scholes_call(86.29, 90 , t, three_month_rf(), 0.3905))  
t =  (datetime.datetime(2026, 1, 16) - datetime.datetime.today()).days / 365  # Time to expiration in years

print(f"The IV of the european call is: {BlackScholes(S=160,
                   K=120,
                   T=1,
                   r=0.05,
                   div=0.07,
                   sigma=0.5,
                   price=36.126).iv()}")

S = 40
K_base = 10
T = 1
r = 0.05
div = 0.07
sigma = 0.5


#v_t = 3*BlackScholes(S, K_base, T, r, div, sigma).call_price()-3*BlackScholes(S, K_base*2, T, r, div, sigma).call_price() + BlackScholes(S, K_base*3, T, r, div, sigma).call_price()- BlackScholes(S, K_base*4, T, r, div, sigma).call_price() + 3*BlackScholes(S, K_base*5, T, r, div, sigma).call_price() - 2*BlackScholes(S, K_base*6, T, r, div, sigma).call_price() + 10*np.exp(-r*T)- 40*np.exp(-div*T)
#print(f"The price of v_t = ${v_t}")