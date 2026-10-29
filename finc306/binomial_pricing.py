import numpy as np
import datetime as date
from datetime import timedelta
import matplotlib.pyplot as plt
import yfinance as yf

class OptionPrice:
    def __init__(self, S, K, r, sigma, delta, tau, N, model="crr", show_nodes=False):
        self.S = S          # Current stock price
        self.K = K          # Strike price
        self.r = r          # Risk-free interest rate
        self.sigma = sigma  # Volatility of the underlying stock
        self.delta = delta  # Dividend yield
        self.tau = tau      # Time to expiration in years
        self.N = N          # Number of time steps in the binomial model
        self.model = model  # Model type: "crr" for Cox-Ross-Rubinstein, "jr" for Jarrow-Rudd, "fwd" for Forward price tree
        self.show_nodes = show_nodes  # Whether to display the full binomial tree of option prices
        if self.show_nodes == True and self.N > 10:
            raise ValueError("A large number of steps will produce a large output. Set show_nodes to False or reduce N < 10.")

    
    def model_params(self):
        # Calculate parameters
        S, K, r, sigma, delta, tau, N = self.S, self.K, self.r, self.sigma, self.delta, self.tau, self.N
        dt = tau / N
        if self.model == "crr":
            u, d = np.exp(sigma * np.sqrt(dt)), np.exp(-sigma * np.sqrt(dt))

            return u,d
        elif self.model == "jr":
            u,d = np.exp((r-delta-0.5*sigma**2)*dt+ sigma * np.sqrt(dt)), np.exp((r-delta-0.5*sigma**2)*dt - sigma * np.sqrt(dt))
            return u,d
        elif self.model == "fwd":
            u,d = np.exp((r-delta)*dt + sigma * np.sqrt(dt)), np.exp((r-delta)*dt - sigma * np.sqrt(dt))
            return u,d
        else:
            raise ValueError("Choose 'crr', 'jr', or 'fwd'")

    def ec(self):
        # Calculate parameters
        S, K, r, sigma, delta, tau, N, show_nodes = self.S, self.K, self.r, self.sigma, self.delta, self.tau, self.N, self.show_nodes
        dt = tau / N
        u, d = self.model_params()
        p = (np.exp((r-delta)*dt) - d) / (u-d)

        ##Initialise asset prices, nodes are set up as (d=j,u=i) (d is rows, u is columns)
        asset_prices = np.zeros((N+1, N+1))
        for i in range(N+1):
            for j in range(i+1):
                asset_prices[j, i] = S * (u**(i-j)) * (d**j)

        # Array of 0s to hold option values at each node with last column (N) being max(S_T-K,0)
        option_values = np.zeros((N+1, N+1))
        ##Option price at maturity is max(0, S_T-K)
        option_values[:, N] = np.maximum(0, asset_prices[:, N] - K)

        # To get option price at t=0
        # Start from second last column and move backwards
        for i in range(N-1, -1, -1):
            for j in range(i+1):
                # Node (j,i) is represented by u^(i-j) * d^j
                # So option value at node (j,i) is the jth row and ith column
                #Start from bottom of column and move up
                option_values[j, i] = np.exp(-r*dt)*(p*option_values[j, i+1] + (1-p)*option_values[j+1, i+1])
        if show_nodes:
            print(f"European call nodes:\n {option_values}")
            return option_values[0,0]
        else: 
            return option_values[0,0]
    def ep(self):
        # Calculate parameters
        S, K, r, sigma, delta, tau, N, show_nodes = self.S, self.K, self.r, self.sigma, self.delta, self.tau, self.N, self.show_nodes
        dt = tau / N
        u, d = self.model_params()
        p = (np.exp((r-delta)*dt) - d) / (u-d)

        ##Initialise asset prices
        asset_prices = np.zeros((N+1, N+1))
        for i in range(N+1):
            for j in range(i+1):
                asset_prices[j, i] = S * (u**(i-j)) * (d**j)

        ##Initialise option values at maturity
        option_values = np.zeros((N+1, N+1))
        option_values[:, N] = np.maximum(0, K- asset_prices[:, N])

        ## To get option price at t=0
        for i in range(N-1, -1, -1):
            for j in range(i+1):
                option_values[j, i] = np.exp(-r*dt)*(p*option_values[j, i+1] + (1-p)*option_values[j+1, i+1])
        
        if show_nodes:
            print(f"European put nodes:\n {option_values}")
            return option_values[0,0]
        else: 
            return option_values[0,0]   

    def ac(self):
        S, K, r, sigma, delta, tau, N, show_nodes = self.S, self.K, self.r, self.sigma, self.delta, self.tau, self.N, self.show_nodes
        # Calculate parameters
        dt = tau / N
        u, d = self.model_params()
        p = (np.exp((r-delta)*dt) - d) / (u-d)

        ##Initialise asset prices
        asset_prices = np.zeros((N+1, N+1))
        for i in range(N+1):
            for j in range(i+1):
                # Set asset price at node (j,i) based off up and down factors
                asset_prices[j, i] = S * (u**(i-j)) * (d**j)

        ##Initialise option values at maturity
        option_values = np.zeros((N+1, N+1))
        ##Slice to get all rows in last column
        option_values[:, N] = np.maximum(0, asset_prices[:, N] - K)

        ## To get option price at t=0
        for i in range(N-1, -1, -1):
            for j in range(i+1):
                ##Let option_values[j,i] be the the max of european call price or early exercise value at node (j,i)
                option_values[j,i] = max(np.exp(-r*dt)*(p*option_values[j, i+1] + (1-p)*option_values[j+1, i+1]), asset_prices[j,i]-K)
        if show_nodes:
            print(f"American call nodes:\n {option_values}")
            return option_values[0,0]
        else: 
            return option_values[0,0]
    def ap(self):
        S, K, r, sigma, delta, tau, N = self.S, self.K, self.r, self.sigma, self.delta, self.tau, self.N
        # Calculate parameters
        dt = tau / N
        u, d = self.model_params()
        p = (np.exp((r-delta)*dt) - d) / (u-d)

        ##Initialise asset prices
        asset_prices = np.zeros((N+1, N+1))
        for i in range(N+1):
            for j in range(i+1):
                # Set asset price at node (j,i) based off up and down factors
                asset_prices[j, i] = S * (u**(i-j)) * (d**j)

        ##Initialise option values at maturity
        option_values = np.zeros((N+1, N+1))
        ##Slice to get all rows in last column
        option_values[:, N] = np.maximum(0, K - asset_prices[:, N])

        ## To get option price at t=0
        for i in range(N-1, -1, -1):
            for j in range(i+1):
                ##Let option_values[j,i] be the the max of american put price or early exercise value at node (j,i)
                option_values[j,i] = max(np.exp(-r*dt)*(p*option_values[j, i+1] + (1-p)*option_values[j+1, i+1]), K-asset_prices[j,i])
        if show_nodes:
            print(f"American put nodes:\n {option_values}")
            return option_values[0,0]
        else: 
            return option_values[0,0]
    def step_plot(self):
        # Get params
        S, K, r, delta, tau, model, sigma, N = self.S, self.K, self.r, self.delta, self.tau, self.model, self.sigma, self.N
        price_list = []
        for n in range(1, N, 1):
            option = OptionPrice(S, K, r, sigma, delta, tau, n, model)
            price_list.append([n, round(option.ac(),4)])
        x,y = zip(*price_list)
        plt.plot(x, y, marker='o')
        plt.xlabel("Number of steps")
        plt.ylabel("American Call Price")
        plt.title("American Call Price vs Steps in Binomial Model")
        plt.plot
        plt.grid()
        plt.show()

# ---- Variable input -----

# Alternative way to set time to expiration for 'real data'
expiry_date = date.datetime(2026, 1, 16) + date.timedelta(days=1)
today = date.datetime.today()
t = (expiry_date - today).days / 365
rf_ticker = yf.Ticker("%5EIRX")  # Risk-free interest rate   
market_rf = rf_ticker.info["previousClose"]/100

spot =  41 # Current stock price
strike =  40 # Strike price
r = 0.08 # Risk-free interest rates
sigma = 0.3   # Volatility of the underlying stock  
delta = 0.1      # Continuous dividend yield
tau =  1    # Time to expiration in years
steps = 3   # Number of time steps in the binomial model (400 gets close to BSM)
model = "fwd"  # (Optional) Model type: "crr" for Cox-Ross-Rubinstein, "jr" for Jarrow-Rudd, "fwd" for Forward price tree
show_nodes = True  # (Optional) Set to True to display the full binomial tree of option prices

## Call method via: option.ec(), option.ep(), option.ac(), option.ap(), option.step_plot() (to see steps convergence)
option = OptionPrice(spot, strike, r, sigma, delta, tau, steps, model, show_nodes)
print(option.ac())