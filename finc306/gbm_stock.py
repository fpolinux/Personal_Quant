import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm, poisson

def get_stock_data(ticker, period="3y"):
    """Get historical stock data and calculate returns"""
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    
    if hist.empty:
        raise ValueError(f"No data found for ticker {ticker}")
    
    # Calculate daily returns
    hist['Daily Return'] = np.log(hist['Close']/hist['Close'].shift(1))
    daily_returns = hist['Daily Return'].dropna()
    
    # Annualized statistics
    trading_days_per_year = 252
    annual_return = daily_returns.mean() * trading_days_per_year
    annual_volatility = daily_returns.std() * np.sqrt(trading_days_per_year)
    current_price = hist['Close'].iloc[-1]
    
    return current_price, annual_return, annual_volatility, hist

def jump_diffusion_model(ticker, T=1, n_steps=252, n_simulations=1000, 
                        jump_intensity=0.5, jump_mean=0.0, jump_volatility=0.3,
                        include_jumps=True):
    """
    Jump Diffusion Model combining:
    - Normal shocks (continuous GBM)
    - Poisson jumps (discontinuous events)
    """
    
    # Get historical data
    s0, mu, sigma, hist_data = get_stock_data(ticker)
    
    dt = T / n_steps
    times = np.linspace(0, T, n_steps + 1)
    prices = np.zeros((n_simulations, n_steps + 1))
    prices[:, 0] = s0
    
    # Arrays to track jumps
    jump_occurrences = np.zeros((n_simulations, n_steps))
    jump_sizes = np.zeros((n_simulations, n_steps))
    
    for i in range(n_simulations):
        current_price = s0
        
        for t in range(1, n_steps + 1):
            # Standard GBM component (normal shocks)
            normal_shock = np.random.normal(0, 1)
            price_change = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * normal_shock
            
            # Jump component (Poisson process)
            jump_component = 0
            jump_occurred = 0
            jump_size = 0
            
            if include_jumps:
                # Number of jumps in this time interval (Poisson distribution)
                num_jumps = np.random.poisson(jump_intensity * dt)
                
                if num_jumps > 0:
                    jump_occurred = 1
                    # Jump size: log-normal distribution
                    for _ in range(num_jumps):
                        jump_size_component = jump_mean + jump_volatility * np.random.normal(0, 1)
                        jump_component += jump_size_component
                    
                    jump_sizes[i, t-1] = jump_component
                    jump_occurrences[i, t-1] = num_jumps
            
            # Combine both components
            total_change = price_change + jump_component
            current_price = current_price * np.exp(total_change)
            prices[i, t] = current_price
    
    return prices, times, jump_occurrences, jump_sizes

def plot_jump_diffusion_results(ticker, prices, times, jump_occurrences, jump_sizes, 
                               jump_intensity, jump_mean, jump_volatility):
    """Plot comprehensive results showing both normal shocks and jumps"""
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # Plot 1: Price paths with jumps highlighted
    n_paths_to_plot = min(20, len(prices))
    for i in range(n_paths_to_plot):
        # Find where jumps occurred in this path
        jump_indices = np.where(jump_occurrences[i] > 0)[0]
        
        # Plot the path
        line = axes[0, 0].plot(times, prices[i], alpha=0.6, linewidth=1)
        color = line[0].get_color()
        
        # Highlight jump points
        if len(jump_indices) > 0:
            jump_times = times[jump_indices + 1]  # +1 because jumps affect next price
            jump_prices = prices[i, jump_indices + 1]
            axes[0, 0].scatter(jump_times, jump_prices, color='red', s=30, zorder=5)
    
    axes[0, 0].set_title(f'Jump Diffusion Simulation for {ticker}\n'
                        f'Red dots indicate jump occurrences')
    axes[0, 0].set_xlabel('Time (years)')
    axes[0, 0].set_ylabel('Price ($)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Distribution of number of jumps per path
    total_jumps_per_path = np.sum(jump_occurrences, axis=1)
    axes[0, 1].hist(total_jumps_per_path, bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[0, 1].set_title(f'Distribution of Total Jumps per Path\n'
                        f'λ = {jump_intensity} (jumps per year)')
    axes[0, 1].set_xlabel('Number of Jumps')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Distribution of jump sizes
    all_jump_sizes = jump_sizes[jump_sizes != 0]
    if len(all_jump_sizes) > 0:
        axes[1, 0].hist(all_jump_sizes, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[1, 0].axvline(x=jump_mean, color='red', linestyle='--', label=f'Mean: {jump_mean:.3f}')
        axes[1, 0].set_title(f'Distribution of Jump Sizes\n'
                           f'Mean: {jump_mean}, Vol: {jump_volatility}')
        axes[1, 0].set_xlabel('Jump Size (log return)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Final price distribution comparison
    final_prices = prices[:, -1]
    axes[1, 1].hist(final_prices, bins=50, alpha=0.7, color='blue', edgecolor='black', density=True)
    axes[1, 1].axvline(x=np.mean(final_prices), color='red', linestyle='--', 
                      label=f'Mean: ${np.mean(final_prices):.2f}')
    axes[1, 1].set_title('Final Price Distribution')
    axes[1, 1].set_xlabel('Final Price ($)')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return final_prices

def compare_gbm_vs_jump_diffusion(ticker, T=1, n_steps=252, n_simulations=1000):
    """Compare standard GBM vs Jump Diffusion model"""
    
    s0, mu, sigma, hist_data = get_stock_data(ticker)
    
    # Standard GBM (no jumps)
    dt = T / n_steps
    times = np.linspace(0, T, n_steps + 1)
    
    # GBM simulation
    gbm_prices = np.zeros((n_simulations, n_steps + 1))
    gbm_prices[:, 0] = s0
    
    for i in range(n_simulations):
        for t in range(1, n_steps + 1):
            shock = np.random.normal(0, 1)
            gbm_prices[i, t] = gbm_prices[i, t-1] * np.exp(
                (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shock
            )
    
    # Jump Diffusion simulation
    jump_prices, times, jump_occurrences, jump_sizes = jump_diffusion_model(
        ticker, T, n_steps, n_simulations, include_jumps=True
    )
    
    # Comparison plot
    plt.figure(figsize=(10, 10))
    
    # Price paths comparison
    plt.subplot(2, 2, 1)
    for i in range(min(10, n_simulations)):
        plt.plot(times, gbm_prices[i], alpha=0.5, color='blue', label='GBM' if i == 0 else "")
        plt.plot(times, jump_prices[i], alpha=0.5, color='red', label='Jump Diffusion' if i == 0 else "")
    plt.title('GBM vs Jump Diffusion: Sample Paths')
    plt.xlabel('Time (years)')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Final price distribution comparison
    plt.subplot(2, 2, 2)
    plt.hist(gbm_prices[:, -1], bins=50, alpha=0.5, color='blue', label='GBM', density=True)
    plt.hist(jump_prices[:, -1], bins=50, alpha=0.5, color='red', label='Jump Diffusion', density=True)
    plt.axvline(s0, color='black', linestyle='--', label='Initial Price')
    plt.title('Final Price Distribution Comparison')
    plt.xlabel('Final Price ($)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Tail comparison (log scale)
    plt.subplot(2, 2, 3)
    gbm_final = gbm_prices[:, -1]
    jump_final = jump_prices[:, -1]
    
    plt.hist(gbm_final, bins=50, alpha=0.5, color='blue', label='GBM', density=True, log=True)
    plt.hist(jump_final, bins=50, alpha=0.5, color='red', label='Jump Diffusion', density=True, log=True)
    plt.title('Tail Behavior (Log Scale)')
    plt.xlabel('Final Price ($)')
    plt.ylabel('Log Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Kurtosis comparison
    from scipy.stats import kurtosis
    gbm_kurtosis = kurtosis(gbm_final)
    jump_kurtosis = kurtosis(jump_final)
    
    plt.subplot(2, 2, 4)
    plt.bar(['GBM', 'Jump Diffusion'], [gbm_kurtosis, jump_kurtosis], 
            color=['blue', 'red'], alpha=0.7)
    plt.title(f'Kurtosis Comparison\nGBM: {gbm_kurtosis:.2f}, Jump: {jump_kurtosis:.2f}')
    plt.ylabel('Excess Kurtosis')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("\n" + "="*60)
    print("MODEL COMPARISON STATISTICS")
    print("="*60)
    print(f"Initial Price: ${s0:.2f}")
    print(f"GBM Final Price Stats:")
    print(f"  Mean: ${np.mean(gbm_final):.2f}, Std: ${np.std(gbm_final):.2f}")
    print(f"  Kurtosis: {gbm_kurtosis:.2f}")
    print(f"Jump Diffusion Final Price Stats:")
    print(f"  Mean: ${np.mean(jump_final):.2f}, Std: ${np.std(jump_final):.2f}")
    print(f"  Kurtosis: {jump_kurtosis:.2f}")
    print(f"  Total jumps simulated: {np.sum(jump_occurrences)}")
    print(f"  Average jumps per path: {np.mean(np.sum(jump_occurrences, axis=1)):.2f}")

# Example usage
if __name__ == "__main__":
    # Single stock analysis with jump diffusion
    ticker = 'SNDK'

    print("JUMP DIFFUSION MODEL ANALYSIS")
    print("="*50)
    
    # Run jump diffusion model
    prices, times, jump_occurrences, jump_sizes = jump_diffusion_model(
        ticker=ticker,
        T=0.5,                    # x year forecast
        n_steps=252,           # Daily steps
        n_simulations=1000,     # Number of paths
        jump_intensity=0.5,     # Average 0.5 jumps per year
        jump_mean=0.0,         # Average jump size (0% mean return)
        jump_volatility=0.15,   # Jump volatility (15%)
        include_jumps=True
    )
    
    final_prices = plot_jump_diffusion_results(
        ticker, prices, times, jump_occurrences, jump_sizes,
        jump_intensity=0.5, jump_mean=0.0, jump_volatility=0.15
    )
    
    # Compare with standard GBM
    compare_gbm_vs_jump_diffusion(ticker)