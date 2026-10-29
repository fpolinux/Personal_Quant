import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import sys
from datetime import date
# Load chosen stock yaml
def load_config(file_path):
    print(f"Usage: python monte_carlo_generic.py {file_path}")
    with open (file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_company_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info

    # Relevant fields
    revenue = info.get("totalRevenue", None) 
    ebitda = info.get("ebitda", None) 
    net_debt = info.get("totalDebt", 0) - info.get("cash", 0) 
    shares_outstanding = info.get("sharesOutstanding", None)
    company_name = info.get("longName", ticker)
    
    ##Return data as a dictionary
    return {
        "ticker": ticker,
        "company_name": company_name,
        "revenue": revenue,
        "ebitda": ebitda,
        "net_debt": net_debt,
        "shares": shares_outstanding
    }

def monte_carlo_dcf(base_revenue, shares, net_debt, n=20000):
    # Load assumptions from yaml config
    years = cfg["model"]["horizon_years"]
    simulated_prices = []
    growth_mean = cfg["model"]["revenue"]["growth"]["mean"]
    growth_sd = cfg["model"]["revenue"]["growth"]["sd"]
    ebit_low = cfg["model"]["ebit_margin"]["low"]
    ebit_mode = cfg["model"]["ebit_margin"]["mode"]
    ebit_high = cfg["model"]["ebit_margin"]["high"]
    capex_low = cfg["model"]["capex"]["low"]
    capex_high = cfg["model"]["capex"]["high"]
    wacc_mean = cfg["model"]["wacc"]["mean"]
    wacc_sd = cfg["model"]["wacc"]["sd"]
    term_g_mean = cfg["model"]["terminal_growth"]["mean"]
    term_g_sd = cfg["model"]["terminal_growth"]["sd"]

    for i in range(n):
        rev = base_revenue
        growth = np.random.normal(growth_mean, growth_sd, years)  # x% mean growth and standard deviation
        ebit_margin = np.random.triangular(ebit_low, ebit_mode, ebit_high)
        capex_pct = np.random.uniform(capex_low, capex_high)
        wacc = np.random.normal(wacc_mean, wacc_sd)
        term_g = np.random.normal(term_g_mean, term_g_sd)

        fcf_list = []
        fcf_undiscounted = []
        for y in range(years):
            rev *= (1 + growth[y])
            ebit = rev * ebit_margin
            tax = ebit * cfg["model"]["tax_rate"]["value"]
            dep = rev * cfg["model"]["depreciation"]["value"]
            capex = rev * capex_pct
            fcf = ebit - tax + dep - capex
            fcf_undiscounted.append(fcf)
            fcf_list.append(fcf / (1 + wacc)**(y + 1))

        # Terminal Value
        terminal_fcf = fcf_undiscounted[-1] * (1 + term_g)
        terminal_val = terminal_fcf / (wacc - term_g)
        terminal_val_discounted = terminal_val / (1 + wacc)**years

        enterprise_value = sum(fcf_list) + terminal_val_discounted
        equity_value = enterprise_value - net_debt
        price_per_share = equity_value / shares
        simulated_prices.append(price_per_share)
    return simulated_prices

cfg = load_config(sys.argv[1])
data = get_company_data(cfg['ticker'])

# Main program
if None in [data['revenue'], data['shares']]:
    print(f"Could not retrieve enough financial data for {ticker}.")
else:
    simulated_prices = monte_carlo_dcf(
        base_revenue=data['revenue'],
        shares=data['shares'],
        net_debt=data['net_debt'],
        n = cfg['model']['simulation_runs']
    )

    # Output Summary
    print(f"\nMonte Carlo Valuation for {data['company_name']} ({cfg['ticker']})")
    print(f"Median Share Price: ${np.median(simulated_prices):.2f}")
    print(f"10th–90th Percentile Range: ${np.percentile(simulated_prices, [10,90])[0]:.2f} – ${np.percentile(simulated_prices, [10,90])[1]:.2f}")
    
    # Plot Distribution
    plt.hist(simulated_prices, bins=50, alpha=0.7)
    plt.title(f"Monte Carlo Valuation Distribution: {cfg['ticker']} (as of {date.today()})")
    plt.xlabel("Implied Share Price")
    plt.ylabel("Frequency")
    plt.plot([np.median(simulated_prices), np.median(simulated_prices)], [0, 100], color='red', linestyle='dashed', linewidth=1, label=f'Median (${np.median(simulated_prices):.2f})')
    plt.plot([np.percentile(simulated_prices, 10), np.percentile(simulated_prices, 10)], [0, 100], color='green', linestyle='dashed', linewidth=1, label=f'10th Percentile (${np.percentile(simulated_prices, 10):.2f})')
    plt.plot([np.percentile(simulated_prices, 90), np.percentile(simulated_prices, 90)], [0, 100], color='blue', linestyle='dashed', linewidth=1, label=f'90th Percentile (${np.percentile(simulated_prices, 90):.2f})')
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.show()
