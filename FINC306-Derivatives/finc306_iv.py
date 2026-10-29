import pandas as pd
import numpy as np
from math import exp

# Load and preprocess data
excel_file = r"C:\Users\nansh\Downloads\jifti0y4i6duvqnz.xlsx"
df_raw = pd.read_excel(excel_file, sheet_name='Raw Data')

# Select and rename columns
df_relevant = df_raw[[
    'Expiration Date of the Option', 'The Date of this Price', 'C=Call, P=Put',
    'Strike Price of the Option Times 1000', 'Highest Closing Bid Across All Exchanges',
    'Lowest Closing Ask Across All Exchanges', 'Implied Volatility of the Option'
]].rename(columns={
    'Expiration Date of the Option': 'exp',
    'The Date of this Price': 'date',
    'C=Call, P=Put': 'type',
    'Strike Price of the Option Times 1000': 'strike',
    'Highest Closing Bid Across All Exchanges': 'bid',
    'Lowest Closing Ask Across All Exchanges': 'ask',
    'Implied Volatility of the Option': 'iv'
})

# Convert dates and calculate TTM (in years)
df_relevant['date'] = pd.to_datetime(df_relevant['date'])
df_relevant['exp'] = pd.to_datetime(df_relevant['exp'])
df_relevant['ttm'] = (df_relevant['exp'] - df_relevant['date']).dt.days / 365.0

# Filter TTM <= 2 years
df_relevant = df_relevant[df_relevant['ttm'] <= 2]

# Separate calls and puts
calls = df_relevant[df_relevant['type'] == 'C'].copy()
puts = df_relevant[df_relevant['type'] == 'P'].copy()

# Calculate mid prices
calls['mid'] = (calls['bid'] + calls['ask']) / 2
puts['mid'] = (puts['bid'] + puts['ask']) / 2

# Merge calls and puts on strike and expiration to align pairs
merged = pd.merge(
    calls[['exp', 'strike', 'ttm', 'mid']],
    puts[['exp', 'strike', 'ttm', 'mid']],
    on=['exp', 'strike'],
    suffixes=('_call', '_put')
)

# Calculate forward price using put-call parity (vectorized)
risk_free_rate = 0.05
merged['forward_price'] = (
    merged['strike'] / 1000 + 
    np.exp(risk_free_rate * merged['ttm_call']) * 
    (merged['mid_call'] - merged['mid_put'])byh
)

print(merged[['exp', 'strike', 'forward_price']])