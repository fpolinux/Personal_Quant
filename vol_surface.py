import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from scipy.interpolate import griddata

##Config
ticker = "SPY"
rf = 0.04

##Fetch data
ticker = yf.Ticker(ticker)
spot_price = ticker.fast_info['lastPrice']
expirations = ticker.options
all_options = []

print("Fetching data...")
for exp in expirations:
    opt = ticker.option_chain(exp)
    calls = opt.calls

    ##Calc TTM
    expiry_date = pd.to_datetime(exp)
    today = pd.to_datetime('today')
    ttm = (expiry_date - today).days / 365.0

    ##Optional realistic filter for liquidity and realistic IVs. IV over 1% and volume over 5 options
    calls = calls[(calls["impliedVolatility"] > 0.01) & (calls["volume"] > 5)]
    if ttm < 1.5:
        temp_df = pd.DataFrame({
            "Strike": calls["strike"],
            "TTM": ttm,
            "IV": calls["impliedVolatility"]
        })
        all_options.append(temp_df)
    else:
        pass

## Create and then tidy up overall dataframe
df = pd.concat(all_options)
df = df[df["TTM"] > 0]
df = df.reset_index(drop = True)

##Create grid with strikes and TTM
strikes = np.linspace(df["Strike"].median()*0.8, df["Strike"].median()*1.2, 50)
ttm = np.linspace(df["TTM"].median()*0.8, df["TTM"].median()*1.2, 50)
X, Y = np.meshgrid(strikes, ttm)

##Place IV on grid
Z = griddata((df["Strike"], df["TTM"]), df["IV"], (X, Y), method='cubic')

##Plot
fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale = 'Magma')])

fig.update_layout(title=f'Volatility Surface: {ticker}', scene = dict(xaxis_title='Strike Price', 
                                                                      yaxis_title='Time to Maturity (Years)', 
                                                                      zaxis_title='Implied Volatility'
                                                                      ),
                                                                      autosize=True,
                                                                      width=800, height=800)
fig.show()
