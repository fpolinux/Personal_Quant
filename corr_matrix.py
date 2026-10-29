## Generating a simple stocks correlation matrix. The inspo for this is to understand how the vix moves compared to real-time sp500 movement.
## Although the vix is built of sp500 futures, I am still interested to see if there is a relationship to today's sp500. 
import yfinance as yf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import datetime

##Find the stock returns
def returns(stocks):
    data = yf.download(stocks, start="2025-01-01", end=datetime.date.today(), period = "1d")["Close"]
    returns = np.log(data/data.shift()).dropna()
    return returns

##Generate correlation data
def corr_matrix(stocks):
    data = returns(stocks)
    corr_matrix = data.corr()
    return corr_matrix

##Purely graphing corr matrix
def graph_corr(stocks, annot=False):
    corr_data = corr_matrix(stocks)
    sns.heatmap(corr_data, annot=annot, cmap="crest", fmt=".2f", square=True, vmin=-1, vmax=1, annot_kws={"size": 8})
    plt.tight_layout()

    plt.show()

def average_corr(stocks):
    corr_data = corr_matrix(stocks)
    corr_df = pd.DataFrame()
    for i in corr_data.columns:
        avg_corr = corr_data[i].drop(i).mean()
        corr_df.at[i, "Average Correlation"] = avg_corr
    print(corr_df.sort_values(by="Average Correlation", ascending=False))


def sharpe(stocks, rf=0.04):
    data = returns(stocks)
    sharpe_ratios = pd.DataFrame()
    
    for i in data.columns:
        mean_return = data[i].mean() * 252
        std_dev = data[i].std() * np.sqrt(252)
        
        # Use annualized values directly
        sharpe_ratio = (mean_return - rf) / std_dev if std_dev != 0 else 0
        
        sharpe_ratios.at[i, "Sharpe Ratio"] = sharpe_ratio
    print(sharpe_ratios.sort_values(by="Sharpe Ratio", ascending=False))


stocks = ["SOFI", "SGHC", "NVDA", "INTC", "HSBC", "AMD", "GOOG", "SPY", "FTNT", "AMZN", "NVO"]
etf_list = "IBB SMH VNQ	XLB	XLE	XLF	XLI	XLP	XLU	KBE	XHB	XME	XRT	KRE	XOP	GDX	GDX XTL	XLV	XLK	XLY	OIH	IYE	IYR	ITB SPY SOYB".split()
vanguard_etfs = """VGT	VOO	VUG	VOOG	MGK	VCR	VTI	VONE	VBK	VIS	VTHR	VOT	VHT	MGC	VTWG	VIG	VDC	VXF	VBR	VAW	VOOV	VPU	VFMO	VTV	VOX	VOE	VTWV	VFQY	VFMF	VSS	VYM	MGV	VFMV	VFH	VFVA	VDE	VIOG	VONG	IVOG	ESGV	VIOO	IVOO	VTEL	MUNY	VTES	IVOV	VTEI	VTWO	VTEC	VIOV	VNQ	VPL	VIGI	VONV	VYMI	VCIT	VGK	VCSH	BSV	VPLS	VTC	VCRB	VCLT	BIV	VEXC	VTP	VGVT	VTG	VSDB	VSDM	VCRM	VGUS	VBIL	VGHY	BND	VXUS	VEU	BLV	VSGX	BNDW	EDV	VWOB	VCEB	VEA	VGIT	VGSH	VGLT	VWO	VGMS	VTEB	VTIP	VUSB	BNDX	VNQI	VMBS
""".split()
couriers = ["UPS", "FDX", "SPY", "DHLGY"]
##Random list built various sectors: cybersec, courier/transport, banking, tech, gold, property, etc. (as different asset classes)
aerospace = ["BA", "EADSY", "GE", "LUNR", "ASTS", "SPY", "RKLB", "LUNR", "AMPX"]
random = ["SOYB", "SPY", "GLD.NZ", "BTC.NZ", "GPR.NZ"]
korea = ["SPY", "EWY", "%5EKS11", "000660.KS", "005930.KS"]
sample_diversified_port = ["FTNT", "UPS", "HSBC", "SHIP", "RDDT", "SPY", "GLD.NZ", "NPF.NZ", "BTC.NZ", "LIV.NZ", "GPR.NZ"]
current_holdings = ["QAN.AX", "WBC.AX", "ACGL", "AMAT", "AMD", "AMZN", "GOOG", "GAP", "HSBC", "INTC", "IYC", "KSPI", "LLY", "NVDA", "SPY", "TMC", "XYZ", "HSBC", "AMPX", "UFO"]
shipping = ["BDRY", "SHIP", "BOAT", "FRO", "SPY", "HAFN"]

graph_corr(shipping, annot=True)
#average_corr(current_holdings)
sharpe(shipping, rf=0.041)
    




