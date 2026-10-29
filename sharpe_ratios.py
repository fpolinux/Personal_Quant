import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime


def get_raw_prices(stocks):
    data = yf.download(stocks, start="2024-10-06", end=datetime.date.today(), period = "1d", auto_adjust = True)["Close"]
    return data

def get_basic_stats(stocks):
    data = get_raw_prices(stocks)
    returns = np.log(data/data.shift()).dropna()
    mu = (returns.mean()*251).values
    std_dev = (returns.std()*(251**0.5)).values

    ##Sharpe_ratios
    rf = 0.04
    sharpe = (mu - rf)/std_dev
    sharpe = pd.DataFrame(sharpe, index=returns.columns, columns=["Sharpe"])
    sharpe = sharpe.sort_values(by="Sharpe", ascending=True)
    return stocks, sharpe

def corr_matrix(stocks):
    data = get_raw_prices(stocks)
    returns = np.log(data/data.shift()).dropna()
    corr = returns.corr()
    sns.heatmap(corr, annot=False, cmap="crest")
    plt.title("Correlation Matrix of Nuclear Stocks")
    plt.show()
    return corr

def cluster_by_corr(stocks):
    data = get_raw_prices(stocks)
    returns = np.log(data/data.shift()).dropna()
    corr = returns.corr()
    sns.clustermap(corr, annot=True, cmap="crest")
    plt.title("Clustered Correlation Matrix of Nuclear Stocks")
    plt.show()
    return corr
print(get_basic_stats("UPS"))
# ## Nuclear stocks sharpe ratios
# string = "MMM	AOS	ABT	ABBV	ACN	ADBE	AMD	AES	AFL	A	APD	ABNB	AKAM	ALB	ARE	ALGN	ALLE	LNT	ALL	GOOGL	GOOG	MO	AMZN	AMCR	AEE	AEP	AXP	AIG	AMT	AWK	AMP	AME	AMGN	APH	ADI	AON	APA	APO	AAPL	AMAT	APP	APTV	ACGL	ADM	ANET	AJG	AIZ	T	ATO	ADSK	ADP	AZO	AVB	AVY	AXON	BKR	BALL	BAC	BAX	BDX	BBY	TECH	BIIB	BLK	BX	XYZ	BK	BA	BKNG	BSX	BMY	AVGO	BR	BRO	BLDR	BG	BXP	CHRW	CDNS	CPT	CPB	COF	CAH	KMX	CCL	CARR	CAT	CBOE	CBRE	CDW	COR	CNC	CNP	CF	CRL	SCHW	CHTR	CVX	CMG	CB	CHD	CI	CINF	CTAS	CSCO	C	CFG	CLX	CME	CMS	KO	CTSH	COIN	CL	CMCSA	CAG	COP	ED	STZ	CEG	COO	CPRT	GLW	CPAY	CTVA	CSGP	COST	CTRA	CRWD	CCI	CSX	CMI	CVS	DHR	DRI	DDOG	DVA	DAY	DECK	DE	DELL	DAL	DVN	DXCM	FANG	DLR	DG	DLTR	D	DPZ	DASH	DOV	DOW	DHI	DTE	DUK	DD	EMN	ETN	EBAY	ECL	EIX	EW	EA	ELV	EME	EMR	ETR	EOG	EPAM	EQT	EFX	EQIX	EQR	ERIE	ESS	EL	EG	EVRG	ES	EXC	EXE	EXPE	EXPD	EXR	XOM	FFIV	FDS	FICO	FAST	FRT	FDX	FIS	FITB	FSLR	FE	FI	F	FTNT	FTV	FOXA	FOX	BEN	FCX	GRMN	IT	GE	GEHC	GEV	GEN	GNRC	GD	GIS	GM	GPC	GILD	GPN	GL	GDDY	GS	HAL	HIG	HAS	HCA	DOC	HSIC	HSY	HPE	HLT	HOLX	HD	HON	HRL	HST	HWM	HPQ	HUBB	HUM	HBAN	HII	IBM	IEX	IDXX	ITW	INCY	IR	PODD	INTC	IBKR	ICE	IFF	IP	IPG	INTU	ISRG	IVZ	INVH	IQV	IRM	JBHT	JBL	JKHY	J	JNJ	JCI	JPM	K	KVUE	KDP	KEY	KEYS	KMB	KIM	KMI	KKR	KLAC	KHC	KR	LHX	LH	LRCX	LW	LVS	LDOS	LEN	LII	LLY	LIN	LYV	LKQ	LMT	L	LOW	LULU	LYB	MTB	MPC	MAR	MMC	MLM	MAS	MA	MTCH	MKC	MCD	MCK	MDT	MRK	META	MET	MTD	MGM	MCHP	MU	MSFT	MAA	MRNA	MHK	MOH	TAP	MDLZ	MPWR	MNST	MCO	MS	MOS	MSI	MSCI	NDAQ	NTAP	NFLX	NEM	NWSA	NWS	NEE	NKE	NI	NDSN	NSC	NTRS	NOC	NCLH	NRG	NUE	NVDA	NVR	NXPI	ORLY	OXY	ODFL	OMC	ON	OKE	ORCL	OTIS	PCAR	PKG	PLTR	PANW	PSKY	PH	PAYX	PAYC	PYPL	PNR	PEP	PFE	PCG	PM	PSX	PNW	PNC	POOL	PPG	PPL	PFG	PG	PGR	PLD	PRU	PEG	PTC	PSA	PHM	PWR	QCOM	DGX	RL	RJF	RTX	O	REG	REGN	RF	RSG	RMD	RVTY	HOOD	ROK	ROL	ROP	ROST	RCL	SPGI	CRM	SBAC	SLB	STX	SRE	NOW	SHW	SPG	SWKS	SJM	SW	SNA	SOLV	SO	LUV	SWK	SBUX	STT	STLD	STE	SYK	SMCI	SYF	SNPS	SYY	TMUS	TROW	TTWO	TPR	TRGP	TGT	TEL	TDY	TER	TSLA	TXN	TPL	TXT	TMO	TJX	TKO	TTD	TSCO	TT	TDG	TRV	TRMB	TFC	TYL	TSN	USB	UBER	UDR	ULTA	UNP	UAL	UPS	URI	UNH	UHS	VLO	VTR	VLTO	VRSN	VRSK	VZ	VRTX	VTRS	VICI	V	VST	VMC	WRB	GWW	WAB	WMT	DIS	WBD	WM	WAT	WEC	WFC	WELL	WST	WDC	WY	WSM	WMB	WTW	WDAY	WYNN	XEL	XYL	YUM	ZBRA	ZBH	ZTS"
# list = string.split()
# #corr_matrix(list)
# list_2 = ["SOYB", "SPY"]
# print(corr_matrix("AMPX"))
# #list = ["CEG", "OKLO", "CCJ", "LEU", "BWXT", "SMR", "DNN", "PEG", "NXE", "UEC", "PCG", "UUUU", "PDN.AX", "NNE", "SLX.AX", "DYL.AX", "EU", "BOE.AX"]
# print(get_basic_stats("AMPX"))

# #print(cluster_by_corr(list))


#x,y = get_basic_stats("NVDL")

#print(x, y)


