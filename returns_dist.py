## Purpose of this script is to map various stocks (and indices) to 
# different returns distributions and compare which is the most accurate via MSE
## Testing empirical returns dist against: student-t, gamma, log-normal, normal 
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def get_data(ticker):
    data = yf.download(ticker, start="2016-01-01", auto_adjust=True, interval="1d")
    return data

def get_returns(ticker):    
    values = get_data(ticker)["Close"]
    log_returns = np.log(values / values.shift())
    return log_returns.dropna()

def graph_dist(ticker):
    raw_returns = get_returns(ticker)
    abs_returns = abs(raw_returns).to_numpy().flatten()
    abs_returns = abs_returns[abs_returns > 0]
    n = len(abs_returns)

    # Fit distributions 
    gamma_shape, gamma_loc, gamma_scale = stats.gamma.fit(abs_returns, floc=0)
    norm_mu, norm_sigma = stats.norm.fit(abs_returns)
    lognorm_s, lognorm_loc, lognorm_scale = stats.lognorm.fit(abs_returns, floc=0)
    student_df, student_loc, student_scale = stats.t.fit(abs_returns, floc=0)
    

    # Empirical quantiles (sorted data)
    empirical_q = np.sort(abs_returns)

    # Probability points
    probs = (np.arange(1, n + 1) - 0.3175) / (n + 0.365)

    # Theoretical quantiles from each fitted distribution 
    gamma_q = stats.gamma.ppf(probs, gamma_shape, gamma_loc, gamma_scale)
    norm_q  = stats.norm.ppf(probs, norm_mu, norm_sigma)
    lognorm_q  = stats.lognorm.ppf(probs, lognorm_s, lognorm_loc, lognorm_scale)
    student_q  = stats.t.ppf(probs, student_df, student_loc, student_scale)

    # Q-Q residuals 
    gamma_qq_resid = empirical_q - gamma_q
    norm_qq_resid  = empirical_q - norm_q
    lognorm_qq_resid = empirical_q - lognorm_q
    student_qq_resid = empirical_q - student_q


    # PDF curves for top panels
    kde = stats.gaussian_kde(abs_returns)
    x = np.linspace(abs_returns.min(), abs_returns.max(), 500)
    gamma_pdf = stats.gamma.pdf(x, gamma_shape, gamma_loc, gamma_scale)
    norm_pdf  = stats.norm.pdf(x, norm_mu, norm_sigma)
    lognorm_pdf = stats.lognorm.pdf(x, lognorm_s, lognorm_loc, lognorm_scale)
    student_pdf = stats.t.pdf(x, student_df, student_loc, student_scale)

    # Plot 
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f"Daily Absolute Log Return Distribution Analysis: {ticker} (2016-01-01 to present)",
                 fontsize=14, fontweight="bold")

    #  Top-left: histogram + three fits 
    ax = axes[0, 0]
    ax.hist(abs_returns, bins=60, density=True, color="steelblue", alpha=0.5, label="Absolute returns")
    ax.plot(x, gamma_pdf,   color="red",   linewidth=2, label=f"Gamma  (α={gamma_shape:.2f}, β={gamma_scale:.4f})")
    ax.plot(x, norm_pdf,    color="green", linewidth=2, label=f"Normal (μ={norm_mu:.4f}, σ={norm_sigma:.4f})")
    ax.plot(x, lognorm_pdf,    color="orange", linewidth=2, label=f"Lognorm (μ={lognorm_s:.4f}, σ={lognorm_scale:.4f})")
    ax.plot(x, student_pdf,    color="purple", linewidth=2, label=f"Student t (μ={student_df:.4f}, σ={student_scale:.4f})")
    ax.plot(x, kde(x), color="black", linewidth=1.5, linestyle="--", label="KDE (empirical)")
    ax.set_title("Abs Log Return Distribution with Fitted Models")
    ax.set_xlabel("Absolute Log Return")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)

    #  Top-right: Q-Q plot — all models vs empirical 
    ax = axes[0, 1]
    ax.scatter(gamma_q, empirical_q, color="red",   s=10, alpha=0.6, label="Gamma")
    ax.scatter(norm_q,  empirical_q, color="green", s=10, alpha=0.6, label="Normal")
    ax.scatter(lognorm_q,  empirical_q, color="orange", s=10, alpha=0.6, label="Lognorm")
    ax.scatter(student_q,  empirical_q, color="purple", s=10, alpha=0.6, label="Student t")
    # Perfect-fit reference line
    lo, hi = empirical_q.min(), empirical_q.max()
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1, linestyle="--", label="Perfect fit")
    ax.set_title("Modelled v Empirical Abs Log Return")
    ax.set_xlabel("Modelled Abs Log Return")
    ax.set_ylabel("Empirical Abs Log Return")
    ax.legend(fontsize=8)

    #  Bottom-left: Gamma Q-Q residuals 
    ax = axes[1, 0]
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.fill_between(empirical_q, gamma_qq_resid, 0,
                    where=(gamma_qq_resid >= 0), color="red",  alpha=0.4, label="Empirical > Gamma")
    ax.fill_between(empirical_q, gamma_qq_resid, 0,
                    where=(gamma_qq_resid <  0), color="blue", alpha=0.4, label="Empirical < Gamma")
    ax.plot(empirical_q, gamma_qq_resid, color="red", linewidth=1.2)
    ax.set_title("Gamma Residuals")
    ax.set_xlabel("True Abs Log Return")
    ax.set_ylabel("Residual from Gamma")
    ax.legend(fontsize=8)

    # Bottom-right: Normal Q-Q residuals 
    ax = axes[1, 1]
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.fill_between(empirical_q, norm_qq_resid, 0,
                    where=(norm_qq_resid >= 0), color="green",  alpha=0.4, label="Empirical > Normal")
    ax.fill_between(empirical_q, norm_qq_resid, 0,
                    where=(norm_qq_resid <  0), color="purple", alpha=0.4, label="Empirical < Normal")
    ax.plot(empirical_q, norm_qq_resid, color="green", linewidth=1.2)
    ax.set_title("Normal Residuals")
    ax.set_xlabel("True Abs Log Return")
    ax.set_ylabel("Residual from Normal")
    ax.legend(fontsize=8)
    # Lognormal residuals 
    ax = axes[2, 0]
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.fill_between(empirical_q, lognorm_qq_resid, 0,
                    where=(lognorm_qq_resid >= 0), color="black",  alpha=0.4, label="Empirical > Lognormal")
    ax.fill_between(empirical_q, lognorm_qq_resid, 0,
                    where=(lognorm_qq_resid <  0), color="orange", alpha=0.4, label="Empirical < Logrmal")
    ax.plot(empirical_q, lognorm_qq_resid, color="green", linewidth=1.2)
    ax.set_title("Lognormal Residuals")
    ax.set_xlabel("True Abs Log Return")
    ax.set_ylabel("Residual from Lognormal")
    ax.legend(fontsize=8)
    #  Student residuals 
    ax = axes[2, 1]
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.fill_between(empirical_q, student_qq_resid, 0,
                    where=(student_qq_resid >= 0), color="green",  alpha=0.4, label="Empirical > Student t")
    ax.fill_between(empirical_q, student_qq_resid, 0,
                    where=(student_qq_resid <  0), color="purple", alpha=0.4, label="Empirical < Student t")
    ax.plot(empirical_q, student_qq_resid, color="green", linewidth=1.2)
    ax.set_title("Student t Residuals")
    ax.set_xlabel("True Abs Log Return")
    ax.set_ylabel("Residual from Student t")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()

    # Summary stats
    gamma_rmse = np.sqrt(np.mean(gamma_qq_resid**2))
    norm_rmse  = np.sqrt(np.mean(norm_qq_resid**2))
    print("=" * 50)
    print(f"Gamma fit  →  shape={gamma_shape:.4f}, loc={gamma_loc:.4f}, scale={gamma_scale:.6f}")
    print(f"Normal fit →  μ={norm_mu:.6f}, σ={norm_sigma:.6f}")
    print(f"Gamma Q-Q RMSE : {gamma_rmse:.6f}")
    print(f"Normal Q-Q RMSE: {norm_rmse:.6f}")
    print("=" * 50)

graph_dist(f"INTC")