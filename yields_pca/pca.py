# The aim of this project is to develop and understanding and apply principal component analysis (PCA)
# The idea is to use as few packages as possible to understand the mathematics behind it
# and the selected dataset is the US Fed Reserve Data, found here: https://www.federalreserve.gov/data/nominal-yield-curve.htm

#%% Imports of packages and data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
data = pd.read_csv(os.path.join(os.getcwd(), "feds200628.csv"), skiprows=9)
data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d")
data = data.set_index("Date").sort_index().dropna()
data.head()

#%% Basic yield plotting
# Remember that for plotting the yield curve, I want multiple yields at one date
# Pick date(s) to view yield curve from. Filter out non-trading days
# Select yield columns from 1-30 and index this as maturities list
mat_cols = [f"SVENPY{m:02d}" for m in range(1,31)]
maturities = list(range(1, 31))

def plot_curve(dates, title, xlabel, ylabel):
    not_trading = [d for d in dates if pd.Timestamp(d) not in data.index]
    trading = [d for d in dates if pd.Timestamp(d) in data.index]

    fig, ax = plt.subplots(figsize=(10,6))
    for d in trading:
        ax.plot(maturities, data.loc[d, mat_cols], marker="o", label=f"{d}", alpha=0.7)
    ax.set_xlabel(f"{xlabel}")
    ax.set_ylabel(f"{ylabel}")
    ax.set_title(f"{title}")
    ax.legend(loc="right")
    ax.grid()
    fig.tight_layout()
    plt.show()
    if len(not_trading) > 0:
        print(f"These dates were not trading days and were skipped:\n{not_trading}")


# %% Yield plot
# Sample dates are latest, lehman collapse, covid fear, normal day in 2021, iran war
dates = [data.index[-1], "2026-05-15", "2008-09-15", "2020-03-09", "2021-12-31"]
plot_curve(dates, 
           "US Treasury Yield Curve for selected dates",
           xlabel="Maturity (years)",
           ylabel="Yield (%)")

# %% Run PCA
# To run a PCA, the general steps are to:
# 1. Standardise
# 2. Find the covariance matrix
# 3. Perform eigenvalue/vector decomp
# 4. Rank and select components
# 5. Project eigenvecors and output

#Standardise data ! Really important note to make sure to de-mean data
subset = data[mat_cols].dropna() # dropna done above, but to make sure
mu = subset.mean(axis=0)
std = subset.std(axis=0)
#z = (subset - mu)/std # returns n-data rows by m columns
z = subset - mu #not using std normalisation since yields all in the same units
#Covariance matrix
C = (1/(len(subset)-1) * z.T @ z) # square matrix
# Eigenvalue decomp
# In theory this is using the det(C-lambda*I) = 0 setup. 
# Due to the large size of the matrix, the resultant polynomial will be > **2 and so,
#  solving is typically numerical. Hence, use numpy instead :D
eigenvalues, eigenvectors = np.linalg.eigh(C)

# Ranking components; note use transpose since linalg outputs eigenvectors as columns and sorting uses rows
# (optional) also make eigenvectors positive for later visualisation (namely PC1) 
eigen_set = sorted(zip(eigenvalues, eigenvectors.T), reverse=True) 
ranked_vectors = [eigen_set[0][1], eigen_set[1][1], eigen_set[2][1]]

# Projection and finalise
proj_matrix = np.column_stack(ranked_vectors) #smushes together colums in to an mx3 matrix
pca_matrix = np.dot(z, proj_matrix) 
pc_df = pd.DataFrame(pca_matrix, index=z.index, columns=[f"PC{i}" for i in range(1,4)])
fig, ax = plt.subplots(figsize=(12,6))
pc_df.plot(ax=ax)
plt.title("PC trajectory")
plt.xlabel("Year")
plt.ylabel("Component value")
plt.grid()
plt.show()
#%% Performance testing
#Numerical performance via explained variance
exp_var = pd.DataFrame([eigen_set[i][0]/eigenvalues.sum() for i in range(3)], columns = ["Explained variance ratio"], index=pc_df.columns)
fig, ax = plt.subplots(figsize = (12,6))
bars = ax.bar(x=exp_var.index, height=exp_var.values.flatten())
ax.bar_label(bars, padding=3, fmt=lambda x: f"{x*100:.1f}%")
plt.title("Explained Variance by Principal Component")
plt.ylabel("Explained Variance")
plt.ylim(0, exp_var["Explained variance ratio"].max()*1.15)  # pad labels
plt.show()

