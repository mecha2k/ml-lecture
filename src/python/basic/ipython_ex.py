# %%
from IPython.display import display, HTML
display(HTML("IPython display example..."))

# %%
import os
print(os.getcwd())

# %%
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pyfolio as pf

from datetime import datetime

plt.style.use("seaborn")
sns.set_palette("cubehelix")
plt.rcParams["figure.figsize"] = [8, 5]
plt.rcParams["figure.dpi"] = 150
warnings.simplefilter(action="ignore", category=FutureWarning)

risky_assets = ["AAPL", "IBM", "MSFT", "TWTR"]
n_assets = len(risky_assets)
start = datetime(2017, 1, 1)
end = datetime(2020, 12, 31)
data = yf.download(risky_assets, start=start, end=end, adjusted=True, progress=False)
prices_df = data["2017":"2018"]

prices_df["Adj Close"].plot(title="Stock prices of the considered assets")
plt.savefig("ch7_im1.png")

returns = prices_df["Adj Close"].pct_change().dropna()
portfolio_weights = n_assets * [1 / n_assets]
portfolio_returns = pd.Series(np.dot(portfolio_weights, returns.T), index=returns.index)

pf.create_simple_tear_sheet(portfolio_returns)
# fig = pf.create_returns_tear_sheet(portfolio_returns, return_fig=True)
# fig.savefig("ch7_im2.png", dpi=300)

# %%
