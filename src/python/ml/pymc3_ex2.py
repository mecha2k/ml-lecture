import numpy as np
import pandas as pd
import pandas_datareader.data as web

import seaborn as sns
import pymc3 as pm
from pymc3.plots import forestplot, plot_posterior, traceplot
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import gridspec

if __name__ == "__main__":
    benchmark = web.DataReader("SP500", data_source="fred", start=2010)
    benchmark.columns = ["benchmark"]

    with pd.HDFStore("data/assets.h5") as store:
        stock = store["quandl/wiki/prices"].adj_close.unstack()["AMZN"].to_frame("stock")
    data = stock.join(benchmark).pct_change().dropna().loc["2010":]
    data.info()

    ## Modeling the Sharpe Ratio
    # To model the Sharpe ratio as a probabilistic model, we need to priors about the distribution of returns and the
    # parameters that govern this distribution. The student t distribution exhibits fat tails relative to the normal
    # distribution for low degrees of freedom (df) and is a reasonable choice to capture this aspect of returns.

    ### Define Probability Model
    # Hence, we need to model the three parameters of this distribution, namely the mean and standard deviation of
    # returns, and the degrees of freedom. We’ll assume normal and uniform distributions for the mean and the standard
    # deviation, respectively, and an exponential distribution for the df with a sufficiently low expected value to
    # ensure fat tails. Returns are based on these probabilistic inputs, and the annualized Sharpe ratio results from
    # the standard computation, ignoring a risk-free rate (using daily returns).
    mean_prior = data.stock.mean()
    std_prior = data.stock.std()
    std_low = std_prior / 1000
    std_high = std_prior * 1000

    with pm.Model() as sharpe_model:
        mean = pm.Normal("mean", mu=mean_prior, sd=std_prior)
        std = pm.Uniform("std", lower=std_low, upper=std_high)
        nu = pm.Exponential("nu_minus_two", 1 / 29, testval=4) + 2.0
        returns = pm.StudentT("returns", nu=nu, mu=mean, sd=std, observed=data.stock)
        sharpe = returns.distribution.mean / returns.distribution.variance ** 0.5 * np.sqrt(252)
        pm.Deterministic("sharpe", sharpe)
    print(sharpe_model.model)
