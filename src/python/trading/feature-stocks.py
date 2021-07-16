import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


def load_asset_data():
    print(os.getcwd())
    asset_file = "src/data/assets1.h5"
    start = 2016
    end = 2017

    with pd.HDFStore(asset_file) as store:
        prices = store["quandl/wiki/prices"]
        stocks = store["us_equities/stocks"]
        stocks.to_csv("stocks.csv")

    prices.info()
    prices = prices.loc[pd.IndexSlice[str(start) : str(end), :], "adj_close"]
    print(prices.head())
    prices = prices.unstack("ticker")
    print(prices.head())
    print(prices.index)
    stocks.info()
    print(stocks.head())
    print(stocks.index)
    stocks = stocks.loc[:, ["marketcap", "ipoyear", "sector"]]
    print(stocks.head())
    print("stocks duplicated count : ", stocks.duplicated().sum())
    stocks = stocks[~stocks.duplicated()]
    stocks.info()
    print(stocks.index.name)
    stocks.index.name = "ticker"
    print(stocks.index.name)

    print(stocks.index)
    print(prices.columns)
    shared = prices.columns.intersection(stocks.index)
    print(shared)
    stocks = stocks.loc[shared, :]
    stocks.info()
    prices = prices.loc[:, shared]
    prices.info()
    print(prices.shape, stocks.shape)
    assert prices.shape[1] == stocks.shape[0]

    print(prices.head())
    monthly_prices = prices.resample("1M").last()
    print(monthly_prices.head())
    monthly_prices.info()

    outlier_cutoff = 0.01
    data = pd.DataFrame()
    lags = [1, 2, 3, 6, 9, 12]
    data = monthly_prices.pct_change(periods=lags[2]).stack()
    print(data.head(n=10))
    data = (
        data.pipe(
            lambda x: x.clip(
                lower=x.quantile(q=outlier_cutoff, interpolation="linear"),
                upper=x.quantile(q=1 - outlier_cutoff, interpolation="linear"),
            )
        )
        .add(1)
        .pow(1 / lags[2])
        .sub(1)
    )
    print(data.head())
    data = data.swaplevel().dropna()
    print(data.head())
    print(data.shape)

    data = pd.DataFrame()
    for lag in lags:
        data[f"return_{lag}m"] = (
            monthly_prices.pct_change(periods=lag)
            .stack()
            .pipe(
                lambda x: x.clip(
                    lower=x.quantile(outlier_cutoff), upper=x.quantile(1 - outlier_cutoff)
                )
            )
            .add(1)
            .pow(1 / lag)
            .sub(1)
        )
    data = data.swaplevel().dropna()
    print(data.info())
    print(data.head())

    min_obs = 120
    nobs = data.groupby(level="ticker").size()
    keep = nobs[nobs > min_obs].index
    print(keep.shape)

    data = data.loc[pd.IndexSlice[keep, :], :]
    print(data.info())
    cmap = sns.diverging_palette(10, 220, as_cmap=True)
    sns.clustermap(data.corr("spearman"), annot=True, center=0, cmap=cmap)
    plt.show()


if __name__ == "__main__":
    load_asset_data()
