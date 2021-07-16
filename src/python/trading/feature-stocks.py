import yfinance as yf
import pandas as pd
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

    monthly_prices = prices.resample("M").last()
    monthly_prices.info()


if __name__ == "__main__":
    load_asset_data()
