import yfinance as yf
import pandas_datareader.data as pdr
from pandas_datareader.famafrench import get_available_datasets
import statsmodels.api as sm
import numpy as np
import datetime
import os


def stats_models():
    duncan_prestige = sm.datasets.get_rdataset("Duncan", "carData")
    Y = duncan_prestige.data["income"]
    X = duncan_prestige.data["education"]
    X = sm.add_constant(X)
    model = sm.OLS(Y, X)
    results = model.fit()
    print(results.params)
    print(results.tvalues)
    print(results.t_test([1, 0]))


def fama_french():
    datasets = get_available_datasets()
    print("No. of datasets:{0}".format(len(datasets)))

    ff_factor = "F-F_Research_Data_5_Factors_2x3"
    ff_factor_data = pdr.DataReader(ff_factor, "famafrench", start="2010", end="2017-12")
    print(ff_factor_data["DESCR"])

    ff_portfolio = "17_Industry_Portfolios"
    ff_portfolio_data = pdr.DataReader(ff_portfolio, "famafrench", start="2010", end="2017-12")
    print(ff_portfolio_data["DESCR"])

    ff_factor_data[0].to_csv("data/F-F_Research_Data_5_Factors_2x3.csv")
    ff_portfolio_data[0].to_csv("data/17_Industry_Portfolios.csv")

    ff_portfolio_data = ff_portfolio_data[0].sub(ff_factor_data[0].RF, axis=0)
    ff_portfolio_data.info()


def get_yahoo_data():
    apple = yf.Ticker("AAPL")
    apple = yf.download("AAPL", start="2019-01-01")
    print(apple.info)

    print(datetime.datetime.now())

    start = datetime.datetime(2010, 1, 1)
    end = datetime.datetime(2011, 1, 1)
    print(start, end)

    samsung = yf.Ticker("005930.KS")
    samsung = yf.download("005930.KS", start="2010-01-01", end="2011-01-01")
    samsung = yf.download("005930.KS", start=start, end=end)
    print(samsung.info)


if __name__ == "__main__":
    # get_yahoo_data()
    # fama_french()
    stats_models()
