import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import pickle
import os

from numpy.linalg import inv
from scipy.optimize import minimize
from datetime import datetime, timedelta


tickers = ["PFE", "INTC", "NFLX", "JPM", "XOM", "GOOG", "JNJ", "AAPL", "AMZN"]
marketcap = {
    "PFE": 201102000000,
    "INTC": 257259000000,
    "NFLX": 184922000000,
    "JPM": 272178000000,
    "XOM": 178228000000,
    "GOOG": 866683000000,
    "JNJ": 403335000000,
    "AAPL": 1208000000000,
    "AMZN": 1178000000000,
}

startDate = datetime(2018, 1, 1)
endDate = datetime(2019, 12, 31)


if __name__ == "__main__":
    filepath = "../../data/stock_data.pickle"
    if os.path.exists(filepath):
        stockdata = pd.read_pickle(filepath)
        print("Data opened successfully")
    else:
        stockdata = list()
        for ticker in tickers:
            df = yf.download(ticker, start=startDate, end=endDate)
            stockdata.append({"price": df, "marketcap": marketcap[ticker]})
        with open(filepath, "wb") as outfile:
            pickle.dump(stockdata, outfile)
            print(f"Data saved successfully to {filepath}")

    prices = list()
    for stock in stockdata:
        prices.append(list(stock["price"]["Adj Close"]))

    capitals = list(marketcap.values())
    weights = np.array(capitals) / sum(capitals)  # 시가총액의 비율계산
    prices = np.matrix(prices)
    print(prices.shape)

    # 수익률 행렬을 만들어 계산
    rows, cols = prices.shape
    returns = np.empty([rows, cols - 1])
    for row in range(rows):
        for col in range(cols - 1):
            p0, p1 = prices[row, col], prices[row, col + 1]
            returns[row, col] = (p1 / p0) - 1
    print(returns.shape)

    # 수익률계산
    exp_rets = np.array([np.mean(returns[row, :]) for row in range(rows)])
    print(exp_rets)

    # 공분산계산
    covars = np.cov(returns)
    rets_annual = (1 + exp_rets) ** 250 - 1  # 연율화
    covs_annual = covars * 250  # 연율화

    # 무위험 이자율
    rf = 0.015

    print(
        pd.DataFrame(
            {"Return": rets_annual, "Weight (based on market cap)": weights}, index=tickers
        ).T
    )
    print(pd.DataFrame(covs_annual, columns=tickers, index=tickers))
