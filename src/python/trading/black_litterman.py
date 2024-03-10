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
        data = pd.read_pickle(filepath)
    else:
        data = list()
        for ticker in tickers:
            df = yf.download(ticker, start=startDate, end=endDate)
            data.append({"price": df, "marketcap": marketcap[ticker]})
        with open(filepath, "wb") as outfile:
            pickle.dump(data, outfile)
            print(f"Data saved successfully to {filepath}")
