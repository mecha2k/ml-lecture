import yfinance as yf
import pandas as pd
import datetime
import os


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
    get_yahoo_data()
