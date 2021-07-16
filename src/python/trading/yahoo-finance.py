import yfinance as yf
import pandas as pd
import os


def get_yahoo_data():
    apple = yf.Ticker("AAPL")
    apple = yf.download("AAPL", start="2019-01-01")
    print(apple.info)


if __name__ == "__main__":
    get_yahoo_data()