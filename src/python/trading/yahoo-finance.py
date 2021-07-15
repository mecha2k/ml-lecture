import yfinance as yf
import pandas as pd


if __name__ == "__main__":
    apple = yf.Ticker("AAPL")
    apple = yf.download('AAPL',start = '2019-01-01')
    print(apple.info)
