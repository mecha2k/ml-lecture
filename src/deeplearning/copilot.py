# find TSLA.US stock price data and save it to a csv file
def get_stock_data(ticker):
    import pandas_datareader.data as web
    import datetime
    import pandas as pd
    start = datetime.datetime(2010, 1, 1)
    end = datetime.datetime(2019, 1, 1)
    df = web.DataReader(ticker, 'yahoo', start, end)
    df.to_csv('stock_data.csv')
    return df


# find the stock price of TSLA.US today
def get_stock_price():
    import pandas_datareader.data as web
    import datetime
    import pandas as pd
    today = datetime.datetime.now()
    today = today.strftime("%Y-%m-%d")
    df = web.DataReader('TSLA', 'yahoo', today, today)
    return df['Close'][0]

def findTSLAstockpriceToday():
    import pandas_datareader.data as web
    import datetime
    import pandas as pd
    today = datetime.datetime.now()
    today = today.strftime("%Y-%m-%d")
    df = web.DataReader('TSLA', 'yahoo', today, today)
    return df['Close'][0]

print(get_stock_price())