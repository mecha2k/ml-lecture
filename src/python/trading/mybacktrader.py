from __future__ import absolute_import, division, print_function, unicode_literals

import backtrader as bt
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import warnings

from datetime import datetime
from icecream import ic

plt.style.use("seaborn-colorblind")
plt.rcParams["figure.figsize"] = [10, 6]
plt.rcParams["figure.dpi"] = 150
warnings.simplefilter(action="ignore", category=FutureWarning)


class TestStrategy(bt.Strategy):
    params = (
        ("maperiod", 15),
        ("printlog", False),
    )

    def log(self, txt, dt=None, doprint=False):
        """Logging function fot this strategy"""
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print("%s, %s" % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close
        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None
        # Add a MovingAverageSimple indicator
        self.sma = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.params.maperiod)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return
        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    "BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f"
                    % (order.executed.price, order.executed.value, order.executed.comm)
                )
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log(
                    "SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f"
                    % (order.executed.price, order.executed.value, order.executed.comm)
                )
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order Canceled/Margin/Rejected")
        # Write down: no pending order
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log("OPERATION PROFIT, GROSS %.2f, NET %.2f" % (trade.pnl, trade.pnlcomm))

    def next(self):
        # Simply log the closing price of the series from the reference
        self.log("Close, %.2f" % self.dataclose[0])
        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return
        # Check if we are in the market
        if not self.position:
            # Not yet ... we MIGHT BUY if ...
            if self.dataclose[0] > self.sma[0]:
                # BUY, BUY, BUY!!! (with all possible default parameters)
                self.log("BUY CREATE, %.2f" % self.dataclose[0])
                # Keep track of the created order to avoid a 2nd order
                self.order = self.buy()
        else:
            if self.dataclose[0] < self.sma[0]:
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log("SELL CREATE, %.2f" % self.dataclose[0])
                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()

    def stop(self):
        self.log(
            "(MA Period %2d) Ending Value %.2f" % (self.params.maperiod, self.broker.getvalue()),
            doprint=True,
        )


if __name__ == "__main__":
    src_data = "data/yf_aapl.pkl"
    start = datetime(2000, 1, 1)
    end = datetime(2020, 12, 31)
    try:
        aapl = pd.read_pickle(src_data)
        print("data reading from file...")
    except FileNotFoundError:
        aapl = yf.download("AAPL", start=start, end=end, auto_adjust=True)
        aapl.to_pickle(src_data)

    aapl_df = aapl["2018-1-1":"2018-12-31"]
    data = bt.feeds.PandasData(dataname=aapl_df)
    ic(aapl_df.head())

    cerebro = bt.Cerebro(stdstats=True)
    cerebro.addstrategy(TestStrategy)
    cerebro.adddata(data)
    cerebro.broker.setcash(1000.0)
    cerebro.addsizer(bt.sizers.FixedSize, stake=10)
    cerebro.broker.setcommission(commission=0.0)

    ic(f"start : {cerebro.broker.getvalue():.2f}")
    cerebro.run(maxcpus=16)
    ic(f"final : {cerebro.broker.getvalue():.2f}")

    cerebro.plot(iplot=False, volume=True)
