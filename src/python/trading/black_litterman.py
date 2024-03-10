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


# 무위험수익률, 수익률,공분산으로 샤프비율을 최대로 하는 접점포트폴리오 비중 계산
def solveWeights(R, C, rf):
    # 파이썬은 함수안에 함수를 정의할 수 있다
    # 최적비중계산을 위해 다음과 같이 목적함수를 정의한다
    def obj(W, R, C, rf):
        mean = sum(R * W)
        var = np.dot(np.dot(W, C), W)
        # 샤프비율을 효용함수로 한다
        util = (mean - rf) / np.sqrt(var)
        # 효용함수 극대화= 효용함수 역함수를 최소화하는 것이다.
        return 1 / util

    n = len(R)  # 투자자산 갯수

    # 동일비중으로 최적화 시작
    W = np.ones([n]) / n
    # 비중범위는 0~100%사이(공매도나 차입조건이 없음)
    bnds = [(0., 1.) for i in range(n)]
    # 제약조건은 비중합=100%
    cons = ({'type': 'eq', 'fun': lambda W: sum(W) - 1.})
    # 최적화
    res = minimize(obj, W, (R, C, rf), method='SLSQP', constraints=cons, bounds=bnds)
    if not res.success:
        # 최적화 실패한 경우
        raise BaseException(res.message)
    # 최적화 결과를 돌려준다
    return res.x

# 효율적 포트폴리오 최적화
def optimizeFrontier(rets_annual, covs_annual, rf=0.015):
    # 접점포트폴리오 계산
    weights = solveWeights(rets_annual, covs_annual, rf)
    # 투자비중으로 계산한 평균과 분산
    tan_mean = sum(R * W)
    tan_var = np.dot(np.dot(W, C), W)

    # 효율적 포트폴리오 계산
    eff_mean, eff_var = solveFrontier(R, C, rf)

    # 비중, 접점포트폴리오의 평균/분산, 효율적 포트폴리오의 평균/분산을
    # 딕셔너리 데이터형으로 돌려준다
    return {'weights':W, 'tan_mean':tan_mean, 'tan_var':tan_var, 'eff_mean':eff_mean, 'eff_var':eff_var}

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

    # 과거 데이터를 이용한 최적화
    # opt1 = optimize_frontier(R, C, rf)


