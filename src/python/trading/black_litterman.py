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


# 무위험수익률, 수익률,공분산으로 효율적 프런티어 계산
def solveFrontier(rets, covs, rf):
    # 최적비중계산 목적함수
    def obj_func(weights, rets, covs, rf):
        mean = sum(rets * weights)
        var = np.dot(np.dot(weights, covs), weights)
        # 최적화 제약조건 페널티
        penalty = 100 * abs(mean - rf)
        return var + penalty

    # 효율적 프론티어를 구성하는 평균-분산을 돌려줄 리스트 준비
    frontier_mean, frontier_var = [], []
    n_rets = len(rets)  # 투자자산 갯수
    # 수익률 최저~최대 사이를 반복
    for ret in np.linspace(min(rets), max(rets), num=20):
        # 최적화 함수에 전달할 초기값으로 동일비중으로 시작
        weights = np.ones([n_rets]) / n_rets
        # 최적화 함수에 전달할 범위조건과 제약조건을 미리 준비
        # 범위조건: 각 구성자산의 투자비중은 0~100% 사이
        # 제약조건: 전체 투자비중은 100%
        bnds = [(0, 1) for i in range(n_rets)]
        cons = ({'type': 'eq', 'fun': lambda wgt: sum(wgt) - 1})
        # 최적화 함수 minimize()은 최적화할 obj함수와 최적화를 시작할 초깃값을 인수로 받음
        results = minimize(obj_func, weights, (rets, covs, ret), method='SLSQP', constraints=cons, bounds=bnds)
        if not results.success:
            raise BaseException(results.message)
        # 효율적 프런티어 평균과 분산리스트에
        # 최적포트폴리오 수익률과 분산 추가
        frontier_mean.append(ret)
        frontier_var.append(np.dot(np.dot(results.x, covs), results.x))

    return np.array(frontier_mean), np.array(frontier_var)

# 무위험수익률, 수익률,공분산으로 샤프비율을 최대로 하는 접점포트폴리오 비중 계산
def solveWeights(rets_annual, covs_annual, rf=0.015):
    # 최적비중계산 목적함수
    def obj_func(weights, rets, covs, rf):
        mean = sum(rets * weights)
        var = np.dot(np.dot(weights, covs), weights)
        # 효용함수 : 샤프비율
        util = (mean - rf) / np.sqrt(var)
        # 효용함수 극대화 = 효용함수 역함수를 최소화
        return 1 / util

    n_rets = len(rets_annual)  # 투자자산 갯수

    # 동일비중으로 최적화 시작
    weights = np.ones([n_rets]) / n_rets
    # 비중범위는 0~100%사이 (공매도나 차입조건이 없음)
    bnds = [(0, 1) for i in range(n_rets)]
    # 제약조건은 비중합=100%
    cons = ({'type': 'eq', 'fun': lambda wgt: sum(wgt) - 1})
    # 최적화
    results = minimize(obj_func, weights, (rets_annual, covs_annual, rf), method='SLSQP', constraints=cons, bounds=bnds)
    if not results.success:
        raise BaseException(results.message)

    return results.x

# 효율적 포트폴리오 최적화
def optimizeFrontier(rets_annual, covs_annual, rf=0.015):
    # 접점포트폴리오 계산
    weights = solveWeights(rets_annual, covs_annual, rf)
    # 투자비중으로 계산한 평균과 분산
    tan_mean = sum(rets_annual * weights)
    tan_var = np.dot(np.dot(rets_annual, covs_annual), weights)
    # 효율적 포트폴리오 계산
    eff_mean, eff_var = solveFrontier(rets_annual, covs_annual, rf)

    # 비중, 접점포트폴리오의 평균/분산, 효율적 포트폴리오의 평균/분산
    return {'weights':weights, 'tan_mean':tan_mean, 'tan_var':tan_var, 'eff_mean':eff_mean, 'eff_var':eff_var}

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
    optim1 = optimizeFrontier(rets_annual, covs_annual, rf)
    print(optim1)


