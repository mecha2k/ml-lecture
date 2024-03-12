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


# 무위험수익률, 수익률, 공분산으로 효율적 프런티어 계산
def solveFrontier(rets, covs):
    # 최적비중계산 목적함수
    def obj_func(x0, rets, covs, rr):
        mean = sum(rets * x0)
        var = np.dot(np.dot(x0, covs), x0)
        # 최적화 제약조건 페널티
        penalty = 100 * abs(mean - rr)
        return var + penalty

    # 효율적 프론티어를 구성하는 평균-분산을 돌려줄 리스트 준비
    frontier_mean, frontier_var = [], []
    n_rets = len(rets)  # 투자자산 갯수
    # 수익률 최저~최대 사이를 반복
    for rr in np.linspace(min(rets), max(rets), num=20):
        # 최적화 함수에 전달할 초기값으로 동일비중으로 시작
        weights = np.random.rand(n_rets)
        # 최적화 함수에 전달할 범위조건과 제약조건을 미리 준비
        # 범위조건: 각 구성자산의 투자비중은 0~100% 사이
        # 제약조건: 전체 투자비중은 100%
        bnds = [(0, 1) for _ in range(n_rets)]
        cons = {"type": "eq", "fun": lambda wgt: sum(wgt) - 1}
        # 최적화 함수 minimize()은 최적화할 obj함수와 최적화를 시작할 초깃값을 인수로 받음
        results = minimize(
            obj_func, weights, (rets, covs, rr), method="SLSQP", constraints=cons, bounds=bnds
        )
        if not results.success:
            raise BaseException(results.message)
        # 효율적 프런티어 평균과 분산리스트에
        # 최적포트폴리오 수익률과 분산 추가
        frontier_mean.append(rr)
        frontier_var.append(np.dot(np.dot(results.x, covs), results.x))

    return np.array(frontier_mean), np.array(frontier_var)


# 무위험수익률, 수익률,공분산으로 샤프비율을 최대로 하는 접점포트폴리오 비중 계산
def solveWeights(rets, covs, rf=0.015):
    # 최적비중계산 목적함수
    def obj_func(x0, rets, covs, rf):
        mean = sum(rets * x0)
        var = np.dot(np.dot(x0, covs), x0)
        # 효용함수 : 샤프비율
        util = (mean - rf) / np.sqrt(var)
        # 효용함수 극대화 = 효용함수 역함수를 최소화
        return 1 / util

    # 초기값
    weights = np.random.rand(len(rets))
    # 비중범위 : 0 ~ 100% (공매도나 차입조건이 없음)
    bnds = [(0, 1) for _ in range(len(rets))]
    # 제약조건은 비중합=100%
    cons = {"type": "eq", "fun": lambda wgt: sum(wgt) - 1}
    # 최적화
    results = minimize(
        obj_func, weights, (rets, covs, rf), method="SLSQP", constraints=cons, bounds=bnds
    )
    if not results.success:
        raise BaseException(results.message)

    return results.x


# 효율적 포트폴리오 최적화
def optimizeFrontier(rets, covs, rf=0.015):
    # 접점포트폴리오 계산
    weights = solveWeights(rets, covs, rf)
    # 투자비중으로 계산한 평균과 분산
    tan_mean = sum(rets * weights)
    tan_var = np.dot(np.dot(rets, covs), weights)
    # 효율적 포트폴리오 계산
    eff_mean, eff_var = solveFrontier(rets, covs)

    # 비중, 접점포트폴리오의 평균/분산, 효율적 포트폴리오의 평균/분산
    return {
        "weights": weights,
        "tan_mean": tan_mean,
        "tan_var": tan_var,
        "eff_mean": eff_mean,
        "eff_var": eff_var,
    }


# 자산에 대한 투자자의 전망과 전망의 기대수익률을 행렬로 만든다
def viewsMatrixPQ(tikers, views):
    # 투자전망과 기대수익률 행렬, views[i][3]에는 기대수익률을 가리킴
    viewsQ = [views[i][3] for i in range(len(views))]

    # 전망행렬 P를 만들기 위해 구성자산 딕셔너리 작성
    # ticsdict = dict(enumerate(tikers))
    # ticsdict = sorted({value: key for key, value in ticsdict.items()})

    # 투자전망
    viewsP = np.zeros((len(views), len(tikers)))
    for n, view in enumerate(views):
        # 가령 전망이 ('MSFT', '>', 'GE', 0.02) 이라면
        # views[i][0] <-- 'MSFT' --> name1
        # views[i][1] <-- '>'
        # views[i][2] <-- 'GE'   --> name2
        # views[i][3] <-- '0.02'
        viewsP[n, tickers.index(view[0])] = +1 if view[1] == ">" else -1
        viewsP[n, tickers.index(view[2])] = -1 if view[1] == ">" else +1

    return viewsP, viewsQ


def plotAssets(tickers, rets, covs, color="black"):
    plt.scatter([covs[i, i] ** 0.5 for i in range(len(tickers))], rets, marker="x", color=color)
    for i in range(len(tickers)):
        plt.text(
            covs[i, i] ** 0.5, rets[i], "  %s" % tickers[i], verticalalignment="center", color=color
        )


def plotFrontier(result, label=None, color="black"):
    plt.text(
        result["tan_var"] ** 0.5,
        result["tan_mean"],
        "tangent",
        verticalalignment="center",
        color=color,
    )
    plt.scatter(result["tan_var"] ** 0.5, result["tan_mean"], marker="o", color=color)
    plt.plot(
        result["eff_var"] ** 0.5,
        result["eff_mean"],
        label=label,
        color=color,
        linewidth=2,
        marker="D",
        markersize=8,
    )


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

    # 블랙-리터만 역최적화
    mean = sum(rets_annual * weights)
    var = np.dot(np.dot(weights, covs_annual), weights)

    # 위험회피계수
    lmbda = (mean - rf) / var
    print(f"LMBDA: {lmbda}")

    # 내재균형초과수익률
    eqPI = lmbda * np.dot(covs_annual, weights)
    print(f"equilibrium PI: {eqPI}")

    # 균형기대수익률로 최적화
    optim2 = optimizeFrontier(eqPI + rf, covs_annual, rf)

    # 투자자 전망과 기대수익률 그리고 전망의 불확실성 계산
    # Q (kx1) = Views on Expected excess returns for some or alll assets
    # P (kxk) = Link matrix, A matrix identifying which assets you have views about
    views = [("XOM", ">", "JPM", 0.02), ("NFLX", "<", "JNJ", 0.02)]
    viewsP, viewsQ = viewsMatrixPQ(tickers, views)
    print(views)
    print(tickers)
    print(viewsP)
    print(viewsQ)

    # 위험조정상수 (~ 1/samples)
    tau = 0.025

    # 투자자 전망의 불확실성 계산
    # tau * P * C * transpose(P)
    omega = tau * viewsP @ covs_annual @ viewsP.T
    print(omega)

    # 투자자 전망과 합쳐진 균형초과수익률 계산, 블랙-리터만 모델 최적화
    bl1 = inv(tau * covs_annual) + viewsP.T @ inv(omega) @ viewsP
    bl2 = inv(tau * covs_annual) @ eqPI + viewsP.T @ inv(omega) @ viewsQ
    bl_eqPI = bl2 @ inv(bl1)

    optim3 = optimizeFrontier(bl_eqPI + rf, covs_annual, rf)

    print("Historical returns")
    print(pd.DataFrame({"Weight": optim1["weights"]}, index=tickers).T)
    print("Intrinsic implied returns")
    print(pd.DataFrame({"Weight": optim2["weights"]}, index=tickers).T)
    print("Implied returns with adjusted views (Black-Litterman)")
    print(pd.DataFrame({"Weight": optim3["weights"]}, index=tickers).T)

    plotAssets(tickers, rets_annual, covs_annual, color="blue")
    plotFrontier(optim1, label="Historical returns", color="blue")
    plotAssets(tickers, eqPI + rf, covs_annual, color="green")
    plotFrontier(optim2, label="Implied returns", color="green")
    plotAssets(tickers, bl_eqPI + rf, covs_annual, color="red")
    plotFrontier(optim3, label="Implied returns (adjusted views)", color="red")

    # 차트 공통 속성 지정 (차트크기, 제목, 범례, 축이름 등)
    plt.rcParams["figure.figsize"] = (32, 24)
    plt.grid(alpha=0.3, color="gray", linestyle="--", linewidth=1)
    plt.title("Portfolio optimization")
    plt.legend()
    plt.xlabel("Variance, $\sigma$")
    plt.ylabel("Mean, $\mu$")
    plt.savefig("black_litterman")
