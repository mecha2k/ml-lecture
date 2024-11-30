import numpy as np
import pandas as pd
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

sns.set_style("whitegrid")

# plt.style.use("seaborn")
# plt.rcParams["font.size"] = 16
# plt.rcParams["figure.dpi"] = 300
# plt.rcParams["font.family"] = "NanumBarunGothic"
# plt.rcParams["axes.unicode_minus"] = False

# df = pd.concat([pk1_df.종가, pk2_df.종가, fdr1_df.Close, fdr2_df.Close, fdr3_df.Close, fdr3_df['Adj Close'], marcap_df.Close], axis = 1)
# df.columns = ['pykrx_Adj', 'pykrx_NotAdj', 'fdr_NAVER', 'fdr_KRX', 'fdr_YAHOO', 'fdr_YAHOO_Adj', 'marcap']
# df.plot(logy = True)
# plt.plot()

#
# # KOSPI Index 코스피 지수 데이터
# df = fdr.DataReader('KS11', '2020') # 2020-01-01 ~ 현재
# df = fdr.DataReader('KS11', '2022-01-01', '2022-12-31') # 2022-01-01 ~ 2022-12-31
#
# # KRX Indices 국내 지수 데이터
start = datetime(2010, 1, 1)
end = datetime(2026, 1, 1)
df = fdr.DataReader("KS11")  # KOSPI 지수 (KRX)
print(df.info())


def plot_train_test_dist(df, pieces=10, img_path=None, mode="train"):
    if mode == "preds":
        anomalies = df["anomaly"].values
        df = df.drop(columns=["anomaly"])
    ncols = len(df.columns.values)
    num_plots = max(np.round(ncols / pieces, 0).astype(int), 1)
    xticks = np.arange(0, len(df), np.ceil(len(df) // 10))
    fig, ax = plt.subplots(pieces, 1, figsize=(12, 3 * pieces))
    for i in range(pieces):
        start = i * num_plots
        end = min(start + num_plots, ncols)
        ax[i].plot(df.iloc[:, start:end])
        if mode == "preds":
            ax[i].plot(anomalies * 1.5, color="red", alpha=0.3, linewidth=5)
        ax[i].set_xticks(xticks)
        ax[i].set_title(f"Columns {start} to {end}")
        ax[i].ticklabel_format(style="scientific", axis="x", scilimits=(0, 0))
        ax[i].legend(df.columns[start:end], loc="upper right")
        ax[i].grid()
        ax[i].set_ylim(-0.5, 1.5)
        ax[i].axhline(0, color="r", alpha=0.5)
        ax[i].axhline(1, color="r", alpha=0.5)
    fig.tight_layout()
    fig.savefig(img_path / f"{mode}_dist.png")


import matplotlib.dates as dates

# fig, ax = plt.subplots(figsize=(10, 7))
#
# # 1. 그래프를 그린다.
# ax.scatter(x=df['date'], y=df['value'])
# # 2. DateFormatter 객체를 생성한다. 포맷팅할 형식도 함께 지정한다.
# dateFmt = mdates.DateFormatter('%Y-%m-%d')
# # 3. x축 레이블을 포맷팅한다.
# ax.xaxis.set_major_formatter(dateFmt)


fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(df.index, df.Close)
dateformat = dates.DateFormatter("%Y-%m")
ax1.xaxis.set_major_formatter(dateformat)
current_values = plt.gca().get_yticks()
ax2 = ax1.twinx()

plt.grid(True)
plt.legend("KOSPI")
plt.xlim([start, end])
plt.ylim([1000, 4001])
plt.xticks(rotation=45)
plt.yticks(np.arange(1000, 4001, 500))
plt.tick_params(axis="both", which="major", labelsize=12)
plt.ticklabel_format(style="plain", axis="y", scilimits=(0, 0))
plt.tight_layout()
plt.savefig("KOSPI.png")

# plot df
# df.Close.plot(logy=True)
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.show()

# df = fdr.DataReader('KQ11') # KOSDAQ 지수 (KRX)
# df = fdr.DataReader('KS200') # KOSPI 200 (KRX)
#
# # US market Indices 미국 시장 지수 데이터
# df = fdr.DataReader('DJI') # 다우존스 지수 (DJI - Dow Jones Industrial Average)
# df = fdr.DataReader('IXIC') # 나스닥 종합지수 (IXIC - NASDAQ Composite)
# df = fdr.DataReader('S&P500') # S&P500 지수 (NYSE)
# df = fdr.DataReader('RUT') # 러셀2000 지수 (RUT - US Small Cap 2000)
# df = fdr.DataReader('VIX') # VIX지수 (VIX - CBOE Volatility Index)
#
# # Global Indices 글로벌 지수 데이터
# df = fdr.DataReader('SSEC') # 상해 종합지수 Shanghai (SSEC -Shanghai Composite)
# df = fdr.DataReader('HSI') # 항셍지수 (HSI - Hang Seng)
# df = fdr.DataReader('N225') # 일본 닛케이지수 (N225 - Nikkei 225)
# df = fdr.DataReader('FTSE') # 영국 FTSE100 (FTSE 100 - Financial Times Stock Exchange)
# df = fdr.DataReader('FCHI') # 프랑스 FCHI 지수 (CAC 40 - CAC quarante)
# df = fdr.DataReader('GDAXI') # 독일 닥스지수  (DAX30 - germany-30)
#
# # KRX stock price 국내 시장 개별종목
# df = fdr.DataReader('005930') # 삼성전자 전체 (1999년 ~ 현재)
# df = fdr.DataReader('000660') # SK하이닉스 전체 (1999년 ~ 현재)
# df = fdr.DataReader('068270') # 셀트리온 전체 (2004년 상장 ~ 현재)
#
# # 여러 종목 종가(Close) 한번에
# # 삼성전자(005930), SK하이닉스(000660), 기아(000270), 카카오(035720), KB금융(105560)
# df = fdr.DataReader('005930,000660,000270,035720,105560', '2020') # 2020년 ~ 현재
#
# # US stock price 미국 시장 개별종목
# df = fdr.DataReader('AAPL', '2017') # Apple(AAPL), 2017-01-01 ~ 현재
# df = fdr.DataReader('AMZN', '2017', '2019-12-31') # AMAZON(AMZN), 2017~2019 (3년)
# df = fdr.DataReader('F', '1980-01-01', '2023-10-01') # Ford 자동차(F) (40년간)
#
# # 여러종목 한번에 종가(Close) 데이터
# df = fdr.DataReader('AAPL, TSLA, AMZN', '2020') # 애플, 테슬라, 아마존 (2020년 ~ 현재)
#
# # 데이터 소스 지정하기
# df = fdr.DataReader('KRX:000150', '2020-01-01') # 두산(000150) (한국거래소)
# df = fdr.DataReader('NAVER:000150', '2020-01-01') # 두산(000150) (네이버 파이낸스)
# df = fdr.DataReader('YAHOO:000150.KS', '2020-01-01') # 두산(000150) (야후 파이낸스)
#
# # TSE (도쿄증권거래소)
# df = fdr.DataReader('TSE:7203', '2020-01-01') # Toyota Motor Corp 토요타 자동차(7203)
# df = fdr.DataReader('TSE:9984', '2020-01-01') # SoftBank Group Corp 소프트뱅크그룹(9984)
#
# # HOSE (호치민증권거래소)
# df = fdr.DataReader('HOSE:VCB', '2020-01-01') # 베트남 무역은행(VCB)
# df = fdr.DataReader('HOSE:VIC') # Vingroup (JSC)
#
# # 글로벌 동일한 종목코드 경우 거래소를 지정
# df = fdr.DataReader('000150', '2020-01-01') # 두산:KRX 종목 (기본:네이버 파이낸스)
# df = fdr.DataReader('KRX:000150', '2020-01-01') # 두산:KRX 종목 (한국거래소 데이터)
# df = fdr.DataReader('SSE:000150', '2020-01-01') # SSE 380 Dividend Index (상하이 거래소)
#
# # 상품 선물 가격 데이터
# df = fdr.DataReader('CL=F') # WTI유 선물 Crude Oil (NYMEX)
# df = fdr.DataReader('BZ=F') # 브렌트유 선물 Brent Oil (NYMEX)
# df = fdr.DataReader('NG=F') # 천연가스 선물 (NYMEX)
# df = fdr.DataReader('GC=F') # 금 선물 (COMEX)
# df = fdr.DataReader('SI=F') # 은 선물 (COMEX)
# df = fdr.DataReader('HG=F') # 구리 선물 (COMEX)
#
# # 환율: 여러 조합 가능(지원 심볼: ['KRW', 'EUR', 'CNY', 'JPY', 'CHF'])
# df = fdr.DataReader('USD/KRW') # 달러 원화
# df = fdr.DataReader('USD/EUR') # 달러 유로화
# df = fdr.DataReader('USD/CNY') # 달러 위엔화
# df = fdr.DataReader('CNY/KRW') # 위엔화 원화
# df = fdr.DataReader('EUR/CNY') # 유로화 위엔화
#
# # 암호화폐 가격 데이터 (원화, 달러)
# # (지원 심볼: ['BTC', 'ETH', 'USDT', 'BNB', 'USDC', 'XRP', 'BUSD', 'ADA', 'SOL', 'DOGE'])
# df = fdr.DataReader('BTC/KRW') # 비트코인/원화
# df = fdr.DataReader('ETH/KRW') # 이더리움/원화
# df = fdr.DataReader('BTC/USD') # 비트코인/달러
# df = fdr.DataReader('ETH/USD') # 이더리움/달러
#
# # KRX delisting stock data 상장폐지 종목 전체 가격 데이터
# df = fdr.DataReader('KRX-DELISTING:036360') # 3SOFT(036360)
#
# # 미국 국채 채권 수익률
# df = fdr.DataReader('US5YT')   # 5년 만기 미국국채 수익률
# df = fdr.DataReader('US10YT') # 10년 만기 미국국채 수익률
# df = fdr.DataReader('US30YT') # 30년 만기 미국국채 수익률
#
# # 종목 리스팅 (종목수는 2022년 10월 25일 기준, 시장 규모 가늠 용도)
# # KRX 상장회사(발행회사)목록 (가격 중심, 주식 종목) - 시가총액순
# stocks = fdr.StockListing('KRX') # KRX: 2,663 종목(=코스피+코스닥+코넥스)
# stocks = fdr.StockListing('KOSPI') # KOSPI: 940 종목
# stocks = fdr.StockListing('KOSDAQ') # KOSDAQ: 1,597 종목
# stocks = fdr.StockListing('KONEX') # KONEX: 126 종목
#
# # KRX 전종목 목록 (설명 중심, 주식+펀드등 전종목)
# stocks = fdr.StockListing('KRX-DESC') # 한국거래소: 7,632 종목
# stocks = fdr.StockListing('KOSPI-DESC') # KOSPI: 5,897 종목
# stocks = fdr.StockListing('KOSDAQ-DESC') # KOSDAQ: 1,609 종목
# stocks = fdr.StockListing('KONEX-DESC') # KONEX: 126 종목
#
# # KRX 특수 종목 리스팅 (상장폐지 종목, 관리종목)
# stocks = fdr.StockListing('KRX-DELISTING') # 3천+ 종목 - 상장폐지 종목 전체
# stocks = fdr.StockListing('KRX-ADMIN') # 50+ 종목 - KRX 관리종목
#
# # US Market listings 미국 시장 거래소별 전종목 리스팅
# stocks = fdr.StockListing('S&P500') # S&P500: 503 종목
# stocks = fdr.StockListing('NASDAQ') # 나스닥 (NASDAQ): 4천+ 종목
# stocks = fdr.StockListing('NYSE') # 뉴욕증권거래소 (NYSE): 3천+ 종목
#
# # Global Market listings 글로벌 시장 거래소별 전종목 리스팅
# stocks = fdr.StockListing('SSE') # 상하이 증권거래소 (Shanghai Stock Exchange: SSE): 1천+ 종목
# stocks = fdr.StockListing('SZSE') # 선전 증권거래소(Shenzhen Stock Exchange: SZSE): 1천+ 종목
# stocks = fdr.StockListing('HKEX') # 홍콩 증권거래소(Hong Kong Exchange: HKEX): 2천5백+ 종목
# stocks = fdr.StockListing('TSE') # 도쿄 증권거래소(Tokyo Stock Exchange: TSE): 3천9백+ 종목
# stocks = fdr.StockListing('HOSE') # 호찌민 증권거래소(Ho Chi Minh City Stock Exchange: HOSE): 4백+ 종목
#
# # KRX ETFs
# etfs = fdr.StockListing('ETF/KR') # 한국 ETF 전종목
#
# # FRED 데이터
# df = fdr.DataReader('FRED:M2') #  M2 통화량
# df = fdr.DataReader('FRED:NASDAQCOM') # NASDAQCOM 나스닥종합지수
# df = fdr.DataReader('FRED:T10Y2Y') # 미국 장단기금리차 (1980년 ~)
#
# # 달러 인덱스
# df = fdr.DataReader('^NYICDX') # ICE U.S. Dollar Index (^NYICDX) 달러인덱스 (1980~현재)
#
# # FRED 데이터 여러 항목 한번에
# df = fdr.DataReader('FRED:M2,HSN1F,NASDAQCOM')  # M2 통화량, HSN1F 주택판매지수, NASDAQCOM 나스닥종합지수
#
# #  KRX지수및 지수 구 성종목
# df = fdr.SnapDataReader('KRX/INDEX/LIST') # KRX 전체 지수목록
#
# df = fdr.SnapDataReader('KRX/INDEX/STOCK/1001') # KOSPI 지수구성종목
# df = fdr.SnapDataReader('KRX/INDEX/STOCK/1028') # 코스피 200
# df = fdr.SnapDataReader('KRX/INDEX/STOCK/5106') # KRX ESG Leaders 150 테마 지수 구성종목
#
#
# # import pandas as pd
# # import pandas_datareader.data as web
# # import requests, pprint
# # import datetime
# #
# #
# # def get_oecd_data(dataset_id, country_code):
# #     url = f"https://stats.oecd.org/SDMX-JSON/data/{dataset_id}/{country_code}/.../all"
# #     response = requests.get(url)
# #     data = response.json()["data"]
# #
# #     # JSON 데이터를 pandas DataFrame으로 변환하는 로직 구현
# #     # (이 부분은 실제 데이터 구조에 따라 달라질 수 있음)
# #
# #     return pd.DataFrame(data)
# #
# #
# # # 사용 예시
# # dataset_id = "MEI_CLI"  # Composite Leading Indicator 데이터셋
# # country_code = "KOR"  # 대한민국
# #
# # url = f"https://stats.oecd.org/SDMX-JSON/data/{dataset_id}/{country_code}/.../all"
# # response = requests.get(url)
# # data = response.json()
# # pprint.pprint(data, depth=4, compact=True)
#
# # df = pd.DataFrame(data["data"]["dataSets"][0])
#
#
# # df = get_oecd_data(dataset_id, country_code)
# # print(df)
# #
# # # 날짜 설정
# # start_date = datetime.datetime(2022, 1, 1)
# # end_date = datetime.datetime(2023, 12, 31)
# #
# # # OECD 데이터베이스 지정
# # database = "MEI_CLI"
# #
# # # 데이터 다운로드
# # cli_data = web.DataReader(database, "oecd", start_date, end_date)
# #
# # # 결과 출력
# # print(cli_data)
