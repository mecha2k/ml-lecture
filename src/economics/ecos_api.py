import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import warnings
import os

from PublicDataReader import Ecos
from dotenv import load_dotenv
from io import StringIO


load_dotenv(verbose=True)
warnings.filterwarnings("ignore", category=UserWarning)

api_key = os.getenv("ECOS_API_KEY")
api = Ecos(api_key)

df = api.get_statistic_table_list()
print(df.head())

df = api.get_statistic_word("소비자동향지수")
print(df.head())

df = api.get_statistic_item_list("301Y013")
print(df.head())

# 국제수지(301Y013) : 경상수지(000000) = 상품수지(100000) + 서비스수지(200000) + 본원소득 + 이전소득
df = api.get_statistic_search("301Y013", "M", "202301", "202512", "000000", translate=False)

df["TIME"] = pd.to_datetime(df["TIME"], format="%Y%m")
df["DATA_VALUE"] = pd.to_numeric(df["DATA_VALUE"])
df = df.set_index("TIME")
print(df.info())

# plt.plot(df["DATA_VALUE"])
# plt.show()


# def get_ecos_data(stat_code, item_code1, item_code2, freq, start_date, end_date):
#     url = "https://ecos.bok.or.kr/api/StatisticSearch/[YOUR_API_KEY]/json/kr/1/100000/"
#     url += f"{stat_code}/{freq}/{start_date}/{end_date}/{item_code1}/{item_code2}"
#
#     response = requests.get(url)
#     data = response.json()["StatisticSearch"]["row"]
#
#     df = pd.DataFrame(data)
#     df["TIME"] = pd.to_datetime(df["TIME"])
#     df["DATA_VALUE"] = pd.to_numeric(df["DATA_VALUE"])
#
#     return df.set_index("TIME")

# # 무역수지 데이터 다운로드
# trade_balance = get_ecos_data("301Y013", "KM", "0000001", "M", "201001", "202412")
#
# # 경상수지 데이터 다운로드
# current_account = get_ecos_data("301Y013", "CA", "0000001", "M", "201001", "202412")
#
# # 데이터 병합
# combined_data = pd.concat([trade_balance, current_account], axis=1)
# combined_data.columns = ["무역수지", "경상수지"]
#
# # CSV 파일로 저장
# combined_data.to_csv("korea_trade_current_account.csv", encoding="utf-8-sig")
#
# print("데이터가 성공적으로 다운로드되어 'korea_trade_current_account.csv' 파일로 저장되었습니다.")
