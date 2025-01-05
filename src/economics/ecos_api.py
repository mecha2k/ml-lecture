import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
import warnings
import os

from PublicDataReader import Ecos
from dotenv import load_dotenv
from io import StringIO
from datetime import datetime
from pathlib import Path


warnings.filterwarnings("ignore", category=UserWarning)


load_dotenv(verbose=True)
api_key = os.getenv("ECOS_API_KEY")
api = Ecos(api_key)

style = plt.style.available
print(style)

plt.style.use("classic")
plt.rcParams["font.size"] = 12
plt.rcParams["figure.dpi"] = 200
plt.rcParams["font.family"] = "NanumBarunGothic"
plt.rcParams["axes.unicode_minus"] = False

start = "202201"
end = "202601"
date_start = datetime.strptime(start, "%Y%m")
date_end = datetime.strptime(end, "%Y%m")


df = api.get_statistic_table_list()
print(df.head())

df = api.get_statistic_word("소비자동향지수")
print(df.head())

df = api.get_statistic_item_list("301Y013")
print(df.head())

# 국제수지(301Y013) : 경상수지(000000) = 상품수지(100000) + 서비스수지(200000) + 본원소득 + 이전소득
# 경상수지(000000)
df = api.get_statistic_search("301Y013", "M", start, end, "000000", translate=False)

df["TIME"] = pd.to_datetime(df["TIME"], format="%Y%m")
df["DATA_VALUE"] = pd.to_numeric(df["DATA_VALUE"])
df1 = df.set_index("TIME")
print(df1.info())

# 상품수지(100000)
df = api.get_statistic_search("301Y013", "M", start, end, "100000", translate=False)

df["TIME"] = pd.to_datetime(df["TIME"], format="%Y%m")
df["DATA_VALUE"] = pd.to_numeric(df["DATA_VALUE"])
df2 = df.set_index("TIME")
print(df2.info())

fig, ax1 = plt.subplots(figsize=(12, 6))
color = "orangered"
ax1.set_ylabel("경상수지 ($Millons)", color=color)
ax1.plot(df1["DATA_VALUE"], color=color, label="경상수지 ($Millions)", linewidth=2)
ax1.tick_params(axis="y", colors=color)

ax2 = ax1.twinx()
color = "blue"
ax2.set_ylabel("상품수지 ($Millons)", color=color)
ax2.plot(df2["DATA_VALUE"], color=color, label="상품수지 ($Millions)", linewidth=2)
ax2.tick_params(axis="y", colors=color)

# ax3 = ax1.twinx()
# color = "green"
# ax3.plot(df_cli["Korea"], color=color, label="OECD CLI_Korea", linewidth=3)
# ax3.spines["right"].set_position(("outward", 60))
# ax3.set_ylabel("OECD CLI_Korea", color=color)
# ax3.tick_params(axis="y", colors=color)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines = lines1 + lines2
labels = labels1 + labels2
plt.legend(lines, labels, loc="best")

plt.gcf().autofmt_xdate()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.gca().xaxis.set_major_locator(mdates.YearLocator(base=1))
ax1.grid(True, which="both", axis="both", linestyle="--", color="gray", alpha=0.5)

ax1.set_xlim(date_start, date_end)
ax1.set_ylim(-10000, 20000)
ax2.set_ylim(-10000, 20000)

plt.tight_layout()
plt.savefig("account_balance.png", dpi=300)

# print(df1["DATA_VALUE"])
# print(df2["DATA_VALUE"])


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
