import pandas as pd
import pandas_datareader.data as web
import requests, pprint
import datetime


def get_oecd_data(dataset_id, country_code):
    url = f"https://stats.oecd.org/SDMX-JSON/data/{dataset_id}/{country_code}/.../all"
    response = requests.get(url)
    data = response.json()["data"]

    # JSON 데이터를 pandas DataFrame으로 변환하는 로직 구현
    # (이 부분은 실제 데이터 구조에 따라 달라질 수 있음)

    return pd.DataFrame(data)


# 사용 예시
dataset_id = "MEI_CLI"  # Composite Leading Indicator 데이터셋
country_code = "KOR"  # 대한민국

url = f"https://stats.oecd.org/SDMX-JSON/data/{dataset_id}/{country_code}/.../all"
response = requests.get(url)
data = response.json()
pprint.pprint(data, depth=4, compact=True)

# df = pd.DataFrame(data["data"]["dataSets"][0])


# df = get_oecd_data(dataset_id, country_code)
# print(df)
#
# # 날짜 설정
# start_date = datetime.datetime(2022, 1, 1)
# end_date = datetime.datetime(2023, 12, 31)
#
# # OECD 데이터베이스 지정
# database = "MEI_CLI"
#
# # 데이터 다운로드
# cli_data = web.DataReader(database, "oecd", start_date, end_date)
#
# # 결과 출력
# print(cli_data)
