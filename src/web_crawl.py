import urllib3
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime


def urllib3_ex():
    http = urllib3.PoolManager()
    url = "http://webcode.me"
    res = http.request("GET", url)

    print(res.status)
    print(res.data.decode("utf-8"))


def read_naver_sise(code, company, pages_to_fetch):
    try:
        url = f"http://finance.naver.com/item/sise_day.nhn?code={code}"
        headers = {"User-agent": "Mozilla/5.0"}
        html = BeautifulSoup(requests.get(url, headers=headers).text, "lxml")
        pgrr = html.find("td", class_="pgRR")
        if pgrr is None:
            return None
        s = str(pgrr.a["href"]).split("=")
        lastpage = s[-1]
        df = pd.DataFrame()
        pages = min(int(lastpage), pages_to_fetch)
        for page in range(1, pages + 1):
            pg_url = f"{url}&page={page}"
            pg_data = pd.read_html(requests.get(pg_url, headers=headers).text)
            pg_data = pg_data[0]
            df = df.append(pg_data)
            tmnow = datetime.now().strftime("%Y-%m-%d %H:%M")
            print(
                f"[{tmnow}] {company} ({code}) : {page:04d}/{pages:04d} pages are downloading...",
                end="\r",
            )
        df = df.rename(
            columns={
                "날짜": "date",
                "종가": "close",
                "전일비": "diff",
                "시가": "open",
                "고가": "high",
                "저가": "low",
                "거래량": "volume",
            }
        )
        df["date"] = df["date"].replace(".", "-")
        df = df.dropna()
        df[["close", "diff", "open", "high", "low", "volume"]] = df[
            ["close", "diff", "open", "high", "low", "volume"]
        ].astype(int)
        df = df[["date", "open", "high", "low", "close", "diff", "volume"]]
    except Exception as e:
        print("Exception occured :", str(e))
        return None
    return df


if __name__ == "__main__":
    data = read_naver_sise("068270", "셀트리온", 500)
    print(data)
