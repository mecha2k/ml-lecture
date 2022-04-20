from bs4 import BeautifulSoup
from urllib.parse import quote
import urllib.request
import requests

url = "https://search.naver.com/search.naver?sm=tab_hty.top&where=news&query="
url += quote("삼성전자+주가")

html = urllib.request.urlopen(url).read()
soup = BeautifulSoup(html, "html.parser")

news = soup.find_all("div", attrs={"class": "info_group"})
for item in news:
    press = item.find("a", attrs={"class": "info press"}).get_text()
    news_url = item.find("a").get("href")
    print(press)
    print(news_url)
# sp_nws1 > div.news_wrap.api_ani_send > div > div.news_info > div.info_group > a:nth-child(3)
print(len(news))
