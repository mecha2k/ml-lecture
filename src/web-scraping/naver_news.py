from bs4 import BeautifulSoup
import urllib.request
import requests
import re

url = "https://search.naver.com/search.naver?where=news&sm=tab_jum&query=삼성전자+주가"

html = requests.get(url)
html.raise_for_status()
print(html.status_code)

# html = urllib.request.urlopen(url.encode("utf8")).read()

soup = BeautifulSoup(html.text, "lxml")
items = soup.find("ul", class_="list_news").find_all("li", class_="bx")
print(len(items))

file = open("soup.txt", "w", encoding="utf8", newline="\n")
for item in items:
    press = item.find("div", class_="info_group").a.text.split()[0]
    news_url = item.find_all("a", class_="info")[1]["href"]
    print(press)
    print(news_url)
    news_link = urllib.request.urlopen(news_url).read()
    soup = BeautifulSoup(news_link, features="lxml")
    title = soup.find("div", class_="media_end_head_title").text
    date_time = soup.find(
        "span", class_="media_end_head_info_datestamp_time _ARTICLE_DATE_TIME"
    ).text
    content = soup.find("div", class_="newsct_article _article_body").text
    content = re.sub(pattern="[\n\t]", repl="", string=content)
    print(title)
    print(date_time)
    print(content.strip())

    file.write(press + "\n")
    file.write(news_url + "\n")
    file.write(title + "\n")
    file.write(date_time + "\n")
    file.write(content + "\n")
