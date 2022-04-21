from bs4 import BeautifulSoup
import requests

url = "https://www.timesjobs.com/candidate/job-search.html"
url += "?searchType=personalizedSearch&from=submit&txtKeywords=python&txtLocation="

html = requests.get(url)
print(html.url)
print(html.status_code)

soup = BeautifulSoup(html.text, "lxml")
jobs = soup.find_all("li", class_="clearfix job-bx wht-shd-bx")
for job in jobs:
    company = job.find("h3", class_="joblist-comp-name")
    print(company.text.strip())
print(len(jobs))

# url = "https://search.naver.com/search.naver?sm=tab_hty.top&where=news&query="
# url += quote("삼성전자+주가")

# html = urllib.request.urlopen(url).read()
# soup = BeautifulSoup(html, "html.parser")

# news = soup.find_all("div", attrs={"class": "info_group"})
# for item in news:
#     press = item.find("a", attrs={"class": "info press"}).get_text()
#     news_url = item.find("a").get("href")
#     print(press)
#     print(news_url)
# # sp_nws1 > div.news_wrap.api_ani_send > div > div.news_info > div.info_group > a:nth-child(3)
# print(len(news))
