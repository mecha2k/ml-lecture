from bs4 import BeautifulSoup
import requests

url = "https://search.naver.com/search.naver?where=news&sm=tab_jum&query=삼성전자+주가"

html = requests.get(url)
print(html.status_code)

soup = BeautifulSoup(html.text, "lxml")

with open("soup.txt", "w") as f:
    f.write(soup.text)
f.close()

# items = soup.find_all("li", class_="bx")
# for item in items:
#     press = item.find("span", class_="thumb_box")
#     print(press.text)

