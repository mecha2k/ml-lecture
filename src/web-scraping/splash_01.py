from urllib import response
import requests
from bs4 import BeautifulSoup

url = "https://www.amazon.com/s?k=canon+50d&crid=2OUJEL41DVUE&sprefix"
url += "=canon+50d%2Caps%2C355&ref=nb_sb_noss_1"

# response = requests.get(url)
# print(response.status_code)
# print(response.text)


response = requests.get("http://localhost:8050/render.html", params={"url": url, "wait": 2})
soup = BeautifulSoup(response.text, "html.parser")
print(soup.title.text)
