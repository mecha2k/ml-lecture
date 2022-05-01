from asyncio import events
import requests
from bs4 import BeautifulSoup

url = "https://www.python.org/events/python-events/"
response = requests.get(url)

soup = BeautifulSoup(response.text, "html.parser")
print(soup.find("ul", class_="list-recent-events"))
events = soup.find("ul", class_="list-recent-events").find_all("li")

for event in events:
    details = event.find("h3").find("a").text
    location = event.find("span", class_="event-location").text
    time = event.find("time").text
    print(details, "|", location, "|", time)
