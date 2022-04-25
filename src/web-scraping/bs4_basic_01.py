from bs4 import BeautifulSoup
import urllib.request

with open("./example.html", "r") as f:
    soup = BeautifulSoup(markup=f, features="html.parser")
print(soup)

print(soup.div)
print(soup.title)
print(soup.title.name)
print(soup.title.string)
print(soup.section)
print(soup.h2.string)
print(soup.title.parent)
print(soup.title.parent.meta)

print(soup.find("div"))
print(soup.find_all("div"))

print(soup.find_all(name="p", attrs={"id": "para1"}))
print(soup.find_all(name="div", attrs={"class": "someclass"}))

print(soup.find("a").get(key="href"))
print(soup.find("a").get_text())

names = soup.find_all(name="a")
for name in names:
    print(name.get_text())
    print(name.get(key="href"))

print(soup.select("a")[0].get_text())
print(soup.select("p#para1"))
print(soup.select("div.someclass")[0].get_text())


html = urllib.request.urlopen("http://suanlab.com").read()
soup = BeautifulSoup(html, features="html.parser")

labels = soup.find_all(["label"])
for label in labels:
    print(label.get_text())

labels = soup.select("#wrapper > section > div > div > div > div > div > label")
for label in labels:
    print(label.get_text())
