from bs4 import BeautifulSoup
from lxml import etree
from selenium import webdriver
from selenium.webdriver.common.by import By
from pathlib import Path


with open("./xpath_ex.html", "r") as f:
    soup = BeautifulSoup(f.read(), "html.parser")

dom = etree.HTML(str(soup))
print(dom.xpath("/html/body/h1")[0].text)

options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome("chromedriver", options=options)
driver.implicitly_wait(time_to_wait=10)

file_url = Path.cwd() / "xpath_ex.html"
driver.get(file_url.as_uri())

elem = driver.find_element(by=By.XPATH, value="/html/body/h1")
print(elem.text)
elem = driver.find_element(by=By.XPATH, value="/html/body/div/span[@class='regular_price']")
print(elem.text)
elem = driver.find_element(by=By.XPATH, value="//span[@class='regular_price']")
print(elem.text)
elem = driver.find_element(by=By.XPATH, value="//img[contains(@class, 'image')]")
print(elem.text)
elem = driver.find_element(by=By.XPATH, value="//div[contains(text(), '상품번호')]")
print(elem.text)
# elem = driver.find_element(by=By.XPATH, value="//script[contains(text(), 'stock')]")
# print(elem.text)
elem = driver.find_element(by=By.XPATH, value="//li[position()=3]")
print(elem.text)
elem = driver.find_element(by=By.XPATH, value="//li[2]")
print(elem.text)
elems = driver.find_elements(by=By.XPATH, value="//li[position()>0]")
for elem in elems:
    print(elem.text)
elem = driver.find_element(by=By.XPATH, value="//img[contains(@src, 'main')]")
print(elem.text)
# elem = driver.find_element(by=By.XPATH, value="")
# print(elem.text)
# elem = driver.find_element(by=By.XPATH, value="")
# print(elem.text)
# elem = driver.find_element(by=By.XPATH, value="")
# print(elem.text)
# elem = driver.find_element(by=By.XPATH, value="")
# print(elem.text)
