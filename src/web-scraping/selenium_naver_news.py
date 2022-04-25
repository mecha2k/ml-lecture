import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import time
import os

import soupsieve

print(selenium.__version__)

load_dotenv(verbose=True)
ID = os.getenv("naverID")
PASS = os.getenv("naverPASS")
print(ID)

options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome("chromedriver", options=options)
driver.implicitly_wait(time_to_wait=10)
# driver.maximize_window()

driver.get("https://www.naver.com/")

# driver.get_screenshot_as_file("naver.png")
# driver.find_element(by=By.CLASS_NAME, value="link_login").click()

# driver.find_element(by=By.ID, value="id").send_keys(ID)
# driver.find_element(by=By.ID, value="pw").send_keys(PASS)
# print(driver.page_source)

# driver.find_element(by=By.ID, value="id").send_keys("ID")
# driver.find_element(by=By.ID, value="pw").send_keys("PASS")

# driver.find_element(by=By.ID, value="id").clear()
# driver.find_element(by=By.ID, value="id").send_keys("newID")

# driver.back()
# driver.forward()
# driver.refresh()

elem = driver.find_element(by=By.ID, value="query")
elem.send_keys("삼성전자 주가")
elem.send_keys(Keys.ENTER)
time.sleep(0.5)

html = driver.page_source
soup = BeautifulSoup(html, "html.parser")
items = soup.find_all("a", class_="news_tit")
for item in items:
    print(item.text)

elem = driver.find_element(by=By.TAG_NAME, value="a")
print(elem.get_attribute("href"))
elem = driver.find_element(by=By.CLASS_NAME, value="list_news")
print(elem.text)


driver.quit()
