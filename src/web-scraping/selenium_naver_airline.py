from matplotlib.pyplot import cla
import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup

from dotenv import load_dotenv
import time
import os

print(selenium.__version__)

load_dotenv(verbose=True)
ID = os.getenv("naverID")
PASS = os.getenv("naverPASS")
print(ID)

options = webdriver.ChromeOptions()
# options.add_argument("--headless")
# options.add_argument("--no-sandbox")
# options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome("chromedriver", options=options)
driver.implicitly_wait(time_to_wait=10)

driver.get("https://flight.naver.com/")

driver.find_element(
    by=By.XPATH, value="//*[@id='__next']/div/div[1]/div[4]/div/div/div[2]/div[1]/button[1]"
).click()
time.sleep(0.2)
driver.find_element(
    by=By.XPATH, value="//*[@id='__next']/div/div[1]/div[9]/div[1]/div/input"
).send_keys("GMP")
time.sleep(0.2)
driver.find_element(
    by=By.XPATH, value="//*[@id='__next']/div/div[1]/div[9]/div[2]/section/div"
).click()
time.sleep(0.2)

driver.find_element(
    by=By.XPATH, value="//*[@id='__next']/div/div[1]/div[4]/div/div/div[2]/div[1]/button[2]"
).click()
time.sleep(0.2)
driver.find_element(
    by=By.XPATH, value="//*[@id='__next']/div/div[1]/div[9]/div[1]/div/input"
).send_keys("CJU")
time.sleep(0.2)
driver.find_element(
    by=By.XPATH, value="//*[@id='__next']/div/div[1]/div[9]/div[2]/section/div"
).click()
time.sleep(0.2)

driver.find_element(
    by=By.XPATH, value="//*[@id='__next']/div/div[1]/div[4]/div/div/div[2]/div[2]/button[1]"
).click()
time.sleep(0.2)

driver.find_elements(
    by=By.XPATH,
    value="//*[@id='__next']/div/div[1]/div[9]/div[2]/div[1]/div[2]/div/div[2]/table/tbody/tr[5]/td[6]/button",
)[0].click()
time.sleep(0.2)

driver.find_element(
    by=By.XPATH, value="//*[@id='__next']/div/div[1]/div[4]/div/div/div[2]/div[2]/button[2]"
).click()
time.sleep(0.2)

driver.find_elements(
    by=By.XPATH,
    value="//*[@id='__next']/div/div[1]/div[9]/div[2]/div[1]/div[2]/div/div[3]/table/tbody/tr[1]/td[5]/button",
)[0].click()
time.sleep(0.2)


driver.find_element(by=By.XPATH, value="//*[@id='__next']/div/div[1]/div[4]/div/div/button").click()
time.sleep(10)

for i in range(1):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
    time.sleep(0.2)

url = driver.page_source
soup = BeautifulSoup(url, "html.parser")

flights = soup.find_all(class_="result")
for flight in flights:
    print(flight.text.split())


driver.quit()
