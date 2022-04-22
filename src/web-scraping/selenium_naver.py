import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from dotenv import load_dotenv
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
driver.maximize_window()

driver.get("https://flight.naver.com/")

elem = driver.find_element(by=By.CLASS_NAME, value="flights")
print(elem)
# driver.find_element(by=By.CLASS_NAME, value="tabContent_option__2y4c6 select_Date__1aF7Y").click()

# driver.find_elements(by=By.LINK_TEXT, value="28")[0].click()
# driver.find_elements(by=By.LINK_TEXT, value="5")[1].click()


# driver.get("https://www.naver.com/")
# driver.get_screenshot_as_file("naver.png")
#
# driver.find_element(by=By.CLASS_NAME, value="link_login").click()
#
# driver.find_element(by=By.ID, value="id").send_keys(ID)
# driver.find_element(by=By.ID, value="pw").send_keys(PASS)
# print(driver.page_source)

# driver.find_element(by=By.ID, value="id").send_keys("ID")
# driver.find_element(by=By.ID, value="pw").send_keys("PASS")
#
# driver.find_element(by=By.ID, value="id").clear()
# driver.find_element(by=By.ID, value="id").send_keys("newID")


# driver.back()
# driver.forward()
# driver.refresh()

# elem = driver.find_element(by=By.ID, value="query")
# print(elem)
#
# elem.send_keys("삼성전자 주가")
# elem.send_keys(Keys.ENTER)
#
# elem = driver.find_element(by=By.TAG_NAME, value="a")
# print(elem.get_attribute("href"))
#
# elem = driver.find_element(by=By.CLASS_NAME, value="list_news")
# print(elem)


driver.quit()
