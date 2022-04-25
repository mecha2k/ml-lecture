import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import ui


print(selenium.__version__)

options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome("chromedriver", options=options)
print(driver)

url = "https://the-internet.herokuapp.com/dynamic_loading/2"
driver.get(url)

driver.find_element(by=By.CSS_SELECTOR, value="#start > button").click()
print("button clicked!")

wait = ui.WebDriverWait(driver, 10)
wait.until(lambda driver: driver.find_element(by=By.CSS_SELECTOR, value="#finish"))

finish = driver.find_element(by=By.CSS_SELECTOR, value="#finish > h4")
print(finish.text)