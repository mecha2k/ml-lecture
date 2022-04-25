import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service


print(selenium.__version__)

options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
# driver = webdriver.Chrome(service=Service("./chromedriver.exe"), options=options)

driver = webdriver.Chrome(executable_path="./chromedriver.exe", options=options)


driver.get("http://suanlab.com/")
driver.get_screenshot_as_file("suanlab.png")

# for label in driver.find_elements(By.TAG_NAME, "label"):
#     print(label.text)

css_selector = "#wrapper > section > div > div > div > div > div > label"
for label in driver.find_elements(By.CSS_SELECTOR, css_selector):
    print(label.text)

for label in driver.find_elements(By.CLASS_NAME, "toggle toggle-transparent toggle-bordered-full"):
    print(label.text)
