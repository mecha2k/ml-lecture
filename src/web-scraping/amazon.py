from nturl2path import url2pathname
from more_itertools import first
from requests import session
from requests_html import HTMLSession

url = "https://www.amazon.com/s?bbn=283155&rh=n%3A283155%2Cp_n_publication_date%3A1250226011&dc&qid=1651455525&rnid=1250225011&ref=lp_1000_nr_p_n_publication_date_0"

session = HTMLSession()
response = session.get(url=url)
response.html.render(sleep=1)

items = {
    "title": response.html.find("#productTitle", first=True).text,

    # "title": response.html.xpath(
    #     "//*[@id='search']/div[1]/div[1]/div/span[3]/div[2]/div[2]/div/div/div/div/div/div[2]/div/div/div[1]/h2/a/span",
    #     first=True,
    # ).text,
    # "price": response.html.xpath(
    #     "//*[@id='search']/div[1]/div[1]/div/span[3]/div[2]/div[2]/div/div/div/div/div/div[2]/div/div/div[3]/div[1]/div/div[1]/div[2]/a/span[2]/span[2]/span[2]",
    #     first=True,
    # ).text,
}
print(items)
