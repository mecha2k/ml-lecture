import scrapy
from scrapy_splash import SplashRequest
from bs4 import BeautifulSoup


class AmazonSpider(scrapy.Spider):
    name = "amazon"
    allowed_domains = ["www.amazon.com"]

    def start_requests(self):
        url = "https://www.amazon.com/s?bbn=283155&rh=n%3A283155%2Cp_n_publication_date%3A1250226011&dc&qid=1651455525&rnid=1250225011&ref=lp_1000_nr_p_n_publication_date_0"

        yield SplashRequest(url=url, method="GET", callback=self.parse)
        # yield scrapy.Request(url=url, method="GET", callback=self.parse)

    def parse(self, response):
        title = response.css("title::text").get()
        self.log(title)

        # soup = BeautifulSoup(response.text, features="html.parser")
        # print(soup.prettify())
