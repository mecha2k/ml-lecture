import scrapy
from scrapy_splash import SplashRequest


class BeerSpider(scrapy.Spider):
    name = "beer"

    def start_requests(self):
        url = "https://www.beerwulf.com/en-gb/c/mixedbeercases"

        yield SplashRequest(url=url, method="GET", callback=self.parse)

    def parse(self, response):
        products = response.css("a.product.search-product.draught-product.notranslate.pack-product")

        for product in products:
            yield {
                "name": product.css("h4::text").get(),
                "price": product.css("span.price::text").get(),
            }
            # name = product.css("h4::text").get()
            # self.log(name)
