import scrapy


class BeerSpider(scrapy.Spider):
    name = "beer"

    def start_requests(self):
        url = "https://www.beerwulf.com/en-gb/c/mixedbeercases"

        yield scrapy.Request(url=url, method="GET", callback=self.parse)

    def parse(self, response):
        products = response.css("a.product.search-product.draught-product.notranslate.pack-product")

        for product in products:
            yield {
                "name": product.css("h4::text").get(),
                "price": product.css("span.price::text").get(),
            }

