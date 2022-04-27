import scrapy


class HemnetSpider(scrapy.Spider):
    name = "hemnet"
    start_urls = ["https://www.hemnet.se/bostader?item_types%5B%5D=villa"]

    def parse(self, response):
        for res in response.css("ul.normal-results > normal-results__hit > a::attr('href')"):
            yield scrapy.Request(url=res.get(), callback=self.parseInner)

        nextPage = response.css("a.next_page::attr('href')").get()
        if nextPage is not None:
            response.follow(nextPage, callback=self.parse)

    def parseInnerPage(self, response):
        streetName = response.css("h1.property-address__street::text").get()
        price = response.css("p.property-info__price::text").get()
