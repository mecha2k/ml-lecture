import scrapy


class WhiskySpider(scrapy.Spider):
    name = "whisky"
    start_urls = ["https://www.whiskyshop.com/scotch-whisky/all"]

    def parse(self, response):
        for res in response.css("div.product-item-info"):
            try:
                yield {
                    "name": res.css("a.product-item-link::text").get(),
                    "price": res.css("span.price::text").get().replace("£", ""),
                    "link": res.css("a.product-item-link").attrib["href"],
                }
            except AttributeError:
                yield res

        next_page = response.css("a.action.next").attrib["href"]
        if next_page is not None:
            yield response.follow(next_page, callbacks=self.parse)
