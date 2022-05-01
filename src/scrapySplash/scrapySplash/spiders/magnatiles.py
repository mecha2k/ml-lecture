import scrapy


class TilesSpider(scrapy.Spider):
    name = "magnatiles"
    allowed_domains = ["magnatiles.com"]
    start_urls = ["https://www.magnatiles.com/products/page/1/"]

    def parse(self, response, **kwargs):
        products = response.css("ul.products li")
        for product in products:
            yield {
                "name": product.css("h2::text").get(),
                "sku": product.css("a.button::attr(data-product_sku)").get(),
                "price": product.css("span.price bdi::text").get(),
            }
