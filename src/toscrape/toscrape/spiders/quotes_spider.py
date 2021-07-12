import scrapy
from toscrape.items import ToscrapeItem


class QuotesSpider(scrapy.Spider):
    name = "quotes"

    def start_requests(self):
        urls = [
            "http://quotes.toscrape.com/page/1/",
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    # start_urls = [
    #     "http://quotes.toscrape.com/page/1/",
    #     "http://quotes.toscrape.com/page/2/",
    # ]

    def parse(self, response):
        page = int(response.url.split("/")[-2])
        filename = f"results/quotes-{page:02d}.html"
        with open(filename, "wb") as f:
            f.write(response.body)
        self.log(f"Saved file {filename}")
        print(response.url)
        print(response.url.split("/"))
        print("-------------------------------------------------------------------------------")

        item = ToscrapeItem()
        for quote in response.css("div.quote"):
            item["text"] = quote.css("span.text::text").get()[:10]
            item["author"] = quote.css("small.author::text").get()
            item["tags"] = quote.css("div.tags a.tag::text").getall()
            # quote_dict = dict(text=text, author=author, tags=tags)
            yield item

        # next_page = response.css("li.next a::attr(href)").get()
        # if next_page is not None:
        #     # next_page = response.urljoin(next_page)
        #     # yield scrapy.Request(next_page, callback=self.parse)
        #     yield response.follow(next_page, callback=self.parse)
