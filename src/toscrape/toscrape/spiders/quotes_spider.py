import scrapy


class QuotesSpider(scrapy.Spider):
    name = "quotes"

    def start_requests(self):
        urls = [
            "http://quotes.toscrape.com/page/1/",
            # "http://quotes.toscrape.com/page/2/",
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    # start_urls = [
    #     "http://quotes.toscrape.com/page/1/",
    #     "http://quotes.toscrape.com/page/2/",
    # ]

    def parse(self, response):
        page = response.url.split("/")[-2]
        filename = f"quotes-{page}.html"
        with open(filename, "wb") as f:
            f.write(response.body)
        self.log(f"Saved file {filename}")

        print(response.url)
        print(response.url.split("/"))
        print("-------------------------------------------------------------------------------")

        for quote in response.css("div.quote"):
            text = quote.css("span.text::text").get()[:10]
            author = quote.css("small.author::text").get()
            tags = quote.css("div.tags a.tag::text").getall()
            quote_dict = dict(text=text, author=author, tags=tags)
            yield quote_dict

        next_page = response.css("li.next a::attr(href)").get()
        if next_page is not None:
            # next_page = response.urljoin(next_page)
            # yield scrapy.Request(next_page, callback=self.parse)
            yield response.follow(next_page, callback=self.parse)
