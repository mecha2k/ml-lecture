import scrapy
from scrapy_splash import SplashRequest


class QuotesSpider(scrapy.Spider):
    name = "quotes_splash"

    def start_requests(self):
        urls = [
            "http://quotes.toscrape.com/page/1/",
        ]
        for url in urls:
            yield SplashRequest(
                url=url,
                callback=self.parse,
                args={"wait": 0.5},
            )

    def parse(self, response):
        page = int(response.url.split("/")[-2])
        filename = f"results/quotes-{page:02d}.html"
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
            yield response.follow(next_page, callback=self.parse)
