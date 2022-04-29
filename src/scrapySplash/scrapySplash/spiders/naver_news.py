# import scrapy
# from scrapy_01.items import Scrapy01Item


# class NewsSpider(scrapy.Spider):
#     name = "news"
#     allowed_domains = ["web"]
#     start_urls = [
#         "https://search.naver.com/search.naver?where=news&ie=utf8&sm=nws_hty&query=삼성전자+주가"
#     ]

#     def parse(self, response):
#         items = Scrapy01Item()
#         list_news = response.css("ul.list_news > li")
#         for news in list_news:
#             title = news.css(".news_area > a::text").extract()
#             title = " ".join(title)
#             items["title"] = title
#             self.log(f"title: {title}")
#             yield items
