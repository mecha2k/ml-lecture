from matplotlib.pyplot import cla
import scrapy


class NaverMovie(scrapy.Spider):
    name = "naver_movie"
    allowed_domains = [""]
    start_urls = ["https://movie.naver.com/movie/running/current.naver"]

    def parse(self, response):
        movies = response.css("ul.lst_detail_t1 > li")
        for movie in movies:
            title = movie.css(".tit > a::text").get()
            print(title)
