import datetime
import pytz

print(datetime.datetime.now())
print(datetime.datetime.now(tz=pytz.utc))
print(datetime.datetime.now(tz=pytz.timezone("Europe/Vienna")))

current = "2022-05-02 08:12:24"
current = datetime.datetime.strptime(current, "%Y-%m-%d %H:%M:%S")
print(current)

# print(pytz.all_timezones)
print(pytz.common_timezones)
print(pytz.country_timezones["KR"])

src_timezone = pytz.timezone("US/Eastern")
tar_timezone = pytz.timezone("Europe/Vienna")

newyork_time = src_timezone.localize(current)
vienna_time = newyork_time.astimezone(tz=tar_timezone)
print(newyork_time)
print(vienna_time)

loc_timezone = pytz.timezone("Asia/Seoul")
seoul_time = loc_timezone.localize(current)
print(seoul_time)
print(seoul_time.tzname())
print(seoul_time.utcoffset())
print(seoul_time.dst())
