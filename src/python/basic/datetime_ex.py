import pandas as pd
import re
from datetime import datetime

start = datetime(2018, 1, 1)
end = datetime(2020, 12, 31)

for tm in pd.date_range(start=start, end=end, freq="A"):
    print(tm.year)

students = [
    ("jack", 34, "Sydeny", "Australia"),
    ("Riti", 30, "Delhi", "India"),
    ("Vikas", 31, "Mumbai", "India"),
    ("Neelu", 32, "Bangalore", "India"),
    ("John", 16, "New York", "US"),
    ("Mike", 17, "las vegas", "US"),
]
df = pd.DataFrame(
    students, columns=["Name", "Age", "City", "Country"], index=["a", "b", "c", "d", "e", "f"]
)
print(df)
# mod_df = df.append({"Name": "Sahil", "Age": 22}, ignore_index=True)
mod_df = pd.DataFrame({"Name": ["Sahil"], "Age": [22]})
mod_df = pd.concat([df, mod_df], ignore_index=True)
print(mod_df)

start_str = start.strftime("%Y%m%d")
print(start_str)

ticker = "A005930"
ticker = re.findall(r"\d+", ticker)
print(ticker[0])

# Return the current time in UTC
print(datetime.utcnow())

def alculateDaysBetweenDates(begin, end):
    return (end - begin).days