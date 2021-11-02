import pandas as pd
from datetime import datetime

start = datetime(2018,1,1)
end = datetime(2020,12,31)

for tm in pd.date_range(start=start, end=end, freq="A"):
    print(tm.year)