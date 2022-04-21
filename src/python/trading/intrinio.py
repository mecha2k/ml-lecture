from __future__ import print_function
import time
import intrinio_sdk as intrinio
from intrinio_sdk.rest import ApiException
from pprint import pprint
from dotenv import load_dotenv
import os
import requests

load_dotenv(verbose=True)
api_key = os.getenv("Intrinio")
print(api_key)

intrinio.ApiClient().configuration.api_key["api_key"] = api_key
intrinio.ApiClient().allow_retries(True)

security_api = intrinio.SecurityApi()

identifier = "AAPL"  # str | A Security identifier (Ticker, FIGI, ISIN, CUSIP, Intrinio ID)
start_date = "2019-01-02"  # date | Return intraday prices starting at the specified date (optional)
end_date = "2019-01-04"  # date | Return intraday prices stopping at the specified date (optional)

try:
    api_response = security_api.get_security_intraday_prices(
        identifier, start_date=start_date, end_date=end_date
    )
    pprint(api_response.intraday_prices)
except ApiException as e:
    print("Exception when calling SecurityApi->get_security_intraday_prices: %s\n" % e)

# Note: For a Pandas DataFrame, import Pandas and use pd.DataFrame(api_response.intraday_prices_dict)

# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
api_key = os.getenv("Alpha_vantage")
url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=AAPL&interval=5min&apikey={api_key}"
res = requests.get(url)
pprint(res.json())
