import pandas as pd
import pandas_profiling as pp

flights = pd.read_csv("data/flights.csv")
profile = pp.ProfileReport(
    flights,
    title="Pandas Profiling",
    minimal=True,
    correlations={"kendall": {"calculate": False}, "cramers": {"calculate": False}},
)
profile.to_file("data/ch09_flights.html")
