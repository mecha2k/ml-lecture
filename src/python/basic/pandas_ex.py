import pandas as pd

df1 = pd.DataFrame(
    {
        "A": ["A0", "A1", "A2", "A3"],
        "B": ["B0", "B1", "B2", "B3"],
        "C": ["C0", "C1", "C2", "C3"],
        "D": ["D0", "D1", "D2", "D3"],
    },
    index=[0, 1, 2, 3],
)
df2 = pd.DataFrame(
    {
        "A": ["A4", "A5", "A6", "A7"],
        "B": ["B4", "B5", "B6", "B7"],
        "C": ["C4", "C5", "C6", "C7"],
        "D": ["D4", "D5", "D6", "D7"],
    },
    index=[4, 5, 6, 7],
)
df3 = pd.DataFrame(
    {
        "A": ["A8", "A9", "A10", "A11"],
        "B": ["B8", "B9", "B10", "B11"],
        "C": ["C8", "C9", "C10", "C11"],
        "D": ["D8", "D9", "D10", "D11"],
    },
    index=[8, 9, 10, 11],
)

print(pd.concat([df1, df2, df3]))
print(pd.concat([df1, df2, df3], keys=["x", "y", "z"]))
print(pd.concat([df1, df2, df3], keys=["x", "y", "z"]).loc["y"])
print(pd.concat([df1, df2, df3], keys=["x", "y", "z"]).loc[("y", 4)])

df4 = pd.DataFrame(
    {
        "B": ["B2", "B3", "B6", "B7"],
        "D": ["D2", "D3", "D6", "D7"],
        "F": ["F2", "F3", "F6", "F7"],
    },
    index=[2, 3, 6, 7],
)
print(df1)
print(df4)
print(pd.concat([df1, df4], axis=1))
print(pd.concat([df1, df4], axis=1, join="inner"))

print(df1.append(df2))
print(pd.concat([df1, df4.reindex(df1.index)], axis=1))

left = pd.DataFrame(
    {
        "key1": ["K0", "K0", "K1", "K2"],
        "key2": ["K0", "K1", "K0", "K1"],
        "A": ["A0", "A1", "A2", "A3"],
        "B": ["B0", "B1", "B2", "B3"],
    }
)
right = pd.DataFrame(
    {
        "key1": ["K0", "K1", "K1", "K2"],
        "key2": ["K0", "K0", "K0", "K0"],
        "C": ["C0", "C1", "C2", "C3"],
        "D": ["D0", "D1", "D2", "D3"],
    }
)
print(left)
print(right)
print(pd.merge(left, right, on=["key1", "key2"]))
print(pd.merge(left, right, how="left", on=["key1", "key2"]))
print(pd.merge(left, right, how="outer", on=["key1", "key2"]))

left = pd.DataFrame({"A": ["A0", "A1", "A2"], "B": ["B0", "B1", "B2"]}, index=["K0", "K1", "K2"])
right = pd.DataFrame({"C": ["C0", "C2", "C3"], "D": ["D0", "D2", "D3"]}, index=["K0", "K2", "K3"])
print(left)
print(right)
print(left.join(right))
print(left.join(right, how="outer"))
print(left.join(right, how="inner"))
print(pd.merge(left, right, left_index=True, right_index=True, how="outer"))
print(pd.merge(left, right, left_index=True, right_index=True, how="inner"))

leftindex = pd.MultiIndex.from_product(
    [list("abc"), list("xy"), [1, 2]], names=["abc", "xy", "num"]
)
left = pd.DataFrame({"v1": range(12)}, index=leftindex)

rightindex = pd.MultiIndex.from_product([list("abc"), list("xy")], names=["abc", "xy"])
right = pd.DataFrame({"v2": [100 * i for i in range(1, 7)]}, index=rightindex)

print(left)
print(right)
print(left.join(right, on=["abc", "xy"], how="inner"))

trades = pd.DataFrame(
    {
        "time": pd.to_datetime(
            [
                "20160525 13:30:00.023",
                "20160525 13:30:00.038",
                "20160525 13:30:00.048",
                "20160525 13:30:00.048",
                "20160525 13:30:00.048",
            ]
        ),
        "ticker": ["MSFT", "MSFT", "GOOG", "GOOG", "AAPL"],
        "price": [51.95, 51.95, 720.77, 720.92, 98.00],
        "quantity": [75, 155, 100, 100, 100],
    },
    columns=["time", "ticker", "price", "quantity"],
)
quotes = pd.DataFrame(
    {
        "time": pd.to_datetime(
            [
                "20160525 13:30:00.023",
                "20160525 13:30:00.023",
                "20160525 13:30:00.030",
                "20160525 13:30:00.041",
                "20160525 13:30:00.048",
                "20160525 13:30:00.049",
                "20160525 13:30:00.072",
                "20160525 13:30:00.075",
            ]
        ),
        "ticker": ["GOOG", "MSFT", "MSFT", "MSFT", "GOOG", "AAPL", "GOOG", "MSFT"],
        "bid": [720.50, 51.95, 51.97, 51.99, 720.50, 97.99, 720.50, 52.01],
        "ask": [720.93, 51.96, 51.98, 52.00, 720.93, 98.01, 720.88, 52.03],
    },
    columns=["time", "ticker", "bid", "ask"],
)
print(trades)
print(quotes)
print(pd.merge_asof(trades, quotes, on="time", by="ticker"))
print(pd.merge_asof(trades, quotes, on="time", by="ticker", tolerance=pd.Timedelta("2ms")))
print(
    pd.merge_asof(
        trades,
        quotes,
        on="time",
        by="ticker",
        tolerance=pd.Timedelta("10ms"),
        allow_exact_matches=False,
    )
)
