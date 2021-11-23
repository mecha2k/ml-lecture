import pandas as pd

df = pd.DataFrame(
    {
        "A": ["foo", "bar", "foo", "bar", "foo", "bar"],
        "B": [1, 2, 3, 4, 5, 6],
        "C": [2.0, 5.0, 8.0, 1.0, 2.0, 9.0],
    }
)
print(df)

grouped = df.groupby("A")
print(grouped)
# df1 = grouped.filter(lambda x: x["B"].mean() > 3.0)
# df1 = grouped.filter(lambda x: x["B"].mean() > 3.0)
df1 = grouped.filter(lambda x: x["B"].quantile() > 3.0)
print(df1)
