import numpy as np
import pandas as pd
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from datetime import datetime
from pathlib import Path


sns.set_style("whitegrid")
sns.set_palette("cubehelix")
plt.rcParams["font.size"] = 12

df_cli = pd.read_csv("oecd_cli.csv", index_col=0, parse_dates=True, skiprows=2)
print(df_cli.info())

kospi_file = Path("kospi.csv")
df_kospi = fdr.DataReader("KS11")
df_kospi.to_csv("kospi.csv")

df_kospi = pd.read_csv("kospi.csv", index_col="Date", parse_dates=True)
print(df_kospi.info())

start = datetime(2018, 1, 1)
end = datetime(2026, 1, 1)

df_usdkrw = fdr.DataReader("USD/KRW", start, end)
print(df_usdkrw.info())

fig, ax1 = plt.subplots(figsize=(12, 6))
color = ["tab:green", "gold", "orangered"]
ax1.set_ylabel("OECD CLI")
for i, col in enumerate(df_cli.columns):
    lw = 3 if col == "Korea" else 1
    ax1.plot(
        df_cli.index,
        df_cli[col],
        color=color[i],
        linewidth=lw,
        label=f"OECD CLI_{col}",
    )
ax1.tick_params(axis="y")

ax2 = ax1.twinx()
color = "blue"
ax2.set_ylabel("Kospi Index", color=color)
ax2.plot(df_kospi.Close, color=color, label="Kospi", linewidth=2)
ax2.tick_params(axis="y", labelcolor=color)

ax3 = ax1.twinx()
color = "dodgerblue"
ax3.plot(df_usdkrw["Close"], color=color, label="USD/KRW")
ax3.spines["right"].set_position(("outward", 60))
ax3.set_ylabel("USD/KRW", color=color)
ax3.tick_params(axis="y", colors=color)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines3, labels3 = ax3.get_legend_handles_labels()
lines = lines1 + lines2 + lines3
labels = labels1 + labels2 + labels3
plt.legend(lines, labels, loc="best")

plt.gcf().autofmt_xdate()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.gca().xaxis.set_major_locator(mdates.YearLocator(base=1))
ax1.grid(
    True, which="both", axis="both", linestyle="--", color="gray", alpha=0.5
)
ax1.set_xlim(start, end)
ax1.set_ylim(94, 104)
ax2.set_ylim(1000, 4000)
ax3.set_ylim(1000, 1500)

ax2.set_yticks(
    np.linspace(
        ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax1.get_yticks())
    )
)
plt.tight_layout()
plt.savefig("KOSPI_OECD_fx.png", dpi=300)
