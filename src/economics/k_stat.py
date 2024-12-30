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

df_k_stat = pd.read_excel("k-stat_export.xlsx", sheet_name="trade", skiprows=0, index_col=0)
df_k_stat.index = pd.to_datetime(df_k_stat.index)
print(df_k_stat.info())

df_kospi = pd.read_csv("kospi.csv", index_col="Date", parse_dates=True)
print(df_kospi.info())

start = datetime(2018, 1, 1)
end = datetime(2026, 1, 1)

df_k_stat["ex_annual"] = df_k_stat["export"].pct_change(periods=12, fill_method=None) * 100
df_k_stat["im_annual"] = df_k_stat["import"].pct_change(periods=12, fill_method=None) * 100


fig, ax1 = plt.subplots(figsize=(12, 6))
color = "orangered"
ax1.set_ylabel("Export (YoY%)", color=color)
ax1.plot(df_k_stat["ex_annual"], color=color, label="Export (YoY%)", linewidth=2)
ax1.tick_params(axis="y", colors=color)

ax2 = ax1.twinx()
color = "blue"
ax2.set_ylabel("Kospi Index", color=color)
ax2.plot(df_kospi.Close, color=color, label="Kospi", linewidth=2)
ax2.tick_params(axis="y", colors=color)

ax3 = ax1.twinx()
color = "green"
ax3.plot(df_cli["Korea"], color=color, label="OECD CLI_Korea", linewidth=3)
ax3.spines["right"].set_position(("outward", 60))
ax3.set_ylabel("OECD CLI_Korea", color=color)
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
ax1.grid(True, which="both", axis="both", linestyle="--", color="gray", alpha=0.5)
ax1.set_xlim(start, end)
ax1.set_ylim(-40, 80)
ax2.set_ylim(1000, 4000)
ax3.set_ylim(94, 106)

plt.tight_layout()
plt.savefig("KOSPI_export_yoy.png", dpi=300)
