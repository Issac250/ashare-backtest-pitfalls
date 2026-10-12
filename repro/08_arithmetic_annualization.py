"""08 对同一份日收益采用三种年化口径。

比较平均日收益推算值、按日历时间计算的 CAGR，
以及按交易日数 / 252 计算年数的 CAGR。交易日历是简化设定。
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(9)

# An A-share-like calendar: weekdays minus approximate public-holiday closures.
days = pd.bdate_range("2019-01-02", "2023-12-29")
ym = pd.Series(days.year * 100 + days.month, index=days)
nth_weekday = ym.groupby(ym).cumcount().to_numpy()       # 0 = first weekday of the month
closed = (
    ((days.month == 2) & (nth_weekday < 5))                # Spring Festival week (approximate)
    | ((days.month == 10) & (nth_weekday < 5))             # National Day week
    | ((days.month == 5) & (nth_weekday < 3))              # Labour Day
    | ((days.month == 4) & (nth_weekday == 3))             # Qingming
    | ((days.month == 6) & (nth_weekday == 9))             # Dragon Boat
    | ((days.month == 9) & (nth_weekday == 12))            # Mid-Autumn
)
days = days[~closed]
per_year = pd.Series(days.year).value_counts().sort_index()
print("trading days per year:", {int(k): int(v) for k, v in per_year.items()})

r = pd.Series(rng.normal(0.0008, 0.022, len(days)), index=days)   # ~35% annual volatility
equity = (1 + r).cumprod()
total = equity.iloc[-1] / 1.0
calendar_years = (days[-1] - days[0]).days / 365.25

arith = (1 + r.mean()) ** 252 - 1
cagr = total ** (1 / calendar_years) - 1
cagr_252 = total ** (252 / len(days)) - 1

print(f"annual volatility: {r.std() * np.sqrt(252):.1%}")
print(f"(1 + mean daily)^252 - 1   (wrong) : {arith:+.1%}")
print(f"compound, calendar years   (right) : {cagr:+.1%}")
print(f"compound, years = days/252 (wrong) : {cagr_252:+.1%}   ({len(days)} days -> "
      f"{len(days) / 252:.2f} 'years', calendar says {calendar_years:.2f})")
assert arith > cagr + 0.03, "the arithmetic annualisation should overstate by the variance drag"
assert cagr_252 > cagr, "a 252-day year shortens the sample and overstates a positive CAGR"
