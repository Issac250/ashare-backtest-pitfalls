"""08 算术年化高估复利，以及「一年 252 个交易日」。

Part 1. Annualising the MEAN daily return, (1 + mean)^252 - 1, ignores the
variance drag (波动拖累): a -50% day followed by a +50% day averages 0% but
loses 25%. On a volatile small-cap-like series the gap is several points a year.

Part 2. A-share years have about 242-244 trading days, not 252 (the exchange
closes for the Spring Festival and National Day weeks, among others). Counting
years as trading days / 252 makes the sample look shorter than it was, which
overstates CAGR whenever the total return is positive.
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
