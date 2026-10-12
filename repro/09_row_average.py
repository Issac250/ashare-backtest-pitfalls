"""09 逐行平均不是组合收益，而且能把结论的符号弄反。

A rule picks stocks every day; a quick study averages the 20-session forward excess
return over all (date, code) rows. A portfolio does something else: it gives
every DAY the same weight, however many names qualified that day.

Synthetic assumption: the rule qualifies many names in rising markets (where
they beat the market a little) and few in falling markets (where they lag a
lot). Row-weighting lets the crowded good days drown out the sparse bad ones.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(10)
T = 1500
up = np.repeat(rng.random(T // 50) < 0.6, 50)            # 50-day regimes, 60% of them rising
rows = []
for t in range(T):
    if up[t]:
        k, mu = int(rng.integers(100, 200)), 0.005         # many qualify, +0.5% excess
    else:
        k, mu = int(rng.integers(5, 30)), -0.015           # few qualify, -1.5% excess
    rows.append(pd.DataFrame({"date": t, "excess": rng.normal(mu, 0.05, k)}))
panel = pd.concat(rows, ignore_index=True)

wrong = panel["excess"].mean()                            # every row weighs the same
by_day = panel.groupby("date")["excess"].mean()
right = by_day.mean()                                     # every day weighs the same (the book)
counts = panel.groupby("date").size()

print(f"rows: {len(panel)}, days: {T}, names per day: rising {counts[up].mean():.0f}, "
      f"falling {counts[~up].mean():.0f}")
print(f"row average of forward excess   (wrong): {wrong:+.2%} per 20 sessions")
print(f"average of daily book returns   (right): {right:+.2%} per 20 sessions")
assert wrong > 0 > right, "row-weighting should flip the sign of the conclusion"
