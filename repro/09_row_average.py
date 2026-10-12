"""09 比较逐行平均与按信号日等权平均。

合成数据让不同市场状态下的入选数量和未来窗口收益不同。
两种聚合使用相同记录；按日期聚合的结果仍是前瞻窗口统计。
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
