"""05 数据覆盖率是未来函数：只有活下来的公司才有数据。

A new dataset (say, a per-company attribute table) is rebuilt by the vendor
from TODAY's listing, so companies that were delisted inside the sample have
no rows at all. "Keep the rows where the field is not missing" then means
"keep the companies that will survive" -- a filter that reads the future.

Synthetic assumption: companies that end up delisted drift down over their
final 120 sessions, as most A-share delistings do. The book holds every
eligible stock equally; nothing about the strategy changes between the runs.
"""
import numpy as np
import pandas as pd
from ashare_bt import BacktestConfig, run_backtest
from common import ZERO_COST, codes, to_bars, to_signal

rng = np.random.default_rng(6)
N, T = 300, 750
dates = pd.bdate_range("2021-01-04", periods=T)
names = codes(N)
n_dead = 45
life = np.full(N, T)
life[:n_dead] = rng.integers(250, T - 20, n_dead)    # last session for the delisted ones

open_ = np.full((T, N), np.nan)
close = np.full((T, N), np.nan)
price = rng.uniform(5, 30, N)
for i in range(N):
    for t in range(life[i]):
        drift = -0.004 if (i < n_dead and t >= life[i] - 120) else 0.0003
        o = price[i] * (1 + rng.normal(0, 0.006))
        c = o * (1 + rng.normal(drift, 0.02))
        open_[t, i], close[t, i] = round(o, 2), round(c, 2)
        price[i] = close[t, i]

open_ = pd.DataFrame(open_, index=dates, columns=names)
close = pd.DataFrame(close, index=dates, columns=names)
bars = to_bars(open_, close).dropna(subset=["open"])

# The vendor table: values exist for survivors only.
feature = pd.DataFrame(rng.normal(size=(T, N)), index=dates, columns=names)
feature.iloc[:, :n_dead] = np.nan
alive = close.notna()

dead = np.arange(N) < n_dead
cov = feature.notna().where(alive)
print(f"coverage, survivors:          {cov.loc[:, ~dead].stack().mean():.1%}")
print(f"coverage, delisted in sample: {cov.loc[:, dead].stack().mean():.1%}")


def hold_all(eligible):
    score = pd.DataFrame(1.0, index=dates, columns=names).where(eligible & alive)
    n = int(eligible.any().sum())
    cfg = BacktestConfig(capital=50_000_000, n_positions=n, rebalance_days=20, costs=ZERO_COST)
    return run_backtest(to_signal(score), bars, cfg).metrics["cagr"]


# WRONG: "use the field where we have it" -- rows without it are dropped.
wrong = hold_all(feature.notna())
# RIGHT: missing maps to a neutral value; every listed stock stays eligible.
right = hold_all(alive)

print(f"universe = rows with the field (wrong) CAGR: {wrong:+.1%}")
print(f"universe = every listed stock  (right) CAGR: {right:+.1%}")
assert wrong > right + 0.01, "dropping uncovered rows should drop exactly the companies that die"
