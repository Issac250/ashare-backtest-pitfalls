"""12 一个「筛选」的对照不是 0，是同样大小的随机子集。

Adding a screen to a universe does two things at once: it removes particular
stocks, and it makes the pool smaller. A smaller pool costs money by itself:
on every rebalance the ranking has fewer good names to choose from, so the
book holds names that rank lower. Comparing "with screen" against "without"
charges that shrinkage to the screen.

Synthetic setup: a characteristic x earns a premium, and the book buys the top
20 by x. The screen keeps 40% of the stocks, and the stocks it removes really
are slightly worse (they drift down a little). So the screen is genuinely
useful -- but not useful enough to pay for a pool 60% smaller.
"""
import numpy as np
import pandas as pd
from ashare_bt import BacktestConfig, run_backtest
from common import codes, to_bars, to_signal

rng = np.random.default_rng(15)
N, T, KEEP = 400, 1000, 160
dates = pd.bdate_range("2020-01-02", periods=T)
names = codes(N)

x = np.zeros((T, N))
for t in range(1, T):
    x[t] = 0.97 * x[t - 1] + rng.normal(0, 0.25, N)
x /= x.std()
passes = np.zeros(N, dtype=bool)
passes[rng.choice(N, KEEP, replace=False)] = True           # the screen's verdict, fixed per stock
drift = np.where(passes, 0.0, -0.0003)                       # removed stocks really are a bit worse
r = (rng.normal(0.0002, 0.008, T)[:, None] + 0.0006 * np.vstack([np.zeros(N), x[:-1]])
     + drift + rng.normal(0, 0.02, (T, N)))

log_close = np.log(rng.uniform(8, 30, N)) + np.cumsum(np.log1p(r), axis=0)
log_open = np.vstack([log_close[:1], log_close[:-1]])
close = pd.DataFrame(np.round(np.exp(log_close), 2), index=dates, columns=names)
open_ = pd.DataFrame(np.round(np.exp(log_open), 2), index=dates, columns=names)
bars = to_bars(open_, close)
score = pd.DataFrame(x, index=dates, columns=names)
cfg = BacktestConfig(capital=10_000_000, n_positions=20, rebalance_days=10)


def book(pool_mask):
    return run_backtest(to_signal(score.loc[:, pool_mask]), bars, cfg).metrics["cagr"]


full = book(np.ones(N, dtype=bool))
screened = book(passes)
random_same_size = []
for k in range(5):
    m = np.zeros(N, dtype=bool)
    m[np.random.default_rng(100 + k).choice(N, KEEP, replace=False)] = True
    random_same_size.append(book(m))
rand = float(np.mean(random_same_size))

print(f"full pool ({N})              CAGR: {full:+.1%}")
print(f"screened pool ({KEEP})         CAGR: {screened:+.1%}")
print(f"random pools ({KEEP}) x5       CAGR: {rand:+.1%}  (range {min(random_same_size):+.1%} .. "
      f"{max(random_same_size):+.1%})")
print(f"screen - full pool        (wrong control): {screened - full:+.1%} a year")
print(f"screen - same-size random (right control): {screened - rand:+.1%} a year")
assert screened - full < 0 < screened - rand, "the two controls should disagree on the sign"
