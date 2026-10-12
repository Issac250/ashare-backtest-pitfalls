"""10 基准是算法产物：「等权全市场」默认是每日零成本再平衡。

The usual "market" benchmark is the average daily return across all stocks,
compounded: an equal-weight book rebalanced back to equal weight EVERY day, at
no cost. When prices mean-revert a little around their fair value (as small
caps do), that daily rebalancing keeps buying the day's losers and selling the
winners, and harvests a return nobody can collect: it needs hundreds of trades
a day for free. What you can actually buy is far lower.
"""
import numpy as np
import pandas as pd
from ashare_bt import BacktestConfig, run_backtest
from common import cagr, codes, to_bars, to_signal

rng = np.random.default_rng(11)
N, T = 300, 1000
dates = pd.bdate_range("2020-01-02", periods=T)
names = codes(N)

mkt = rng.normal(0.0003, 0.012, T)
fair = np.log(rng.uniform(5, 30, N)) + np.cumsum(mkt[:, None] + rng.normal(0, 0.012, (T, N)), axis=0)
noise = np.zeros((T, N))                                   # a transient deviation that decays
for t in range(1, T):
    noise[t] = 0.85 * noise[t - 1] + rng.normal(0, 0.02, N)
log_close = fair + noise
log_open = np.vstack([log_close[:1], log_close[:-1]]) + rng.normal(0, 0.004, (T, N))
close = pd.DataFrame(np.round(np.exp(log_close), 2), index=dates, columns=names)
open_ = pd.DataFrame(np.round(np.exp(log_open), 2), index=dates, columns=names)
bars = to_bars(open_, close)
ret = close / close.shift(1) - 1


def curve(daily):
    return pd.concat([pd.Series([1.0], index=[dates[0]]), (1 + daily).cumprod()])


# WRONG benchmark: mean daily return across all stocks, compounded.
ew_daily = cagr(curve(ret.mean(axis=1).iloc[1:]))
# An investable alternative: buy every stock once, equal weight, and hold.
bh = cagr(pd.concat([pd.Series([1.0], index=[dates[0]]),
                     (close.iloc[1:] / close.iloc[0]).mean(axis=1)]))
# And what an account gets trying to track the benchmark: hold all 300,
# rebalance to equal weight every 20 sessions, pay ordinary costs.
cfg = BacktestConfig(capital=50_000_000, n_positions=N, rebalance_days=20)
engine = run_backtest(to_signal(pd.DataFrame(1.0, index=dates, columns=names)), bars, cfg).metrics["cagr"]

print(f"equal weight, rebalanced daily, free   (wrong) CAGR: {ew_daily:+.1%}")
print(f"equal weight, buy and hold             (right) CAGR: {bh:+.1%}")
print(f"engine: all {N}, rebalance every 20 sessions, with costs CAGR: {engine:+.1%}")
assert ew_daily > bh + 0.03 and ew_daily > engine + 0.03, "the daily-rebalanced benchmark is not buyable"
