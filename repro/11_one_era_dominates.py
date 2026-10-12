"""11 全样本数字被一个已经结束的时代撑着。

A factor pays a premium for the first two thirds of the sample and nothing in
the last third (the trade got crowded, the rules changed -- the reason does not
matter). The full-sample Sharpe still looks healthy, because it averages the
good era in. The latest segment is the honest estimate of what running it today
would feel like, and it is the one a full-sample number hides.
"""
import numpy as np
import pandas as pd
from ashare_bt import BacktestConfig, run_backtest
from common import codes, to_bars, to_signal

rng = np.random.default_rng(14)
N, T = 200, 1500
dates = pd.bdate_range("2018-01-02", periods=T)
names = codes(N)

x = np.zeros((T, N))                                         # an observable characteristic
for t in range(1, T):
    x[t] = 0.98 * x[t - 1] + rng.normal(0, 0.2, N)
x /= x.std()
premium = np.where(np.arange(T) < T * 2 // 3, 0.0005, 0.0)  # per unit of x, per day
mkt = rng.normal(0.0001, 0.006, T)
r = mkt[:, None] + premium[:, None] * np.vstack([np.zeros(N), x[:-1]]) + rng.normal(0, 0.02, (T, N))

log_close = np.log(rng.uniform(8, 30, N)) + np.cumsum(np.log1p(r), axis=0)
log_open = np.vstack([log_close[:1], log_close[:-1]])        # open at yesterday's close
close = pd.DataFrame(np.round(np.exp(log_close), 2), index=dates, columns=names)
open_ = pd.DataFrame(np.round(np.exp(log_open), 2), index=dates, columns=names)
bars = to_bars(open_, close)
score = pd.DataFrame(x, index=dates, columns=names)

res = run_backtest(to_signal(score), bars, BacktestConfig(capital=10_000_000, n_positions=20,
                                                          rebalance_days=5))
daily = res.equity.pct_change().dropna()


def sharpe(d):
    return d.mean() / d.std() * np.sqrt(252)


cut = np.linspace(0, len(daily), 4).astype(int)
thirds = [daily.iloc[a:b] for a, b in zip(cut[:-1], cut[1:])]
full = sharpe(daily)
print(f"full sample  {daily.index[0].date()} -> {daily.index[-1].date()}  Sharpe {full:.2f}  (wrong to quote alone)")
for k, seg in enumerate(thirds, 1):
    print(f"segment {k}    {seg.index[0].date()} -> {seg.index[-1].date()}  Sharpe {sharpe(seg):.2f}")
latest = sharpe(thirds[-1])
print(f"latest segment Sharpe (right to report next to it): {latest:.2f}")
assert full > latest + 0.5, "the full-sample Sharpe should be carried by the earlier era"
