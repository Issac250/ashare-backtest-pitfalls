"""01 收盘价成交不了：买卖价差跳动（bid-ask bounce）制造的假收益。

Every stock has a true value that is a pure random walk, so no strategy can
beat it. Each printed price (open or close) lands on the bid or the ask at
random. A one-day reversal signal then looks spectacular on close-to-close
returns, and earns nothing once orders fill at the next open.
"""
import numpy as np
import pandas as pd
from ashare_bt import BacktestConfig, run_backtest
from common import ZERO_COST, cagr, codes, to_bars, to_signal

rng = np.random.default_rng(1)
N, T = 200, 750
HALF_SPREAD = 0.0025          # a 0.5% bid-ask spread, ordinary for a small cap
dates = pd.bdate_range("2021-01-04", periods=T)
names = codes(N)

# True (mid) value: log random walk, split into an overnight and an intraday leg.
log_open = np.empty((T, N))
log_close = np.empty((T, N))
prev = np.log(rng.uniform(8, 40, N))
for t in range(T):
    log_open[t] = prev + rng.normal(0, 0.006, N)
    log_close[t] = log_open[t] + rng.normal(0, 0.015, N)
    prev = log_close[t]


def printed(log_mid):
    side = rng.choice([-1.0, 1.0], size=log_mid.shape)    # last trade hit the bid or the ask
    return pd.DataFrame(np.round(np.exp(log_mid) * (1 + HALF_SPREAD * side), 2),
                        index=dates, columns=names)


open_, close = printed(log_open), printed(log_close)
bars = to_bars(open_, close)

# Signal at the close of day t: yesterday's biggest losers (one-day reversal).
ret = close / close.shift(1) - 1
score = -ret
K = 20

# WRONG: buy the 20 losers at today's close, sell at tomorrow's close.
picks = score.rank(axis=1, ascending=False) <= K
daily = ret.shift(-1).where(picks).mean(axis=1).dropna()
wrong = cagr(pd.concat([pd.Series([1.0], index=[daily.index[0]]), (1 + daily).cumprod()]))

# RIGHT: same signal, filled at the next open by the engine, zero costs.
cfg = BacktestConfig(capital=10_000_000, n_positions=K, rebalance_days=1, costs=ZERO_COST)
right = run_backtest(to_signal(score), bars, cfg).metrics["cagr"]

# And what an account would actually see once it pays commissions and spread.
real = run_backtest(to_signal(score), bars, BacktestConfig(capital=10_000_000, n_positions=K,
                                                           rebalance_days=1)).metrics["cagr"]

print(f"close-to-close, no costs (wrong) CAGR: {wrong:+.1%}")
print(f"next-open fill, no costs (right) CAGR: {right:+.1%}")
print(f"next-open fill, default costs    CAGR: {real:+.1%}")
assert wrong > right + 0.10, "the close-price book should look far better than the fillable one"
