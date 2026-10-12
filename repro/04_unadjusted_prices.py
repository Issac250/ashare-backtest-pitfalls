"""04 未复权价：送转和分红被记成亏损，而基准没有这个问题。

Twenty stocks, bought once and held. A few of them do a bonus issue (10送10:
the raw price halves overnight, the holder's wealth does not move) and all of
them pay a yearly cash dividend. Valuing the book as shares x RAW close books
every bonus issue as a -50% day and every dividend as a loss. The benchmark,
built from the exchange's daily percentage change (a total return), carries
none of that, so the strategy "loses" to a benchmark that holds the same stocks.
"""
import numpy as np
import pandas as pd
from ashare_bt import BacktestConfig, run_backtest
from common import ZERO_COST, cagr, codes, to_bars, to_signal

rng = np.random.default_rng(5)
N, T = 20, 750
dates = pd.bdate_range("2021-01-04", periods=T)
names = codes(N)
SPLIT = {2: 200, 7: 380, 11: 520, 16: 610}          # stock index -> day of a 10送10
DIV_DAYS = (120, 370, 620)                          # one cash dividend a year
DIV_YIELD = 0.03

open_ = np.empty((T, N))
close = np.empty((T, N))
preclose = np.empty((T, N))
price = rng.uniform(10, 40, N)
for t in range(T):
    for i in range(N):
        pc = price[i]
        if SPLIT.get(i) == t:
            pc = pc / 2                              # ex-date: the reference price halves
        if t in DIV_DAYS:
            pc = pc - round(price[i] * DIV_YIELD, 2)  # ex-date: the dividend leaves the price
        pc = round(pc, 2)
        o = round(pc * (1 + rng.normal(0.0002, 0.006)), 2)
        c = round(o * (1 + rng.normal(0.0002, 0.018)), 2)
        preclose[t, i], open_[t, i], close[t, i] = pc, o, c
        price[i] = c

frame = lambda a: pd.DataFrame(a, index=dates, columns=names)
open_, close, preclose = frame(open_), frame(close), frame(preclose)
bars = to_bars(open_, close, preclose=preclose)

# Buy everything at the first open, equal weight, never rebalance.
start = dates[1]
ret = close / preclose - 1                           # what the exchange reports: a total return

# WRONG: shares fixed at the purchase, valued at the raw close.
shares = 1.0 / open_.loc[start]
raw_value = (close.loc[start:] * shares).sum(axis=1) / N
wrong = cagr(pd.concat([pd.Series([1.0], index=[dates[0]]), raw_value]))

# BENCHMARK: the same 20 stocks, buy-and-hold on the total return.
growth = (1 + ret.loc[start:]).cumprod()
growth.loc[start] = close.loc[start] / open_.loc[start]
bench_value = growth.mean(axis=1)
bench = cagr(pd.concat([pd.Series([1.0], index=[dates[0]]), bench_value]))

# RIGHT: the engine carries each position in yuan and advances it by
# open/preclose and close/open, so an ex-date is not a loss.
flag = frame(np.ones((T, N)))
cfg = BacktestConfig(capital=10_000_000, n_positions=N, rebalance_days=10**6, costs=ZERO_COST)
right = run_backtest(to_signal(flag), bars, cfg).metrics["cagr"]

print(f"raw shares x raw close  (wrong) CAGR: {wrong:+.1%}")
print(f"engine, total return    (right) CAGR: {right:+.1%}")
print(f"benchmark on pctChg-style returns CAGR: {bench:+.1%}")
print(f"wrong vs benchmark: {wrong - bench:+.1%} a year  |  right vs benchmark: {right - bench:+.1%} a year")
assert wrong < right - 0.05, "raw-price valuation should book the bonus issues and dividends as losses"
assert abs(right - bench) < 0.01, "the engine and the benchmark hold the same stocks the same way"
