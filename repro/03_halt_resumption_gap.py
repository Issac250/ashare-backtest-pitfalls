"""03 比较公告日起的价格变化与复牌后入场的收益。

合成股票公告后停牌 20 个交易日，复牌跳空均值设为 +20%。
模型设定复牌开盘可以交易，检查停牌前的价格是否被误作新买单成本。
"""
import numpy as np
import pandas as pd
from ashare_bt import BacktestConfig, run_backtest
from common import ZERO_COST, codes, to_bars, to_signal

rng = np.random.default_rng(4)
N, T, HALT, HOLD = 60, 500, 20, 10
dates = pd.bdate_range("2021-01-04", periods=T)
names = codes(N)

close = np.empty((T, N))
open_ = np.empty((T, N))
tradable = np.ones((T, N), dtype=bool)
announce = rng.integers(30, T - HALT - HOLD - 10, size=N)   # one event per stock
price = rng.uniform(8, 40, N)
for t in range(T):
    for i in range(N):
        a = announce[i]
        if a < t <= a + HALT:                       # halted: frozen row
            open_[t, i] = close[t, i] = price[i]
            tradable[t, i] = False
            continue
        if t == a + HALT + 1:                       # resumption: the gap is in the open
            o = price[i] * (1 + rng.normal(0.20, 0.15))
        else:
            o = price[i] * (1 + rng.normal(0, 0.006))
        c = o * (1 + rng.normal(0, 0.02))
        open_[t, i], close[t, i] = round(o, 2), round(c, 2)
        price[i] = close[t, i]

open_ = pd.DataFrame(open_, index=dates, columns=names)
close = pd.DataFrame(close, index=dates, columns=names)
bars = to_bars(open_, close, tradable=pd.DataFrame(tradable, index=dates, columns=names))
# A resumption after a long halt is not bound by yesterday's band in this panel.
bars["open_limit_up"] = False
bars["open_limit_down"] = False

# WRONG: event study on the data rows -- enter at the announcement-day close,
# exit HOLD sessions after resumption.
exit_day = announce + HALT + 1 + HOLD
wrong = np.mean([close.iat[exit_day[i], i] / close.iat[announce[i], i] - 1 for i in range(N)])

# RIGHT: the engine is told "hold this name" from the announcement until the
# exit day. It can only buy once the stock trades again, at the resumption open.
flag = pd.DataFrame(np.nan, index=dates, columns=names)
for i in range(N):
    flag.iloc[announce[i]:exit_day[i], i] = 1.0
cfg = BacktestConfig(capital=10_000_000, n_positions=N, rebalance_days=1, costs=ZERO_COST,
                     rebalance_weights=False)   # one buy and one sell per event
trades = run_backtest(to_signal(flag, keep_empty_days=True), bars, cfg).trades
legs = trades.groupby(["code", "side"])["notional"].sum().unstack()
assert legs.dropna().shape[0] == N, "every event should be bought once and sold once"
right = (legs["sell"] / legs["buy"] - 1).mean()

print(f"events: {N}, halt {HALT} sessions, hold {HOLD} sessions after resumption")
print(f"enter at announcement close (wrong) mean event return: {wrong:+.1%}")
print(f"enter at first tradable open (right) mean event return: {right:+.1%}")
assert wrong > right + 0.10, "the frozen rows should hand the event study a gap it could not buy"
