"""02 检查涨跌停判断是否用了正确时点。

订单按开盘价执行，只替换开盘和收盘两种涨跌停标记。
使用随机评分和零费用，隔离标记口径的影响。
"""
from ashare_bt import BacktestConfig, run_backtest
from ashare_bt.sizing import limit_prices
from ashare_bt.synthetic import make_market, make_signal
from common import ZERO_COST

bars = make_market(n_stocks=200, n_days=750, seed=3, daily_vol=0.04, limit_open_prob=0.01)
signal = make_signal(bars, kind="random", seed=3)    # no alpha: only the gate differs
cfg = BacktestConfig(capital=10_000_000, n_positions=50, rebalance_days=5, costs=ZERO_COST)

up, dn = limit_prices(bars["preclose"], 0.10)
close_up = bars["close"] >= up
close_dn = bars["close"] <= dn
share_up = (~bars["open_limit_up"][close_up]).mean()
share_dn = (~bars["open_limit_down"][close_dn]).mean()
print(f"closed limit-up but opened below the limit:   {share_up:.1%} of {close_up.sum()} rows")
print(f"closed limit-down but opened above the limit: {share_dn:.1%} of {close_dn.sum()} rows")

# WRONG: the gates are computed from the close, then used for an open fill.
wrong_bars = bars.assign(open_limit_up=close_up, open_limit_down=close_dn)
wrong = run_backtest(signal, wrong_bars, cfg).metrics["cagr"]
# RIGHT: the gates describe the open, the moment the order fills.
right = run_backtest(signal, bars, cfg).metrics["cagr"]

print(f"close-based gate (wrong) CAGR: {wrong:+.1%}")
print(f"open-based gate  (right) CAGR: {right:+.1%}")
assert wrong < right, "a close-based gate rejects the day's winners, so it should understate"
