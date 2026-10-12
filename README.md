# A 股回测避坑手册 · ashare-backtest-pitfalls

[![ci](https://github.com/Issac250/ashare-backtest-pitfalls/actions/workflows/ci.yml/badge.svg)](https://github.com/Issac250/ashare-backtest-pitfalls/actions/workflows/ci.yml)

**What.** A Chinese-language handbook of twelve ways an A-share backtest lies, each with a runnable minimal reproduction on synthetic data that prints the wrong number, the right number, and asserts the gap goes the way the chapter says.

**Why.** Every one of these mistakes produces a number that looks reasonable: close-price fills that harvest bid-ask bounce, limit-up gates computed at the wrong time of day, halted rows that hand an event study a gap nobody could buy, unadjusted prices that book splits as losses, data coverage that quietly knows who will be delisted, benchmarks no account can hold. Reading about them is not enough; you have to see the two numbers side by side.

**What's inside.** Twelve chapters (`docs/`), twelve scripts (`repro/`), and a test that runs all of them in CI. Fills, limit gates, T+1, lot sizes and total-return accounting come from [ashare-bt](https://github.com/Issac250/ashare-bt), an open-source A-share backtester. Site: <https://issac250.github.io/ashare-backtest-pitfalls/>

---

## 中文说明

一句话：**回测是怎么骗人的，以及怎么抓住它。** 每一条都附一段能直接运行的最小复现。

在线阅读：<https://issac250.github.io/ashare-backtest-pitfalls/>

| 编号 | 坑 | 一句话后果 |
|---|---|---|
| [01](docs/01-close-price-fills.md) | 收盘价成交不了 | 买卖价差跳动让一个没有信息的反转信号年化 +19.9%，按次日开盘成交只剩 +0.5% |
| [02](docs/02-limit-gate-timing.md) | 用收盘涨跌停判断开盘能否成交 | 拒绝了开盘能买、收盘才涨停的票，随机组合从 +1.9% 变成 −2.7% |
| [03](docs/03-halt-resumption-gap.md) | 停牌：复牌跳空你买不进去 | 冻结行把复牌跳空送给了事件研究：+16.6%，能成交时只有 −0.5% |
| [04](docs/04-unadjusted-prices.md) | 未复权价把送转记成亏损 | 和基准持有同样的股票，却"每年跑输 7.0%" |
| [05](docs/05-coverage-lookahead.md) | 数据覆盖率是未来函数 | `dropna()` 悄悄删掉了将要退市的公司，CAGR +5.3% 变 +8.4% |
| [06](docs/06-ratio-denominator-date.md) | 比值的分母也有日期 | 事后的分母凭空造出 −4.01% 的"事件效应"，真实效应 −0.06% |
| [07](docs/07-halt-relative-return.md) | 停牌造出来的相对收益 | 均值看不出来，但最大赢家里停牌股是基础比例的 3.6 倍 |
| [08](docs/08-arithmetic-annualization.md) | 算术年化与 252 天 | 算术年化 +26.3%，实际复利 +18.3% |
| [09](docs/09-row-average.md) | 逐行平均不是组合收益 | 逐行 +0.29%，按天 −0.50%，结论方向相反 |
| [10](docs/10-benchmark-rebalance.md) | 基准本身是算法造出来的 | 每日再平衡的"大盘" +21.6%，真能买到的 +14.9% |
| [11](docs/11-one-era-dominates.md) | 全样本被一个时代撑着 | 全样本 Sharpe 1.05，最近一段 −0.60 |
| [12](docs/12-screen-control.md) | 筛选的对照是同样大小的随机子集 | 对着全池看"有害" −1.3%，对着随机池看其实 +6.6% |

每章结构：现象 → 为什么会这样 → 最小复现 → 怎么防 → 一行自查。

### 运行复现

需要 Python ≥ 3.11。

```bash
git clone https://github.com/Issac250/ashare-backtest-pitfalls
cd ashare-backtest-pitfalls
pip install -e ".[dev]"            # 会从 GitHub 安装固定版本的 ashare-bt
python repro/01_close_price_fills.py
pytest                             # 逐个运行全部 12 个复现
```

每个脚本都只用合成数据，先打印错的数和对的数，最后用 `assert` 检查差别的方向。脚本是逐个、在独立进程里运行的。

本地预览网站：

```bash
pip install -e ".[docs]"
mkdocs serve
```

### 读之前要知道的

合成数据只用来展示机制。每个复现都刻意把机制放大，好让方向看得清楚，数字的大小不代表真实市场里的量级。

### License

MIT
