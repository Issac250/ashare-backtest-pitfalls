"""07 检查停牌记录是否影响相对收益的极端排名。

停牌时记录价格冻结，复牌时补到模型价格。
比较均值，以及停牌记录在前 10% 和前 1% 样本中的占比。
"""
import numpy as np

rng = np.random.default_rng(8)
N, T, H = 1000, 600, 120
mkt = rng.normal(0.0004, 0.010, T)
mkt[250:290] = rng.normal(-0.012, 0.025, 40)               # a two-month crash
beta = rng.uniform(0.7, 1.3, N)
shadow = np.cumsum(beta * mkt[:, None] + rng.normal(0, 0.02, (T, N)), axis=0)  # log "fair" price

halted = np.zeros((T, N), dtype=bool)
for i in rng.choice(N, 120, replace=False):                # 12% of stocks have one long halt
    s = int(rng.integers(150, 400))
    halted[s:s + int(rng.integers(30, 80)), i] = True

# Observed log price: frozen while halted, catches up to the shadow on resumption.
logp = shadow.copy()
for t in range(1, T):
    logp[t] = np.where(halted[t], logp[t - 1], shadow[t])

snap = np.arange(100, T - H, 20)                            # month-end snapshots
fwd = logp[snap + H] - logp[snap]                           # log return over the next 120 sessions
rel = fwd - fwd.mean(axis=1, keepdims=True)                 # relative to the market that month
halt_days = np.array([halted[s + 1:s + H + 1].sum(axis=0) for s in snap])
flag = (halt_days > 20).ravel()
rel = rel.ravel()

base = flag.mean()
top1 = rel >= np.quantile(rel, 0.99)
top10 = rel >= np.quantile(rel, 0.90)
print(f"snapshots with >20 halted sessions ahead: {base:.1%} of {flag.size}")
print(f"mean test (wrong check): relative return, halted {rel[flag].mean():+.2%} "
      f"vs others {rel[~flag].mean():+.2%} -- looks harmless")
print(f"share of the top 10% winners that were halted: {flag[top10].mean():.1%} ({flag[top10].mean() / base:.1f}x)")
print(f"tail test (right check): share of the top 1% winners that were halted: "
      f"{flag[top1].mean():.1%} ({flag[top1].mean() / base:.1f}x the base rate)")
assert flag[top1].mean() > 2 * base, "frozen stocks should be over-represented in the top tail"
assert abs(rel[flag].mean() - rel[~flag].mean()) < 0.02, "...while their mean looks ordinary"
