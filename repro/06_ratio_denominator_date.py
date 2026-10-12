"""06 比值的分母也有日期：供应商算好的「占市值比例」常常用的是事后的市值。

An event is announced on day t and takes effect on day t+20. Its size in yuan
is fixed at t. A vendor publishes "size / market cap", but the market cap is
the one on the day BEFORE the event takes effect -- information from after t.
A stock that falls between t and the event date gets a bigger ratio, so it is
more likely to pass a "big event" threshold; and the fall is part of the
return we are measuring from t. The event has no effect at all in this panel.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
n = 20_000
cap_t = np.exp(rng.normal(np.log(5e9), 0.8, n))        # market cap on the signal day t
size = cap_t * np.exp(rng.normal(np.log(0.02), 0.6, n))  # event size in yuan, fixed at t
r_pre = rng.normal(0.0, 0.12, n)                         # t -> day before the event
r_post = rng.normal(0.0, 0.10, n)                        # event -> 20 sessions later
fwd = (1 + r_pre) * (1 + r_post) - 1                     # return measured from t
THRESHOLD = 0.03

events = pd.DataFrame({
    "ratio_vendor": size / (cap_t * (1 + r_pre)),         # denominator dated after t
    "ratio_at_t": size / cap_t,                           # denominator known at t
    "fwd": fwd,
})


def big_minus_small(col):
    big = events[col] > THRESHOLD
    return events.loc[big, "fwd"].mean() - events.loc[~big, "fwd"].mean(), big.mean()


wrong, wrong_share = big_minus_small("ratio_vendor")
right, right_share = big_minus_small("ratio_at_t")
print(f"flagged 'big event': vendor ratio {wrong_share:.1%} of events, ratio at t {right_share:.1%}")
print(f"big - small forward return, vendor denominator (wrong): {wrong:+.2%}")
print(f"big - small forward return, denominator at t   (right): {right:+.2%}")
assert wrong < right - 0.02, "a post-t denominator should manufacture a negative 'event effect'"
assert abs(right) < 0.01, "with the denominator dated at t there is no effect to find"
