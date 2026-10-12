"""复现脚本共用的合成数据转换和指标函数。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from ashare_bt import CostModel
from ashare_bt.sizing import limit_prices

# A cost model that charges nothing, so a repro can show that a return never
# existed, rather than that costs ate it.
ZERO_COST = CostModel(commission_rate=0.0, commission_min=0.0, stamp_duty=0.0,
                      transfer_fee=0.0, slippage_bps=0.0, impact_coef=0.0)


def to_bars(open_: pd.DataFrame, close: pd.DataFrame, preclose: pd.DataFrame | None = None,
            tradable: pd.DataFrame | None = None, amount: float = 1e9,
            limit_width: float = 0.10) -> pd.DataFrame:
    """Wide [date x code] price frames -> the long bars frame ashare-bt reads.

    `preclose` defaults to the previous close (no corporate actions). The limit
    flags describe the OPEN, which is what the engine needs.
    """
    if preclose is None:
        preclose = close.shift(1).fillna(open_)
    if tradable is None:
        tradable = pd.DataFrame(True, index=close.index, columns=close.columns)
    long = pd.DataFrame({
        "open": open_.stack(), "close": close.stack(), "preclose": preclose.stack(),
        "tradable": tradable.stack().astype(bool),
    })
    long.index.names = ["date", "code"]
    long = long.reset_index()
    long["ret"] = long["close"] / long["preclose"] - 1.0
    long["amount"] = np.where(long["tradable"], amount, 0.0)
    up, dn = limit_prices(long["preclose"], limit_width)
    long["open_limit_up"] = long["tradable"] & (long["open"] >= up)
    long["open_limit_down"] = long["tradable"] & (long["open"] <= dn)
    return long.sort_values(["date", "code"]).reset_index(drop=True)


def to_signal(score: pd.DataFrame, keep_empty_days: bool = False) -> pd.DataFrame:
    """Wide [date x code] score frame -> long [date, code, score].

    NaN scores are dropped. With `keep_empty_days`, NaN rows are kept instead,
    so a date on which nothing qualifies is still a rebalance day (the engine
    then sells everything), not a day the engine never looks at.
    """
    s = score.stack(future_stack=True).rename("score").reset_index()
    s.columns = ["date", "code", "score"]
    if not keep_empty_days:
        s = s.dropna(subset=["score"])
    return s.reset_index(drop=True)


def cagr(equity: pd.Series) -> float:
    """Compound annual growth on calendar time, the same definition ashare-bt uses."""
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)


def codes(n: int) -> list[str]:
    return [f"S{i:04d}" for i in range(1, n + 1)]
