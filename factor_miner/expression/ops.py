"""算子库: 全部作用于 (T交易日 x N股票) 的 pd.DataFrame, NaN感知, 两引擎共享.

三类算子(方案§5):
- 逐元素: add sub mul div neg abs slog sign sqrt(带符号) inv
- 时序(沿T轴滚动): ts_mean ts_std ts_sum ts_min ts_max ts_med ts_rank ts_delay
  ts_delta ts_ret ts_corr ts_cov ts_ema ts_skew ts_slope decay_linear
- 截面(沿N轴逐日): cs_rank cs_zscore cs_demean cs_scale cs_indneutral
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

_EPS = 1e-12


def _mp(w: int) -> int:
    """rolling 最小样本数: 半窗, 至少2。"""
    return max(2, w // 2)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan)


# ---------------- 逐元素 ----------------
def op_add(a, b):
    return a + b


def op_sub(a, b):
    return a - b


def op_mul(a, b):
    return a * b


def op_div(a, b):
    return _clean(a / b.where(b.abs() > _EPS))


def op_neg(a):
    return -a


def op_abs(a):
    return a.abs()


def op_slog(a):
    return np.sign(a) * np.log1p(a.abs())


def op_sign(a):
    return np.sign(a)


def op_sqrt(a):
    """带符号平方根, 对负值封闭。"""
    return np.sign(a) * np.sqrt(a.abs())


def op_inv(a):
    return _clean(1.0 / a.where(a.abs() > _EPS))


# ---------------- 时序 ----------------
def op_ts_mean(a, w):
    return a.rolling(w, min_periods=_mp(w)).mean()


def op_ts_std(a, w):
    return a.rolling(w, min_periods=_mp(w)).std()


def op_ts_sum(a, w):
    return a.rolling(w, min_periods=_mp(w)).sum()


def op_ts_min(a, w):
    return a.rolling(w, min_periods=_mp(w)).min()


def op_ts_max(a, w):
    return a.rolling(w, min_periods=_mp(w)).max()


def op_ts_med(a, w):
    return a.rolling(w, min_periods=_mp(w)).median()


def op_ts_rank(a, w):
    return a.rolling(w, min_periods=_mp(w)).rank(pct=True)


def op_ts_skew(a, w):
    return a.rolling(w, min_periods=max(3, w // 2)).skew()


def op_ts_delay(a, w):
    return a.shift(w)


def op_ts_delta(a, w):
    return a - a.shift(w)


def op_ts_ret(a, w):
    base = a.shift(w)
    return _clean(a / base.where(base.abs() > _EPS) - 1.0)


def op_ts_corr(a, b, w):
    return _clean(a.rolling(w, min_periods=_mp(w)).corr(b))


def op_ts_cov(a, b, w):
    return _clean(a.rolling(w, min_periods=_mp(w)).cov(b))


def op_ts_ema(a, w):
    return a.ewm(span=w, min_periods=_mp(w), adjust=False).mean()


def _wshift_sum(a: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    """Σ_k weights[k] * a.shift(k)。窗口内含NaN则结果NaN(严格模式)。"""
    acc: pd.DataFrame | None = None
    for k, c in enumerate(weights):
        term = a.shift(k) * float(c)
        acc = term if acc is None else acc + term
    return acc


def op_decay_linear(a, w):
    wts = np.arange(w, 0, -1, dtype=np.float64)  # 今日权重最大
    return _wshift_sum(a, wts / wts.sum())


def op_ts_slope(a, w):
    """对窗口内序列做时间回归的斜率(单位/天)。"""
    k = np.arange(w, dtype=np.float64)          # k=滞后
    c = ((w - 1) / 2.0 - k) / (w * (w * w - 1) / 12.0)
    return _wshift_sum(a, c)


# ---------------- 截面 ----------------
def op_cs_rank(a):
    return a.rank(axis=1, pct=True)


def op_cs_demean(a):
    return a.sub(a.mean(axis=1), axis=0)


def op_cs_zscore(a):
    sd = a.std(axis=1).replace(0, np.nan)
    return _clean(a.sub(a.mean(axis=1), axis=0).div(sd, axis=0))


def op_cs_scale(a):
    s = a.abs().sum(axis=1).replace(0, np.nan)
    return _clean(a.div(s, axis=0))


def op_cs_indneutral(a, industry: pd.DataFrame):
    """按行业逐日去均值。industry: (T,N) 行业代码(float, NaN=无行业)。"""
    va = a.to_numpy(dtype=np.float64, copy=True)
    vi = industry.reindex_like(a).to_numpy(dtype=np.float64)
    out = np.full_like(va, np.nan)
    for t in range(va.shape[0]):
        x, g = va[t], vi[t]
        m = np.isfinite(x) & np.isfinite(g)
        if m.sum() < 2:
            continue
        codes, inv = np.unique(g[m].astype(np.int64), return_inverse=True)
        sums = np.bincount(inv, weights=x[m])
        cnts = np.bincount(inv)
        out[t, m] = x[m] - (sums / cnts)[inv]
    return pd.DataFrame(out, index=a.index, columns=a.columns, dtype=np.float32)


# ---------------- 注册表 ----------------
@dataclass(frozen=True)
class OpSpec:
    name: str
    fn: Callable
    n_args: int          # 数据参数个数
    window: bool         # 是否带窗口参数(最后一个参数)
    kind: str            # elementwise / ts / cs
    commutative: bool = False
    needs_industry: bool = False


_SPECS = [
    OpSpec("add", op_add, 2, False, "elementwise", commutative=True),
    OpSpec("sub", op_sub, 2, False, "elementwise"),
    OpSpec("mul", op_mul, 2, False, "elementwise", commutative=True),
    OpSpec("div", op_div, 2, False, "elementwise"),
    OpSpec("neg", op_neg, 1, False, "elementwise"),
    OpSpec("abs", op_abs, 1, False, "elementwise"),
    OpSpec("slog", op_slog, 1, False, "elementwise"),
    OpSpec("sign", op_sign, 1, False, "elementwise"),
    OpSpec("sqrt", op_sqrt, 1, False, "elementwise"),
    OpSpec("inv", op_inv, 1, False, "elementwise"),
    OpSpec("ts_mean", op_ts_mean, 1, True, "ts"),
    OpSpec("ts_std", op_ts_std, 1, True, "ts"),
    OpSpec("ts_sum", op_ts_sum, 1, True, "ts"),
    OpSpec("ts_min", op_ts_min, 1, True, "ts"),
    OpSpec("ts_max", op_ts_max, 1, True, "ts"),
    OpSpec("ts_med", op_ts_med, 1, True, "ts"),
    OpSpec("ts_rank", op_ts_rank, 1, True, "ts"),
    OpSpec("ts_skew", op_ts_skew, 1, True, "ts"),
    OpSpec("ts_delay", op_ts_delay, 1, True, "ts"),
    OpSpec("ts_delta", op_ts_delta, 1, True, "ts"),
    OpSpec("ts_ret", op_ts_ret, 1, True, "ts"),
    OpSpec("ts_corr", op_ts_corr, 2, True, "ts", commutative=True),
    OpSpec("ts_cov", op_ts_cov, 2, True, "ts", commutative=True),
    OpSpec("ts_ema", op_ts_ema, 1, True, "ts"),
    OpSpec("ts_slope", op_ts_slope, 1, True, "ts"),
    OpSpec("decay_linear", op_decay_linear, 1, True, "ts"),
    OpSpec("cs_rank", op_cs_rank, 1, False, "cs"),
    OpSpec("cs_demean", op_cs_demean, 1, False, "cs"),
    OpSpec("cs_zscore", op_cs_zscore, 1, False, "cs"),
    OpSpec("cs_scale", op_cs_scale, 1, False, "cs"),
    OpSpec("cs_indneutral", op_cs_indneutral, 1, False, "cs", needs_industry=True),
]

OP_REGISTRY: dict[str, OpSpec] = {s.name: s for s in _SPECS}
