"""因子预处理: MAD去极值 -> 可选行业/市值中性化 -> 截面zscore. 全部逐日截面操作."""
from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize_mad(f: pd.DataFrame, k: float = 5.0) -> pd.DataFrame:
    med = f.median(axis=1)
    mad = (f.sub(med, axis=0)).abs().median(axis=1)
    lo = med - k * 1.4826 * mad
    hi = med + k * 1.4826 * mad
    return f.clip(lower=lo, upper=hi, axis=0)


def zscore(f: pd.DataFrame) -> pd.DataFrame:
    sd = f.std(axis=1).replace(0, np.nan)
    return f.sub(f.mean(axis=1), axis=0).div(sd, axis=0)


def _industry_demean_np(x: np.ndarray, g: np.ndarray) -> np.ndarray:
    out = np.full_like(x, np.nan)
    for t in range(x.shape[0]):
        xt, gt = x[t], g[t]
        m = np.isfinite(xt) & np.isfinite(gt)
        if m.sum() < 2:
            out[t, m] = xt[m]
            continue
        codes, inv = np.unique(gt[m].astype(np.int64), return_inverse=True)
        mean = np.bincount(inv, weights=xt[m]) / np.bincount(inv)
        out[t, m] = xt[m] - mean[inv]
    return out


def neutralize(f: pd.DataFrame, industry: pd.DataFrame | None,
               log_mv: pd.DataFrame | None) -> pd.DataFrame:
    """行业去均值 + 对数市值单变量回归取残差(逐日)。任一为None则跳过该步。"""
    x = f.to_numpy(dtype=np.float64, copy=True)
    if industry is not None:
        x = _industry_demean_np(x, industry.reindex_like(f).to_numpy(np.float64))
    if log_mv is not None:
        v = log_mv.reindex_like(f).to_numpy(np.float64)
        vc = v - np.nanmean(v, axis=1, keepdims=True)
        xc = x - np.nanmean(x, axis=1, keepdims=True)
        both = np.isfinite(xc) & np.isfinite(vc)
        xm = np.where(both, xc, 0.0)
        vm = np.where(both, vc, 0.0)
        var = (vm * vm).sum(axis=1)
        beta = np.divide((xm * vm).sum(axis=1), var,
                         out=np.zeros_like(var), where=var > 1e-12)
        x = np.where(both, xc - beta[:, None] * vc, np.where(np.isfinite(xc), xc, np.nan))
    return pd.DataFrame(x, index=f.index, columns=f.columns)


def preprocess(f: pd.DataFrame, universe: pd.DataFrame,
               industry: pd.DataFrame | None = None,
               log_mv: pd.DataFrame | None = None,
               winsor_k: float = 5.0) -> pd.DataFrame:
    """标准管道: 限定股票池 -> 去极值 -> 中性化 -> zscore。"""
    f = f.where(universe.astype(bool))
    f = winsorize_mad(f, winsor_k)
    f = neutralize(f, industry, log_mv)
    return zscore(f)
