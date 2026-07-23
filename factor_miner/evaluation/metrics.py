"""指标计算: RankIC序列/ICIR(核心指标)/分层回测/换手/相关性. 输入均为对齐宽表."""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_XS = 30  # 单日有效截面最小样本数


def daily_rank_ic(f: pd.DataFrame, y: pd.DataFrame) -> pd.Series:
    """逐日 Spearman RankIC (对秩做Pearson)。无效日为NaN。"""
    valid = f.notna() & y.notna()
    fr = f.where(valid).rank(axis=1)
    yr = y.where(valid).rank(axis=1)
    fc = fr.sub(fr.mean(axis=1), axis=0)
    yc = yr.sub(yr.mean(axis=1), axis=0)
    num = (fc * yc).sum(axis=1)
    den = np.sqrt((fc * fc).sum(axis=1) * (yc * yc).sum(axis=1))
    ic = num / den.replace(0, np.nan)
    ic[valid.sum(axis=1) < MIN_XS] = np.nan
    return ic


def ic_stats(ic: pd.Series, horizon: int) -> dict:
    """⭐ICIR = IC均值/IC标准差; 年化 = ICIR*sqrt(252/H)(按不重叠周期近似)。"""
    s = ic.dropna()
    if len(s) < 20:
        return {"n_days": int(len(s)), "ic_mean": np.nan, "ic_std": np.nan,
                "icir": np.nan, "icir_ann": np.nan,
                "rank_ic": np.nan, "ic_skew": np.nan, "win_rate": np.nan}
    mean, std = float(s.mean()), float(s.std())
    icir = mean / std if std > 1e-12 else np.nan
    return {
        "n_days": int(len(s)),
        "ic_mean": round(mean, 5),
        "ic_std": round(std, 5),
        "icir": round(icir, 4),
        "icir_ann": round(icir * np.sqrt(252.0 / horizon), 4),
        "rank_ic": round(float(s.mean()), 5),            # = IC 均值(冗余, 方便规则里显式选 RankIC)
        "ic_skew": round(float(s.skew()), 4),            # IC 偏度: >0 正偏(极端正IC概率略高)
        "win_rate": round(float((np.sign(s) == np.sign(mean)).mean()), 4),
    }


def layered_backtest(f: pd.DataFrame, daily_ret: pd.DataFrame,
                     tradable: pd.DataFrame, n_q: int, horizon: int) -> dict:
    """等权分层, 每 H 日调仓, 持有期内逐日收益。返回各组年化/多空绩效/单调性/换手。"""
    fv = f.where(tradable.astype(bool))
    T = len(f.index)
    rebal = list(range(0, T - horizon - 1, horizon))
    group_daily: list[np.ndarray] = []       # 每持有日各组均值收益 (Q,)
    prev_top: set | None = None
    top_turn: list[float] = []
    R = daily_ret.to_numpy(np.float64)
    ranks = fv.rank(axis=1, pct=True).to_numpy(np.float64)
    for i in rebal:
        r = ranks[i]
        m = np.isfinite(r)
        if m.sum() < MIN_XS * 2:
            continue
        q = np.clip((r[m] * n_q).astype(int), 0, n_q - 1)
        idx = np.where(m)[0]
        members = [idx[q == k] for k in range(n_q)]
        top = set(idx[q == n_q - 1].tolist())
        if prev_top:
            top_turn.append(1.0 - len(top & prev_top) / max(len(top), 1))
        prev_top = top
        for d in range(i + 2, min(i + 2 + horizon, T)):  # t+1收盘建仓 -> t+2起产生收益
            row = R[d]
            group_daily.append(np.array([
                np.nanmean(row[mem]) if len(mem) else np.nan for mem in members
            ]))
    if not group_daily:
        return {"ls_ann_ret": np.nan, "ls_sharpe": np.nan, "ls_mdd": np.nan,
                "monotonicity": np.nan, "top_turnover": np.nan, "group_ann": []}
    G = np.vstack(group_daily)                            # (D, Q)
    ann = np.nanmean(G, axis=0) * 252
    ls = G[:, -1] - G[:, 0]
    ls = ls[np.isfinite(ls)]
    ls_ann = float(ls.mean() * 252)
    ls_sharpe = float(ls.mean() / ls.std() * np.sqrt(252)) if ls.std() > 1e-12 else np.nan
    nav = np.cumprod(1 + ls)
    mdd = float((1 - nav / np.maximum.accumulate(nav)).max()) if len(nav) else np.nan
    order = pd.Series(ann).rank().corr(pd.Series(range(n_q)).rank())
    return {
        "ls_ann_ret": round(ls_ann, 4),
        "ls_sharpe": round(ls_sharpe, 3),
        "ls_mdd": round(mdd, 4),
        "monotonicity": round(float(order), 3),
        "top_turnover": round(float(np.mean(top_turn)), 4) if top_turn else np.nan,
        "group_ann": [round(float(x), 4) for x in ann],
    }


def rank_autocorr(f: pd.DataFrame, horizon: int, step: int = 5) -> float:
    """因子秩自相关(lag=H): 换手率代理, 越高换手越低。"""
    r = f.rank(axis=1)
    vals = []
    for i in range(horizon, len(f.index), step):
        a, b = r.iloc[i], r.iloc[i - horizon]
        m = a.notna() & b.notna()
        if m.sum() >= MIN_XS:
            vals.append(a[m].corr(b[m]))
    return round(float(np.nanmean(vals)), 4) if vals else np.nan


def coverage(f: pd.DataFrame, universe: pd.DataFrame) -> float:
    u = universe.astype(bool)
    denom = u.sum(axis=1).replace(0, np.nan)
    return round(float((f.notna() & u).sum(axis=1).div(denom).mean()), 4)


def value_corr(f: pd.DataFrame, other: pd.DataFrame, step: int = 5) -> float:
    """两因子值矩阵的截面相关均值(隔step日采样)。向量化实现, 零方差截面记NaN。"""
    A = f.to_numpy(np.float64)[::step]
    B = other.to_numpy(np.float64)[::step]
    m = np.isfinite(A) & np.isfinite(B)
    n = m.sum(axis=1)
    A = np.where(m, A, 0.0)
    B = np.where(m, B, 0.0)
    nf = np.where(n > 0, n, np.nan)
    a_mean = A.sum(axis=1) / nf
    b_mean = B.sum(axis=1) / nf
    Ac = np.where(m, A - np.nan_to_num(a_mean)[:, None], 0.0)
    Bc = np.where(m, B - np.nan_to_num(b_mean)[:, None], 0.0)
    cov = (Ac * Bc).sum(axis=1)
    va = (Ac * Ac).sum(axis=1)
    vb = (Bc * Bc).sum(axis=1)
    # 常数截面判定用相对阈值: 浮点求和误差会让"零方差"残留 ~1e-29, 绝对判 0 不可靠
    scale_a = (A * A).sum(axis=1)
    scale_b = (B * B).sum(axis=1)
    const_a = va <= 1e-12 * scale_a
    const_b = vb <= 1e-12 * scale_b
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / np.sqrt(va * vb)
    corr[(n < MIN_XS) | const_a | const_b] = np.nan
    return round(float(np.nanmean(corr)), 4) if np.isfinite(corr).any() else np.nan
