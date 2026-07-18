"""M3 评估层单元测试: IC/ICIR/分层/预处理 对照 scipy/朴素实现."""
import numpy as np
import pandas as pd
from scipy import stats

from factor_miner.evaluation import metrics as M
from factor_miner.evaluation.preprocess import neutralize, winsorize_mad, zscore

T, N = 120, 60
IDX = pd.date_range("2023-01-02", periods=T, freq="B")
COLS = [str(i) for i in range(N)]


def _df(seed):
    return pd.DataFrame(np.random.default_rng(seed).normal(size=(T, N)),
                        index=IDX, columns=COLS)


def test_daily_rank_ic_matches_scipy():
    f, y = _df(1), _df(2)
    ic = M.daily_rank_ic(f, y)
    for t in (5, 50, 100):
        ref = stats.spearmanr(f.iloc[t], y.iloc[t]).statistic
        assert np.isclose(ic.iloc[t], ref, atol=1e-10)


def test_daily_rank_ic_nan_and_min_samples():
    f, y = _df(3), _df(4)
    f.iloc[10, :40] = np.nan          # 有效样本20 < MIN_XS=30
    ic = M.daily_rank_ic(f, y)
    assert np.isnan(ic.iloc[10])


def test_ic_stats_known_values():
    ic = pd.Series([0.05, 0.03, 0.04, 0.06, 0.02] * 20)
    st = M.ic_stats(ic, horizon=10)
    assert np.isclose(st["ic_mean"], 0.04, atol=1e-9)
    assert np.isclose(st["icir"], 0.04 / ic.std(), atol=1e-3)
    assert np.isclose(st["icir_ann"], st["icir"] * np.sqrt(25.2), atol=1e-2)
    assert st["win_rate"] == 1.0


def test_layered_monotonic_factor():
    """构造 因子=未来收益 的完美因子: 分组应单调, 多空为正。"""
    rng = np.random.default_rng(9)
    ret = pd.DataFrame(rng.normal(0, 0.02, size=(T, N)), index=IDX, columns=COLS)
    price = (1 + ret).cumprod() * 100
    daily_ret = price / price.shift(1) - 1
    horizon = 5
    fwd = price.shift(-(1 + horizon)) / price.shift(-1) - 1
    mask = pd.DataFrame(True, index=IDX, columns=COLS)
    out = M.layered_backtest(fwd, daily_ret, mask, n_q=5, horizon=horizon)
    assert out["monotonicity"] > 0.9
    assert out["ls_ann_ret"] > 0


def test_rank_autocorr_persistent_factor():
    base = _df(5).iloc[0]
    f = pd.DataFrame([base] * T, index=IDX)  # 完全不变的因子
    assert M.rank_autocorr(f, horizon=10) > 0.999


def test_value_corr_self():
    f = _df(6)
    assert M.value_corr(f, f) > 0.999
    assert abs(M.value_corr(f, _df(7))) < 0.2


def test_winsorize_bounds():
    f = _df(8)
    f.iloc[0, 0] = 1e6
    w = winsorize_mad(f, 5.0)
    assert w.iloc[0, 0] < 1e6
    med = f.iloc[0].median()
    assert w.iloc[0].max() <= med + 5 * 1.4826 * (f.iloc[0] - med).abs().median() + 1e-6


def test_zscore_rowwise():
    z = zscore(_df(10))
    assert np.allclose(z.mean(axis=1), 0, atol=1e-10)
    assert np.allclose(z.std(axis=1), 1, atol=1e-6)


def test_neutralize_industry_and_size():
    f = _df(11)
    ind = pd.DataFrame(np.tile(np.repeat([1, 2, 3], N // 3), (T, 1)),
                       index=IDX, columns=COLS, dtype=float)
    mv = _df(12).abs() + 1
    out = neutralize(f, ind, np.log(mv))
    row, g = out.iloc[3], ind.iloc[3]
    for code in (1, 2, 3):
        assert abs(row[g == code].mean()) < 0.15   # 行业均值近0(再经市值回归有残差)
    # 与log市值的相关应显著下降
    lm = np.log(mv).iloc[3]
    assert abs(np.corrcoef(out.iloc[3], lm)[0, 1]) < 0.1
