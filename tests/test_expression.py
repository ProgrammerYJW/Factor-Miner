"""M2 表达式/算子层单元测试: 对照朴素参考实现."""
import numpy as np
import pandas as pd
import pytest

from factor_miner.expression import EvalContext, Expr, parse
from factor_miner.expression.ops import (
    op_cs_indneutral, op_decay_linear, op_ts_slope,
)

RNG = np.random.default_rng(7)
T, N = 60, 8
IDX = pd.date_range("2024-01-01", periods=T, freq="B")
COLS = [str(i) for i in range(N)]


def mk(seed=0):
    r = np.random.default_rng(seed)
    return pd.DataFrame(r.normal(size=(T, N)), index=IDX, columns=COLS)


@pytest.fixture
def ctx():
    return EvalContext(features={"close": mk(1).abs() + 1, "volume": mk(2).abs() * 1e6})


# ---------- 解析器 ----------
def test_parse_roundtrip():
    s = "cs_rank(ts_corr(cs_rank(close), cs_rank(volume), 10))"
    e = parse(s)
    assert e.to_string() == s
    assert parse(e.to_string()).key() == e.key()


def test_parse_window_position():
    e = parse("ts_mean(close, 20)")
    assert e.window == 20 and e.children[0].feature == "close"
    with pytest.raises(Exception):
        parse("ts_mean(close)")          # 缺窗口
    with pytest.raises(Exception):
        parse("badop(close, 5)")         # 未知算子


def test_canonical_commutative():
    a, b = parse("add(close, volume)"), parse("add(volume, close)")
    assert a.key() == b.key()
    c, d = parse("sub(close, volume)"), parse("sub(volume, close)")
    assert c.key() != d.key()


def test_validate():
    e = parse("ts_mean(close, 7)")
    errs = e.validate({"close"}, {3, 5, 10}, max_depth=8, max_nodes=24)
    assert any("窗口" in x for x in errs)
    deep = parse("neg(neg(neg(neg(close))))")
    assert any("深度" in x for x in deep.validate({"close"}, set(), 3, 24))


# ---------- 算子正确性 ----------
def test_ts_mean_matches_reference(ctx):
    out = parse("ts_mean(close, 5)").evaluate(ctx)
    ref = ctx.get("close").rolling(5, min_periods=2).mean()
    pd.testing.assert_frame_equal(out, ref)


def test_ts_rank_last_pct(ctx):
    out = parse("ts_rank(close, 5)").evaluate(ctx)
    x = ctx.get("close").iloc[:, 0]
    w = x.iloc[10:15]
    expect = w.rank(pct=True).iloc[-1]
    assert np.isclose(out.iloc[14, 0], expect)


def test_ts_corr_vs_loop(ctx):
    out = parse("ts_corr(close, volume, 10)").evaluate(ctx)
    a, b = ctx.get("close").iloc[:, 3], ctx.get("volume").iloc[:, 3]
    ref = a.iloc[20:30].corr(b.iloc[20:30])
    assert np.isclose(out.iloc[29, 3], ref, atol=1e-6)


def test_ts_slope_vs_polyfit():
    x = mk(5)
    out = op_ts_slope(x, 10)
    col = x.iloc[20:30, 2].to_numpy()
    ref = np.polyfit(np.arange(10), col, 1)[0]
    assert np.isclose(out.iloc[29, 2], ref, atol=1e-8)


def test_decay_linear_manual():
    x = mk(6)
    out = op_decay_linear(x, 3)
    w = np.array([3, 2, 1], dtype=float) / 6  # 今日权重最大
    ref = x.iloc[12, 4] * w[0] + x.iloc[11, 4] * w[1] + x.iloc[10, 4] * w[2]
    assert np.isclose(out.iloc[12, 4], ref, atol=1e-8)


def test_div_zero_is_nan(ctx):
    zero = ctx.get("close") * 0.0
    ctx.features["zero"] = zero
    out = parse("div(close, zero)").evaluate(ctx)
    assert out.isna().all().all()


def test_cs_zscore_rowwise(ctx):
    out = parse("cs_zscore(close)").evaluate(ctx)
    row = ctx.get("close").iloc[7]
    ref = (row - row.mean()) / row.std()
    np.testing.assert_allclose(out.iloc[7].to_numpy(), ref.to_numpy(), atol=1e-6)


def test_cs_indneutral_group_demean():
    x = mk(9)
    ind = pd.DataFrame(np.tile([1, 1, 1, 2, 2, 2, 3, 3], (T, 1)),
                       index=IDX, columns=COLS, dtype=float)
    out = op_cs_indneutral(x, ind)
    g1 = x.iloc[5, :3]
    np.testing.assert_allclose(out.iloc[5, :3].to_numpy(),
                               (g1 - g1.mean()).to_numpy(), atol=1e-6)
    # 每行业组内均值应为0
    assert abs(out.iloc[5, 3:6].mean()) < 1e-6


def test_nan_propagation(ctx):
    c = ctx.get("close").copy()
    c.iloc[10, 0] = np.nan
    ctx.features["close"] = c
    out = parse("ts_delta(close, 1)").evaluate(ctx)
    assert np.isnan(out.iloc[10, 0]) and np.isnan(out.iloc[11, 0])


def test_expr_metrics():
    e = parse("cs_rank(ts_corr(cs_rank(close), cs_rank(volume), 10))")
    assert e.n_nodes() == 6 and e.depth() == 4
    assert not e.uses_industry()
    assert parse("cs_indneutral(close)").uses_industry()
