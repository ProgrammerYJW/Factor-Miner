"""总览页课本公式渲染与小表兜底测试."""
import pytest

from factor_miner.expression.parser import parse
from factor_miner.expression.pretty import to_textbook


def test_overview_cell_renders_katex():
    exprs = [
        "neg(decay_linear(free_turnover, 10))",
        "ts_sum(sub(ts_cov(ts_slope(close, 60), volume, 10), vwap), 40)",
        "cs_rank(ts_corr(cs_rank(close), cs_rank(volume), 10))",
    ]
    from factor_miner.webapp.common import fmt_summary
    import pandas as pd

    rows = [{"id": i + 1, "name": f"f{i + 1}", "engine": "GP", "status": "active",
             "icir10_train": 0.5 + i * 0.1, "icir10_valid": 0.4 + i * 0.1,
             "ic10_train": 0.03 + i * 0.01, "ic10_valid": 0.02 + i * 0.01,
             "ls_ann_train": 0.1, "ls_sharpe_train": 1.0,
             "rank_autocorr": 0.5, "coverage": 0.9, "n_nodes": i * 5 + 5,
             "created_at": "2026-07-18", "expression": s}
            for i, s in enumerate(exprs)]
    df = pd.DataFrame(rows)
    s = fmt_summary(df)
    tex = s["📐表达式(课本格式)"].tolist()
    assert r"\operatorname{WMA}" in tex[0]
    assert r"\operatorname{Corr}" in tex[2]
    for t in tex:
        assert t.startswith(r"\displaystyle "), f"缺displaystyle: {t[:30]}"


def test_pretty_crash_returns_raw():
    from factor_miner.webapp.common import fmt_summary
    import pandas as pd

    rows = [{"id": 99, "name": "bad", "engine": "GP", "status": "active",
             "icir10_train": 0.3, "icir10_valid": 0.3,
             "ic10_train": 0.02, "ic10_valid": 0.02,
             "ls_ann_train": 0.0, "ls_sharpe_train": 0.0,
             "rank_autocorr": 0.0, "coverage": 0.0, "n_nodes": 0,
             "created_at": "", "expression": "(crashes"}]
    df = pd.DataFrame(rows)
    s = fmt_summary(df)
    assert s["📐表达式(课本格式)"].iloc[0] == "(crashes"  # 渲染失败原地保留
