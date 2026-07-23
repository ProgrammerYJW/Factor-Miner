"""GP工作进程: 每个进程持有一份 FastContext, 批量评估表达式适应度.

Windows spawn 模式: initializer 在子进程内一次性加载特征数据;
因子池(库内已入库因子)以表达式字符串传入, 进程内缓存其因子网格。
"""
from __future__ import annotations

import numpy as np

from factor_miner.evaluation import metrics as M
from factor_miner.evaluation.fast_ctx import FastContext
from factor_miner.expression.parser import parse

_CTX: FastContext | None = None
_POOL_CACHE: dict[str, object] = {}


def init_worker() -> None:
    global _CTX
    import logging

    logging.basicConfig(level=logging.WARNING)
    _CTX = FastContext()


def _pool_grid(expr_str: str):
    if expr_str not in _POOL_CACHE:
        try:
            _POOL_CACHE[expr_str] = _CTX.factor_on_grid(parse(expr_str))
        except Exception:  # noqa: BLE001
            _POOL_CACHE[expr_str] = None
    return _POOL_CACHE[expr_str]


def eval_one(expr_str: str, pool_exprs: list[str],
             lambda_corr: float, lambda_complexity: float,
             min_coverage: float) -> tuple[float, dict]:
    """返回 (fitness, info)。任何异常 -> (-inf, error)。"""
    assert _CTX is not None, "worker 未初始化"
    try:
        expr = parse(expr_str)
        f = _CTX.factor_on_grid(expr)
    except Exception as e:  # noqa: BLE001
        return float("-inf"), {"error": str(e)[:120]}
    uni = _CTX.universe.iloc[:: _CTX.step]
    cov = M.coverage(f, uni)
    if not np.isfinite(cov) or cov < min_coverage:
        return float("-inf"), {"error": f"coverage={cov}"}
    ic = M.daily_rank_ic(f, _CTX.label_on_grid())
    st = M.ic_stats(ic, _CTX.horizon)
    if not np.isfinite(st["ic_mean"]):
        return float("-inf"), {"error": "ic nan"}
    max_corr = 0.0
    for ps in pool_exprs:
        g = _pool_grid(ps)
        if g is None:
            continue
        c = M.value_corr(f, g, step=2)
        if np.isfinite(c):
            max_corr = max(max_corr, abs(c))
    fit = abs(st["ic_mean"]) - lambda_corr * max_corr \
        - lambda_complexity * expr.n_nodes()
    return float(fit), {**st, "max_corr_pool": round(max_corr, 4),
                        "coverage": cov}


def eval_batch(expr_strs: list[str], pool_exprs: list[str],
               lambda_corr: float, lambda_complexity: float,
               min_coverage: float) -> list[tuple[float, dict]]:
    return [eval_one(s, pool_exprs, lambda_corr, lambda_complexity, min_coverage)
            for s in expr_strs]
