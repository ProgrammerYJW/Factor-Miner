"""准入管道: 候选表达式 -> 可配置规则检查 -> 方向归一(IC为正) -> 全量评估 -> 入库.

筛选标准由 RuleSet(artifacts/admission_rules.json) 定义, 可在Web界面"筛选标准"页修改,
每次提交都重新读取 —— 修改即时生效, 无需重启挖掘进程。
"""
from __future__ import annotations

import logging

import numpy as np

from factor_miner.config import Config, get_config
from factor_miner.evaluation import metrics as M
from factor_miner.evaluation.evaluator import Evaluator
from factor_miner.expression.nodes import Expr
from factor_miner.library.rules import RuleSet
from factor_miner.library.store import FactorLibrary

log = logging.getLogger(__name__)


class Admission:
    def __init__(self, evaluator: Evaluator, library: FactorLibrary,
                 cfg: Config | None = None):
        self.ev = evaluator
        self.lib = library
        self.cfg = cfg or get_config()

    def submit(self, expr: Expr, engine: str) -> tuple[bool, str, int | None]:
        """返回 (是否入库, 原因, 因子id)。规则现读现用, Web端修改即时生效。"""
        rules = RuleSet.load(self.cfg)
        if self.lib.exists(expr.key()):
            return False, "重复表达式(canonical)", None

        rep = self.ev.evaluate_full(expr)
        tr = rep["metrics"].get(f"h{self.ev.primary_h}_train", {})
        # 方向归一: 训练段IC为负则取相反数重评
        if np.isfinite(tr.get("ic_mean", np.nan)) and tr["ic_mean"] < 0:
            expr = Expr.call("neg", expr)
            if self.lib.exists(expr.key()):
                return False, "重复表达式(取负后)", None
            rep = self.ev.evaluate_full(expr)

        ok, fails = rules.check_metrics(rep["metrics"])
        if not ok:
            return False, "; ".join(fails[:3]), None

        max_corr = 0.0
        for fid, other in self.lib.active_value_matrices().items():
            c = M.value_corr(rep["factor"], other, step=5)
            if np.isfinite(c) and abs(c) > max_corr:
                max_corr = abs(c)
        rep["metrics"]["max_corr_with_library"] = round(max_corr, 4)
        ok, fails = rules.check_global(rep["metrics"])
        if not ok:
            return False, "; ".join(fails[:3]), None

        fid = self.lib.add(
            expression=expr.to_string(), expr_key=expr.key(), engine=engine,
            metrics=rep["metrics"], factor=rep["factor"], ic_series=rep["ic_series"],
        )
        icir = rep["metrics"][f"h{self.ev.primary_h}_train"].get("icir")
        log.info("[OK] 入库 #%d %s IR10=%.3f", fid, expr.to_string()[:60], icir or 0)
        return True, "ok", fid
