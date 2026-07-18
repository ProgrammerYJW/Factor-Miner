"""评估器: 装载特征/标签/掩码, 输出全周期x全时段指标(⭐ICIR为核心)与引擎快评适应度."""
from __future__ import annotations

import logging
from functools import cached_property

import numpy as np
import pandas as pd

from factor_miner.config import Config, get_config
from factor_miner.evaluation import metrics as M
from factor_miner.evaluation.preprocess import preprocess
from factor_miner.expression.nodes import EvalContext, Expr

log = logging.getLogger(__name__)

BASE_FEATURES = ["open", "high", "low", "close", "vwap", "volume", "amount",
                 "turnover", "free_turnover", "total_mv", "neg_mv",
                 "ep_ttm", "bp", "sp_ttm"]


class Evaluator:
    """用法: ev = Evaluator(); rep = ev.evaluate_full(expr); fit = ev.fast_fitness(expr, pool)."""

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or get_config()
        fd = self.cfg.features_dir
        self.features: dict[str, pd.DataFrame] = {
            n: pd.read_parquet(fd / f"{n}.parquet") for n in BASE_FEATURES
        }
        self.close = self.features["close"]
        self.calendar = self.close.index
        self.universe = pd.read_parquet(fd / "universe.parquet").astype(bool)
        lu = pd.read_parquet(fd / "limit_up_oneline.parquet").fillna(0.0)
        ld = pd.read_parquet(fd / "limit_down_oneline.parquet").fillna(0.0)
        # t日建仓不可行: t+1一字板(买不进/卖不出); float域移位避免bool->object
        nxt_lock = (lu.shift(-1).fillna(0.0) > 0.5) | (ld.shift(-1).fillna(0.0) > 0.5)
        self.tradable = self.universe & ~nxt_lock
        p = fd / "industry.parquet"
        self.industry = pd.read_parquet(p) if p.exists() else None
        self.log_mv = np.log(self.features["total_mv"].clip(lower=1.0))
        self.horizons = list(self.cfg["label"]["horizons"])
        self.primary_h = int(self.cfg["label"]["primary_horizon"])
        self.ctx = EvalContext(features=self.features, industry=self.industry)
        sp = self.cfg["split"]
        self.segments = {
            "train": (sp["train_start"], sp["train_end"]),
            "valid": (sp["valid_start"], sp["valid_end"]),
            "observe": (sp["observe_start"], sp["observe_end"] or str(self.calendar[-1].date())),
        }
        log.info("Evaluator就绪: %d天 x %d股, 特征%d个", *self.close.shape, len(self.features))

    # ---------- 标签 ----------
    @cached_property
    def labels(self) -> dict[int, pd.DataFrame]:
        """ret_H(t) = C_adj(t+1+H)/C_adj(t+1) - 1 (t收盘算因子, t+1收盘建仓)。"""
        out = {}
        for h in self.horizons:
            c1 = self.close.shift(-1)
            out[h] = (self.close.shift(-(1 + h)) / c1 - 1.0).astype(np.float32)
        return out

    @cached_property
    def daily_ret(self) -> pd.DataFrame:
        return (self.close / self.close.shift(1) - 1.0).astype(np.float32)

    def _seg_slice(self, seg: str) -> slice:
        a, b = self.segments[seg]
        idx = self.calendar
        return slice(idx.searchsorted(pd.Timestamp(a)),
                     idx.searchsorted(pd.Timestamp(b), side="right"))

    # ---------- 因子值 ----------
    def factor_values(self, expr: Expr, preprocessed: bool = True) -> pd.DataFrame:
        raw = expr.evaluate(self.ctx)
        if not preprocessed:
            return raw
        ev = self.cfg["evaluation"]
        return preprocess(
            raw, self.universe,
            industry=self.industry if ev["neutralize_industry"] else None,
            log_mv=self.log_mv if ev["neutralize_size"] else None,
            winsor_k=float(ev["winsor_mad"]),
        ).astype(np.float32)

    # ---------- 完整评估 ----------
    def evaluate_full(self, expr: Expr) -> dict:
        """全周期x全时段指标。返回 {'metrics': {...}, 'ic_series': {...}, 'factor': df}。"""
        f = self.factor_values(expr)
        nq = int(self.cfg["evaluation"]["n_quantiles"])
        report: dict[str, dict] = {}
        ic_store: dict[str, pd.Series] = {}
        for h in self.horizons:
            ic_all = M.daily_rank_ic(f, self.labels[h])
            ic_store[str(h)] = ic_all
            for seg in self.segments:
                sl = self._seg_slice(seg)
                key = f"h{h}_{seg}"
                stats = M.ic_stats(ic_all.iloc[sl], h)
                if h == self.primary_h:
                    stats.update(M.layered_backtest(
                        f.iloc[sl], self.daily_ret.iloc[sl],
                        self.tradable.iloc[sl], nq, h))
                    stats["rank_autocorr"] = M.rank_autocorr(f.iloc[sl], h)
                report[key] = stats
        report["coverage"] = M.coverage(f, self.universe)
        report["expression"] = expr.to_string()
        report["n_nodes"] = expr.n_nodes()
        return {"metrics": report, "ic_series": ic_store, "factor": f}

    # ---------- 引擎快评 ----------
    def fast_fitness(self, expr: Expr, pool: list[pd.DataFrame]) -> tuple[float, dict]:
        """训练段降采样适应度: |RankIC| - λ1·max|corr(池)| - λ2·节点数。失败返回-inf。"""
        try:
            f = self.factor_values(expr)
        except Exception as e:  # noqa: BLE001 非法表达式直接淘汰
            return float("-inf"), {"error": str(e)[:120]}
        step = int(self.cfg["evaluation"]["sample_step"])
        sl = self._seg_slice("train")
        fs = f.iloc[sl].iloc[::step]
        ys = self.labels[self.primary_h].iloc[sl].iloc[::step]
        cov = M.coverage(fs, self.universe.iloc[sl].iloc[::step])
        if not np.isfinite(cov) or cov < float(self.cfg["evaluation"]["min_coverage"]):
            return float("-inf"), {"error": f"coverage={cov}"}
        ic = M.daily_rank_ic(fs, ys)
        st = M.ic_stats(ic, self.primary_h)
        if not np.isfinite(st["ic_mean"]):
            return float("-inf"), {"error": "ic nan"}
        fit = abs(st["ic_mean"])
        max_corr = 0.0
        for other in pool:
            c = M.value_corr(fs, other.iloc[sl].iloc[::step])
            if np.isfinite(c):
                max_corr = max(max_corr, abs(c))
        fcfg = self.cfg["fitness"]
        fit -= float(fcfg["lambda_corr"]) * max_corr
        fit -= float(fcfg["lambda_complexity"]) * expr.n_nodes()
        info = {**st, "max_corr_pool": round(max_corr, 4), "coverage": cov}
        return float(fit), info
