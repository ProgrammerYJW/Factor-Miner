"""引擎快评上下文: 训练段全行 + 股票子采样, 保证时序算子语义正确.

主进程与GP工作进程都用它: 加载特征parquet -> 切训练段 -> 抽样N_max只股票 ->
构建 EvalContext / 主周期标签 / 股票池掩码。IC计算阶段再做隔日降采样。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from factor_miner.config import Config, get_config
from factor_miner.evaluation.evaluator import BASE_FEATURES
from factor_miner.expression.nodes import EvalContext

log = logging.getLogger(__name__)

FAST_MAX_STOCKS = 2000
FAST_SEED = 20260717


class FastContext:
    """轻量快评环境(约百MB), 适应度 = |RankIC(隔step日)| 的计算底座。"""

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or get_config()
        fd = self.cfg.features_dir
        sp = self.cfg["split"]
        close = pd.read_parquet(fd / "close.parquet")
        idx = close.index
        sl = slice(idx.searchsorted(pd.Timestamp(sp["train_start"])),
                   idx.searchsorted(pd.Timestamp(sp["train_end"]), side="right"))
        uni_full = pd.read_parquet(fd / "universe.parquet").iloc[sl]
        # 固定种子抽样股票列(在训练段内曾入池的股票中抽)
        ever = uni_full.sum(axis=0) > 0
        cols = uni_full.columns[ever]
        if len(cols) > FAST_MAX_STOCKS:
            rng = np.random.default_rng(FAST_SEED)
            cols = pd.Index(sorted(rng.choice(cols, FAST_MAX_STOCKS, replace=False)))
        self.cols = cols
        feats = {}
        for n in BASE_FEATURES:
            feats[n] = pd.read_parquet(fd / f"{n}.parquet").iloc[sl][cols].astype(np.float32)
        p = fd / "industry.parquet"
        ind = pd.read_parquet(p).iloc[sl][cols] if p.exists() else None
        self.ctx = EvalContext(features=feats, industry=ind)
        self.universe = uni_full[cols].astype(bool)
        c = feats["close"]
        h = int(self.cfg["label"]["primary_horizon"])
        self.label = (c.shift(-(1 + h)) / c.shift(-1) - 1.0).astype(np.float32)
        self.step = int(self.cfg["evaluation"]["sample_step"])
        self.horizon = h
        log.info("FastContext: %d天 x %d股 (训练段)", *c.shape)

    def factor_on_grid(self, expr) -> pd.DataFrame:
        """求值并限定股票池, 隔step日采样(供IC与池相关性)。"""
        f = expr.evaluate(self.ctx)
        return f.where(self.universe).iloc[:: self.step]

    def label_on_grid(self) -> pd.DataFrame:
        return self.label.iloc[:: self.step]
