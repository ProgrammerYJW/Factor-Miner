"""Web界面公共组件: 缓存加载器与指标格式化."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from factor_miner.config import get_config  # noqa: E402
from factor_miner.library import FactorLibrary  # noqa: E402


@st.cache_resource
def lib() -> FactorLibrary:
    return FactorLibrary()


@st.cache_resource(show_spinner="加载全量评估器(首次操作较慢)...")
def evaluator():
    from factor_miner.evaluation import Evaluator

    return Evaluator()


def cfg():
    return get_config()


SEG_NAMES = {"train": "训练段", "valid": "验证段", "observe": "观察段"}
METRIC_COLS = [
    ("rank_ic", "⭐RankIC"), ("ic_std", "IC标准差"), ("icir", "⭐IC/IR"),
    ("icir_ann", "年化IC/IR"), ("ic_skew", "偏度"),
    ("win_rate", "胜率"),
    ("n_days", "样本天数"),
]


def metrics_matrix(m: dict, horizons: list[int], seg: str) -> pd.DataFrame:
    rows = []
    for h in horizons:
        d = m.get(f"h{h}_{seg}", {})
        rows.append({"周期": f"{h}日", **{cn: d.get(k) for k, cn in METRIC_COLS}})
    return pd.DataFrame(rows).set_index("周期")


def fmt_summary(df: pd.DataFrame) -> pd.DataFrame:
    ph = int(cfg()["label"]["primary_horizon"])
    show = df[[c for c in [
        "id", "name", "engine", "status", "icir_train", "icir_valid",
        "rankic_train", "rankic_valid", "rank_autocorr", "coverage",
        "n_nodes", "created_at",
    ] if c in df.columns]].copy()
    return show.rename(columns={
        "id": "ID", "name": "名称", "engine": "引擎", "status": "状态",
        "icir_train": f"⭐IC/IR({ph}日,训练)", "icir_valid": f"IC/IR({ph}日,验证)",
        "rankic_train": "⭐RankIC(训练)", "rankic_valid": "RankIC(验证)",
        "rank_autocorr": "秩自相关(低换手↑)", "coverage": "覆盖率",
        "n_nodes": "节点数", "created_at": "入库时间",
    })
