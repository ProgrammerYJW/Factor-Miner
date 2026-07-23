"""可配置准入规则引擎: Web界面可编辑, 存 artifacts/admission_rules.json.

规则形式(用户需求): 任一评估指标 与 用户设定数值 的大小关系(可带绝对值)。
指标含两类: ①分周期/时段指标(IC均值/IC÷IR/胜率/多空绩效/换手等)
           ②全局指标(与已有因子相关性 = 候选因子与库内活跃因子相关系数绝对值的最大者,
             覆盖率同为全局但作为专项字段)。
另有专项: 覆盖率下限、可开关的"验证段IC须与训练段同号"。所有规则之间为"与"关系。
引擎与Web每次提交/保存都即时读写本文件, 修改立即生效, 无需重启挖掘进程。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from factor_miner.config import Config, get_config

# 可用于规则的指标 (与Web界面展示一致)
METRIC_CHOICES = {
    "rank_ic": "RankIC", "ic_std": "IC标准差", "icir": "IC/IR(=IC均值÷IC标准差)",
    "icir_ann": "年化IC/IR", "ic_skew": "偏度",
    "ls_mdd": "多空最大回撤", "top_turnover": "Top组换手率",
    "rank_autocorr": "秩自相关", "max_corr_with_library": "与已有因子相关性",
}
# 全局指标: 不分周期/时段 (界面上周期/时段选"全局")
GLOBAL_METRICS = {"max_corr_with_library"}
OP_CHOICES = {"abs>=": "|值| ≥", "abs<=": "|值| ≤", ">=": "≥", "<=": "≤"}
SEG_CHOICES = {"train": "训练段", "valid": "验证段", "observe": "观察段"}


@dataclass
class Rule:
    metric: str                 # METRIC_CHOICES 之一
    horizon: int                # 1/5/10/20; 全局指标为 0
    segment: str                # train/valid/observe; 全局指标为 "global"
    op: str                     # abs>= / abs<= / >= / <=
    threshold: float
    enabled: bool = True
    note: str = ""

    def check(self, metrics: dict) -> tuple[bool, str]:
        if self.metric in GLOBAL_METRICS:
            v = metrics.get(self.metric)
            label = METRIC_CHOICES.get(self.metric, self.metric)
        else:
            v = metrics.get(f"h{self.horizon}_{self.segment}", {}).get(self.metric)
            label = (f"{SEG_CHOICES.get(self.segment, self.segment)}{self.horizon}日 "
                     f"{METRIC_CHOICES.get(self.metric, self.metric)}")
        if v is None or not np.isfinite(v):
            return False, f"{label} 缺失"
        x = abs(v) if self.op.startswith("abs") else v
        ok = x >= self.threshold if self.op.endswith(">=") else x <= self.threshold
        return ok, "" if ok else (f"{label}={v:.4g} 不满足 "
                                  f"{OP_CHOICES[self.op]} {self.threshold:g}")


@dataclass
class RuleSet:
    rules: list[Rule] = field(default_factory=list)
    min_coverage: float = 0.6               # 专项: 覆盖率下限
    require_same_sign: bool = True          # 专项: 验证段IC与训练段同号
    primary_horizon: int = 10

    # ---------- 持久化 ----------
    @staticmethod
    def path(cfg: Config | None = None) -> Path:
        return (cfg or get_config()).artifacts_dir / "admission_rules.json"

    @classmethod
    def load(cls, cfg: Config | None = None) -> "RuleSet":
        cfg = cfg or get_config()
        p = cls.path(cfg)
        if not p.exists():
            rs = cls.default(cfg)
            rs.save(cfg)
            return rs
        d = json.loads(p.read_text("utf-8"))
        rules = [Rule(**r) for r in d.get("rules", [])]
        # 迁移旧版: 顶层 max_corr_with_library 字段 -> 一条全局相关性规则
        legacy = d.get("max_corr_with_library")
        if legacy is not None and not any(r.metric == "max_corr_with_library" for r in rules):
            rules.append(Rule("max_corr_with_library", 0, "global", "<=",
                              float(legacy), note="自旧版配置迁移"))
        return cls(rules=rules,
                   min_coverage=float(d.get("min_coverage", 0.6)),
                   require_same_sign=bool(d.get("require_same_sign", True)),
                   primary_horizon=int(d.get("primary_horizon", 10)))

    def save(self, cfg: Config | None = None) -> None:
        p = self.path(cfg)
        p.parent.mkdir(parents=True, exist_ok=True)
        d = {"rules": [asdict(r) for r in self.rules],
             "min_coverage": self.min_coverage,
             "require_same_sign": self.require_same_sign,
             "primary_horizon": self.primary_horizon}
        p.write_text(json.dumps(d, ensure_ascii=False, indent=1), "utf-8")

    @classmethod
    def default(cls, cfg: Config | None = None) -> "RuleSet":
        """从 settings.toml [admission] 播种默认规则(向后兼容原有门槛)。"""
        cfg = cfg or get_config()
        ac = cfg["admission"]
        h = int(cfg["label"]["primary_horizon"])
        return cls(
            rules=[
                Rule("ic_mean", h, "train", "abs>=", float(ac["min_abs_rank_ic"]),
                     note="训练段IC下限(默认)"),
                Rule("icir", h, "train", "abs>=", float(ac["min_abs_icir"]),
                     note="训练段IC/IR下限(默认)"),
                Rule("icir", h, "valid", "abs>=", float(ac["valid_min_abs_icir"]),
                     note="验证段IC/IR下限(默认)"),
                Rule("max_corr_with_library", 0, "global", "<=",
                     float(ac["max_corr_with_library"]),
                     note="与已有因子相关性上限(默认)"),
            ],
            min_coverage=float(cfg["evaluation"]["min_coverage"]),
            primary_horizon=h,
        )

    # ---------- 判定 ----------
    def check_metrics(self, metrics: dict) -> tuple[bool, list[str]]:
        """先行检查: 非全局规则 + 覆盖率 + 同号(相关性等全局规则在 check_global)。"""
        fails: list[str] = []
        cov = metrics.get("coverage")
        if cov is None or not np.isfinite(cov) or cov < self.min_coverage:
            fails.append(f"覆盖率={cov} < 下限{self.min_coverage:g}")
        if self.require_same_sign:
            h = self.primary_horizon
            tr = metrics.get(f"h{h}_train", {}).get("ic_mean")
            va = metrics.get(f"h{h}_valid", {}).get("ic_mean")
            if tr is None or va is None or not (np.isfinite(tr) and np.isfinite(va)) \
                    or np.sign(tr) != np.sign(va):
                fails.append(f"验证段IC({va})与训练段IC({tr})不同号")
        for r in self.rules:
            if r.enabled and r.metric not in GLOBAL_METRICS:
                ok, why = r.check(metrics)
                if not ok:
                    fails.append(why)
        return len(fails) == 0, fails

    def check_global(self, metrics: dict) -> tuple[bool, list[str]]:
        """后置检查全局规则(需先把 max_corr_with_library 等写入 metrics)。"""
        fails = []
        for r in self.rules:
            if r.enabled and r.metric in GLOBAL_METRICS:
                ok, why = r.check(metrics)
                if not ok:
                    fails.append(why)
        return len(fails) == 0, fails

    def check_all(self, metrics: dict) -> tuple[bool, list[str]]:
        ok1, f1 = self.check_metrics(metrics)
        ok2, f2 = self.check_global(metrics)
        return ok1 and ok2, f1 + f2

    def threshold_of(self, metric: str, segment: str = "train",
                     op_prefix: str = "abs") -> float | None:
        """给挖掘引擎预筛用: 取该指标当前启用规则的阈值(无则None)。"""
        for r in self.rules:
            if (r.enabled and r.metric == metric and r.segment == segment
                    and r.horizon == self.primary_horizon and r.op.startswith(op_prefix)):
                return float(r.threshold)
        return None
