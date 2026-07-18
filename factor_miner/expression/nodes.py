"""表达式AST: 不可变节点 + 求值 + 序列化 + 规范化哈希 + 合法性检查."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from factor_miner.expression.ops import OP_REGISTRY


@dataclass
class EvalContext:
    """求值上下文: 基础特征宽表 + 可选行业代码矩阵."""

    features: dict[str, pd.DataFrame]
    industry: Optional[pd.DataFrame] = None

    def get(self, name: str) -> pd.DataFrame:
        if name not in self.features:
            raise KeyError(f"未知特征端子: {name}")
        return self.features[name]


@dataclass(frozen=True)
class Expr:
    """op 为 None 时是特征端子(feature); 否则为算子节点。window 仅 ts_* 使用。"""

    op: Optional[str] = None
    children: tuple["Expr", ...] = field(default_factory=tuple)
    feature: Optional[str] = None
    window: Optional[int] = None

    # ---------- 构造 ----------
    @staticmethod
    def leaf(feature: str) -> "Expr":
        return Expr(feature=feature)

    @staticmethod
    def call(op: str, *children: "Expr", window: int | None = None) -> "Expr":
        spec = OP_REGISTRY[op]
        if len(children) != spec.n_args:
            raise ValueError(f"{op} 需要 {spec.n_args} 个子表达式, 实得 {len(children)}")
        if spec.window and window is None:
            raise ValueError(f"{op} 缺少窗口参数")
        if not spec.window:
            window = None
        return Expr(op=op, children=tuple(children), window=window)

    # ---------- 属性 ----------
    @property
    def is_leaf(self) -> bool:
        return self.op is None

    def n_nodes(self) -> int:
        return 1 + sum(c.n_nodes() for c in self.children)

    def depth(self) -> int:
        return 1 if self.is_leaf else 1 + max(c.depth() for c in self.children)

    def iter_nodes(self):
        yield self
        for c in self.children:
            yield from c.iter_nodes()

    def uses_industry(self) -> bool:
        return any((not n.is_leaf) and OP_REGISTRY[n.op].needs_industry
                   for n in self.iter_nodes())

    # ---------- 求值 ----------
    def evaluate(self, ctx: EvalContext) -> pd.DataFrame:
        if self.is_leaf:
            return ctx.get(self.feature)
        spec = OP_REGISTRY[self.op]
        args = [c.evaluate(ctx) for c in self.children]
        if spec.needs_industry:
            if ctx.industry is None:
                raise ValueError(f"{self.op} 需要行业矩阵, 但上下文未提供")
            return spec.fn(*args, ctx.industry)
        if spec.window:
            return spec.fn(*args, self.window)
        return spec.fn(*args)

    # ---------- 序列化 ----------
    def to_string(self) -> str:
        if self.is_leaf:
            return self.feature
        parts = [c.to_string() for c in self.children]
        if self.window is not None:
            parts.append(str(self.window))
        return f"{self.op}({', '.join(parts)})"

    def __str__(self) -> str:  # noqa: DunderStr
        return self.to_string()

    def canonical(self) -> str:
        """规范形式: 交换律算子的子节点按字典序排序, 用于查重。"""
        if self.is_leaf:
            return self.feature
        spec = OP_REGISTRY[self.op]
        parts = [c.canonical() for c in self.children]
        if spec.commutative:
            parts = sorted(parts)
        if self.window is not None:
            parts.append(str(self.window))
        return f"{self.op}({','.join(parts)})"

    def key(self) -> str:
        return hashlib.md5(self.canonical().encode()).hexdigest()[:16]

    # ---------- 合法性 ----------
    def validate(self, features: set[str], windows: set[int],
                 max_depth: int, max_nodes: int) -> list[str]:
        errs: list[str] = []
        for n in self.iter_nodes():
            if n.is_leaf:
                if n.feature not in features:
                    errs.append(f"未知特征: {n.feature}")
            else:
                if n.op not in OP_REGISTRY:
                    errs.append(f"未知算子: {n.op}")
                elif OP_REGISTRY[n.op].window and n.window not in windows:
                    errs.append(f"{n.op} 非法窗口: {n.window}")
        if self.depth() > max_depth:
            errs.append(f"深度超限: {self.depth()} > {max_depth}")
        if self.n_nodes() > max_nodes:
            errs.append(f"节点数超限: {self.n_nodes()} > {max_nodes}")
        return errs
