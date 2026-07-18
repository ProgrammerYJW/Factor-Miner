"""随机表达式采样与遗传变异算子(GP引擎核心工具, RL不使用).

ramped half-and-half 初始化; 子树交叉/子树变异/点变异/hoist变异。
所有生成结果都保证语法合法(窗口参数只取白名单)。
"""
from __future__ import annotations

import numpy as np

from factor_miner.expression.nodes import Expr
from factor_miner.expression.ops import OP_REGISTRY

_ELEM_TS = [s for s in OP_REGISTRY.values() if s.kind in ("elementwise", "ts")]
_ALL_OPS = list(OP_REGISTRY.values())


class ExprSampler:
    def __init__(self, features: list[str], windows: list[int],
                 max_depth: int = 8, max_nodes: int = 24,
                 use_industry_ops: bool = True, seed: int = 0):
        self.features = list(features)
        self.windows = list(windows)
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.ops = [s for s in _ALL_OPS if use_industry_ops or not s.needs_industry]
        self.rng = np.random.default_rng(seed)

    # ---------- 生成 ----------
    def _leaf(self) -> Expr:
        return Expr.leaf(str(self.rng.choice(self.features)))

    def _call(self, spec, children) -> Expr:
        w = int(self.rng.choice(self.windows)) if spec.window else None
        return Expr.call(spec.name, *children, window=w)

    def grow(self, depth: int) -> Expr:
        if depth <= 1 or self.rng.random() < 0.3:
            return self._leaf()
        spec = self.ops[self.rng.integers(len(self.ops))]
        return self._call(spec, [self.grow(depth - 1) for _ in range(spec.n_args)])

    def full(self, depth: int) -> Expr:
        if depth <= 1:
            return self._leaf()
        spec = self.ops[self.rng.integers(len(self.ops))]
        return self._call(spec, [self.full(depth - 1) for _ in range(spec.n_args)])

    def ramped(self, n: int, d_min: int = 2, d_max: int = 6) -> list[Expr]:
        out = []
        for i in range(n):
            d = int(self.rng.integers(d_min, d_max + 1))
            e = self.full(d) if i % 2 == 0 else self.grow(d)
            out.append(e)
        return out

    # ---------- 变异 ----------
    def _nodes_with_path(self, e: Expr, path=()) -> list[tuple]:
        out = [(path, e)]
        for i, c in enumerate(e.children):
            out.extend(self._nodes_with_path(c, (*path, i)))
        return out

    @staticmethod
    def _replace(e: Expr, path: tuple, new: Expr) -> Expr:
        if not path:
            return new
        i = path[0]
        children = list(e.children)
        children[i] = ExprSampler._replace(children[i], path[1:], new)
        return Expr(op=e.op, children=tuple(children), feature=e.feature, window=e.window)

    def _rand_path(self, e: Expr) -> tuple:
        nodes = self._nodes_with_path(e)
        return nodes[self.rng.integers(len(nodes))][0]

    def crossover(self, a: Expr, b: Expr) -> Expr:
        pa = self._rand_path(a)
        nodes_b = self._nodes_with_path(b)
        sub = nodes_b[self.rng.integers(len(nodes_b))][1]
        return self._clip(self._replace(a, pa, sub))

    def mutate_subtree(self, e: Expr) -> Expr:
        p = self._rand_path(e)
        return self._clip(self._replace(e, p, self.grow(int(self.rng.integers(2, 5)))))

    def mutate_hoist(self, e: Expr) -> Expr:
        nodes = self._nodes_with_path(e)
        return nodes[self.rng.integers(len(nodes))][1]

    def mutate_point(self, e: Expr) -> Expr:
        """同元数算子替换 / 特征替换 / 窗口替换, 保持结构不变。"""
        path = self._rand_path(e)
        node = e
        for i in path:
            node = node.children[i]
        if node.is_leaf:
            return self._replace(e, path, self._leaf())
        spec = OP_REGISTRY[node.op]
        cands = [s for s in self.ops
                 if s.n_args == spec.n_args and s.window == spec.window
                 and s.name != node.op]
        if cands and self.rng.random() < 0.7:
            new_spec = cands[self.rng.integers(len(cands))]
            new = Expr.call(new_spec.name, *node.children,
                            window=node.window if new_spec.window else None)
        elif spec.window:
            new = Expr.call(node.op, *node.children,
                            window=int(self.rng.choice(self.windows)))
        else:
            return self._replace(e, path, self._leaf())
        return self._replace(e, path, new)

    def _clip(self, e: Expr) -> Expr:
        """超限个体截断: 深度/节点数超限时用其随机子树替代(直至合规)。"""
        for _ in range(20):
            if e.depth() <= self.max_depth and e.n_nodes() <= self.max_nodes:
                return e
            nodes = [n for _, n in self._nodes_with_path(e)
                     if n.depth() < e.depth() or n.n_nodes() < e.n_nodes()]
            e = nodes[self.rng.integers(len(nodes))]
        return self._leaf()
