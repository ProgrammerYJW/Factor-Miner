"""RL token空间: RPN逐token构造表达式 + 动作合法性掩码 + 解码.

设计: 特征端子/(算子,窗口)融合token/END 组成动作表; 栈式RPN语义:
  - 特征token: 压栈
  - 算子token: 弹出n_args个操作数, 压回结果
  - END: 栈深恰为1且至少含一个算子时合法, 结束回合
掩码同时保证长度预算内可收敛(剩余步数足够把栈收敛到1)。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from factor_miner.expression.nodes import Expr
from factor_miner.expression.ops import OP_REGISTRY


@dataclass(frozen=True)
class Token:
    kind: str                  # feature / op / end
    name: str
    op: str | None = None
    window: int | None = None
    n_args: int = 0


class TokenSpace:
    def __init__(self, features: list[str], windows: list[int],
                 max_tokens: int = 24, max_stack: int = 8,
                 use_industry_ops: bool = False):
        self.max_tokens = max_tokens
        self.max_stack = max_stack
        toks: list[Token] = [Token("end", "<END>")]
        for f in features:
            toks.append(Token("feature", f))
        for spec in OP_REGISTRY.values():
            if spec.needs_industry and not use_industry_ops:
                continue
            if spec.window:
                for w in windows:
                    toks.append(Token("op", f"{spec.name}_{w}", op=spec.name,
                                      window=w, n_args=spec.n_args))
            else:
                toks.append(Token("op", spec.name, op=spec.name, n_args=spec.n_args))
        self.tokens = toks
        self.n_actions = len(toks)
        self.END = 0

    def valid_mask(self, seq: list[int]) -> np.ndarray:
        """当前前缀下每个动作是否合法。"""
        stack, n_ops = 0, 0
        for a in seq:
            t = self.tokens[a]
            if t.kind == "feature":
                stack += 1
            elif t.kind == "op":
                stack += 1 - t.n_args
                n_ops += 1
        remaining = self.max_tokens - len(seq)
        mask = np.zeros(self.n_actions, dtype=bool)
        for i, t in enumerate(self.tokens):
            if t.kind == "end":
                mask[i] = stack == 1 and n_ops >= 1
            elif t.kind == "feature":
                # 压栈后仍需 (stack+1-1) 个二元算子 + END 步收敛
                mask[i] = stack < self.max_stack and stack + 1 <= remaining - 1
            else:
                if stack >= t.n_args:
                    new_stack = stack + 1 - t.n_args
                    mask[i] = new_stack <= remaining - 1 or (new_stack == 1)
        return mask

    def decode(self, seq: list[int]) -> Expr:
        stack: list[Expr] = []
        for a in seq:
            t = self.tokens[a]
            if t.kind == "end":
                break
            if t.kind == "feature":
                stack.append(Expr.leaf(t.name))
            else:
                args = [stack.pop() for _ in range(t.n_args)][::-1]
                stack.append(Expr.call(t.op, *args, window=t.window))
        if len(stack) != 1:
            raise ValueError(f"非法token序列, 终态栈深={len(stack)}")
        return stack[0]
