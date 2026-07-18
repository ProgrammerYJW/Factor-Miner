"""表达式字符串解析器: parse('cs_rank(ts_corr(close, volume, 10))') -> Expr.

与 Expr.to_string() 互为逆运算(往返一致)。
"""
from __future__ import annotations

import re

from factor_miner.expression.nodes import Expr
from factor_miner.expression.ops import OP_REGISTRY

_TOKEN = re.compile(r"\s*([A-Za-z_][A-Za-z_0-9]*|\d+|[(),])")


class ParseError(ValueError):
    pass


def _tokenize(s: str) -> list[str]:
    out, pos = [], 0
    while pos < len(s):
        m = _TOKEN.match(s, pos)
        if not m:
            if s[pos:].strip():
                raise ParseError(f"位置 {pos} 非法字符: {s[pos:pos + 10]!r}")
            break
        out.append(m.group(1))
        pos = m.end()
    return out


class _Parser:
    def __init__(self, tokens: list[str]):
        self.toks = tokens
        self.i = 0

    def peek(self) -> str | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self, expect: str | None = None) -> str:
        if self.i >= len(self.toks):
            raise ParseError(f"意外结束, 期望 {expect!r}")
        t = self.toks[self.i]
        if expect is not None and t != expect:
            raise ParseError(f"期望 {expect!r}, 实得 {t!r}")
        self.i += 1
        return t

    def parse_expr(self) -> Expr:
        t = self.take()
        if t.isdigit():
            raise ParseError(f"数字 {t} 只能作为窗口参数出现在 ts_* 末位")
        if self.peek() != "(":
            return Expr.leaf(t)
        if t not in OP_REGISTRY:
            raise ParseError(f"未知算子: {t}")
        spec = OP_REGISTRY[t]
        self.take("(")
        children: list[Expr] = []
        window: int | None = None
        while True:
            if self.peek() == ")":
                break
            nxt = self.peek()
            if nxt is not None and nxt.isdigit():
                window = int(self.take())
            else:
                children.append(self.parse_expr())
            if self.peek() == ",":
                self.take(",")
                continue
        self.take(")")
        if spec.window and window is None:
            raise ParseError(f"{t} 缺少窗口参数")
        return Expr.call(t, *children, window=window)


def parse(s: str) -> Expr:
    p = _Parser(_tokenize(s))
    e = p.parse_expr()
    if p.peek() is not None:
        raise ParseError(f"表达式末尾有多余内容: {p.toks[p.i:]}")
    return e
