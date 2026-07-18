"""表达式 -> 课本数学格式渲染器.

输出为 KaTeX 标记, 由 Streamlit 渲染成教科书式排版(分数线/下标/正体函数名),
用户在界面上只看到排版后的数学公式, 不会看到任何标记源码。
"""
from __future__ import annotations

from factor_miner.expression.nodes import Expr

# 特征端子 -> 课本变量名
_FEAT = {
    "open": r"\text{Open}", "high": r"\text{High}", "low": r"\text{Low}",
    "close": r"\text{Close}", "vwap": r"\text{VWAP}", "volume": r"\text{Vol}",
    "amount": r"\text{Amt}", "turnover": r"\text{Turn}",
    "free_turnover": r"\text{Turn}^{\text{free}}", "total_mv": r"\text{MV}",
    "neg_mv": r"\text{MV}^{\text{float}}", "ep_ttm": r"\text{EP}",
    "bp": r"\text{BP}", "sp_ttm": r"\text{SP}",
}

# 时序算子 -> 函数名(窗口作下标)
_TS_FN = {
    "ts_mean": "Mean", "ts_std": "Std", "ts_sum": "Sum", "ts_min": "Min",
    "ts_max": "Max", "ts_med": "Med", "ts_rank": "TSRank", "ts_skew": "Skew",
    "ts_ret": "Ret", "ts_corr": "Corr", "ts_cov": "Cov", "ts_ema": "EMA",
    "ts_slope": "Slope", "decay_linear": "WMA",
}
_CS_FN = {
    "cs_rank": "Rank", "cs_zscore": "ZScore", "cs_scale": "Scale",
    "cs_indneutral": "IndNeutral",
}

# 优先级: 1=加减, 1.5=取负, 2=乘, 3=原子(函数/分数/变量)
_PREC_ATOM, _PREC_MUL, _PREC_NEG, _PREC_ADD = 3, 2, 1.5, 1


def _wrap(s: str, prec: float, need: float) -> str:
    return rf"\left({s}\right)" if prec < need else s


def _render(e: Expr) -> tuple[str, float]:
    """返回 (katex标记, 该子式的优先级)。"""
    if e.is_leaf:
        return _FEAT.get(e.feature, rf"\text{{{e.feature}}}"), _PREC_ATOM
    op, ch, w = e.op, e.children, e.window

    if op == "add":
        a, pa = _render(ch[0])
        b, pb = _render(ch[1])
        return f"{a} + {b}", _PREC_ADD
    if op == "sub":
        a, pa = _render(ch[0])
        b, pb = _render(ch[1])
        return f"{a} - {_wrap(b, pb, _PREC_MUL)}", _PREC_ADD
    if op == "mul":
        a, pa = _render(ch[0])
        b, pb = _render(ch[1])
        return rf"{_wrap(a, pa, _PREC_MUL)} \cdot {_wrap(b, pb, _PREC_MUL)}", _PREC_MUL
    if op == "div":
        a, _ = _render(ch[0])
        b, _ = _render(ch[1])
        return rf"\frac{{{a}}}{{{b}}}", _PREC_ATOM
    if op == "neg":
        a, pa = _render(ch[0])
        return f"-{_wrap(a, pa, _PREC_MUL)}", _PREC_NEG
    if op == "abs":
        a, _ = _render(ch[0])
        return rf"\left|{a}\right|", _PREC_ATOM
    if op == "sign":
        a, _ = _render(ch[0])
        return rf"\operatorname{{sgn}}\left({a}\right)", _PREC_ATOM
    if op == "sqrt":
        a, _ = _render(ch[0])
        return rf"\operatorname{{sgn}}\left({a}\right)\sqrt{{\left|{a}\right|}}", _PREC_MUL
    if op == "slog":
        a, _ = _render(ch[0])
        return rf"\operatorname{{sgn}}\left({a}\right)\ln\left(1+\left|{a}\right|\right)", _PREC_MUL
    if op == "inv":
        a, _ = _render(ch[0])
        return rf"\frac{{1}}{{{a}}}", _PREC_ATOM
    if op == "ts_delay":
        a, pa = _render(ch[0])
        return rf"{_wrap(a, pa, _PREC_ATOM)}_{{\,t-{w}}}", _PREC_ATOM
    if op == "ts_delta":
        a, _ = _render(ch[0])
        return rf"\Delta_{{{w}}}\left({a}\right)", _PREC_ATOM
    if op in _TS_FN:
        args = ",\ ".join(_render(c)[0] for c in ch)
        return rf"\operatorname{{{_TS_FN[op]}}}_{{{w}}}\left({args}\right)", _PREC_ATOM
    if op == "cs_demean":
        a, pa = _render(ch[0])
        return rf"{_wrap(a, pa, _PREC_MUL)} - \overline{{{a}}}", _PREC_ADD
    if op in _CS_FN:
        a, _ = _render(ch[0])
        return rf"\operatorname{{{_CS_FN[op]}}}\left({a}\right)", _PREC_ATOM
    # 未知算子兜底: 函数式
    args = ",\ ".join(_render(c)[0] for c in ch)
    sub = f"_{{{w}}}" if w is not None else ""
    return rf"\operatorname{{{op}}}{sub}\left({args}\right)", _PREC_ATOM


def to_textbook(e: Expr) -> str:
    """表达式 -> 课本格式数学标记(交给 st.latex 渲染)。"""
    return _render(e)[0]


VARIABLE_LEGEND = (
    "变量说明: Close/Open/High/Low=后复权价, VWAP=均价, Vol=成交量, Amt=成交额, "
    "Turn=换手率(free=自由流通口径), MV=总市值(float=流通), EP/BP/SP=盈利/净资产/营收市值比; "
    "函数下标为滚动窗口(交易日), Rank/ZScore为截面算子, WMA为线性衰减加权均值"
)
