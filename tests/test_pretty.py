"""课本格式公式渲染器测试."""
from factor_miner.expression.parser import parse
from factor_miner.expression.pretty import to_textbook


def test_div_renders_as_fraction():
    s = to_textbook(parse("div(close, total_mv)"))
    assert r"\frac{\text{Close}}{\text{MV}}" == s


def test_ts_window_as_subscript():
    s = to_textbook(parse("ts_mean(close, 20)"))
    assert s == r"\operatorname{Mean}_{20}\left(\text{Close}\right)"


def test_nested_and_neg():
    s = to_textbook(parse("neg(decay_linear(free_turnover, 10))"))
    assert s == r"-\operatorname{WMA}_{10}\left(\text{Turn}^{\text{free}}\right)"


def test_delay_and_corr():
    s = to_textbook(parse("ts_corr(cs_rank(close), cs_rank(volume), 10)"))
    assert r"\operatorname{Corr}_{10}" in s and r"\operatorname{Rank}" in s


def test_precedence_parens():
    s = to_textbook(parse("mul(add(close, open), vwap)"))
    assert s == r"\left(\text{Close} + \text{Open}\right) \cdot \text{VWAP}"


def test_no_crash_on_all_ops():
    from factor_miner.expression.ops import OP_REGISTRY

    for name, spec in OP_REGISTRY.items():
        args = ["close", "volume"][: spec.n_args]
        w = ", 5" if spec.window else ""
        expr = parse(f"{name}({', '.join(args)}{w})")
        out = to_textbook(expr)
        assert isinstance(out, str) and len(out) > 0
