"""可配置准入规则引擎测试."""
import json

import pytest

from factor_miner.library.rules import Rule, RuleSet

METRICS = {
    "h10_train": {"ic_mean": 0.05, "ic_std": 0.09, "icir": 0.55, "ls_mdd": 0.12,
                  "ls_sharpe": 2.0, "top_turnover": 0.4},
    "h10_valid": {"ic_mean": 0.04, "icir": 0.40},
    "coverage": 0.85,
}


def test_rule_ops():
    assert Rule("icir", 10, "train", "abs>=", 0.5).check(METRICS)[0]
    assert not Rule("icir", 10, "train", "abs>=", 0.6).check(METRICS)[0]
    assert Rule("ls_mdd", 10, "train", "<=", 0.2).check(METRICS)[0]
    assert not Rule("ls_mdd", 10, "train", "<=", 0.1).check(METRICS)[0]
    assert Rule("ls_sharpe", 10, "train", ">=", 1.5).check(METRICS)[0]
    ok, why = Rule("icir", 20, "train", "abs>=", 0.1).check(METRICS)   # 周期缺失
    assert not ok and "缺失" in why


def test_negative_metric_abs():
    m = {"h10_train": {"ic_mean": -0.05, "icir": -0.55}, "coverage": 0.9}
    assert Rule("icir", 10, "train", "abs>=", 0.5).check(m)[0]         # 绝对值口径
    assert not Rule("icir", 10, "train", ">=", 0.5).check(m)[0]        # 有符号口径


def test_ruleset_check_and_reasons():
    rs = RuleSet(rules=[Rule("icir", 10, "train", "abs>=", 0.3),
                        Rule("icir", 10, "valid", "abs>=", 0.15)],
                 min_coverage=0.6, require_same_sign=True, primary_horizon=10)
    ok, fails = rs.check_metrics(METRICS)
    assert ok and fails == []
    rs.rules.append(Rule("top_turnover", 10, "train", "<=", 0.3))      # 换手超限
    ok, fails = rs.check_metrics(METRICS)
    assert not ok and any("换手" in f for f in fails)


def test_same_sign_and_coverage():
    rs = RuleSet(rules=[], min_coverage=0.9, primary_horizon=10)
    ok, fails = rs.check_metrics(METRICS)                              # 覆盖率0.85<0.9
    assert not ok and any("覆盖率" in f for f in fails)
    bad = {**METRICS, "coverage": 0.95,
           "h10_valid": {"ic_mean": -0.04, "icir": -0.4}}
    rs2 = RuleSet(rules=[], min_coverage=0.6, require_same_sign=True, primary_horizon=10)
    ok, fails = rs2.check_metrics(bad)
    assert not ok and any("同号" in f for f in fails)
    rs2.require_same_sign = False
    assert rs2.check_metrics(bad)[0]


def test_disabled_rule_skipped():
    rs = RuleSet(rules=[Rule("icir", 10, "train", "abs>=", 9.9, enabled=False)],
                 min_coverage=0.5, require_same_sign=False)
    assert rs.check_metrics(METRICS)[0]


def test_global_corr_rule():
    m = {**METRICS, "max_corr_with_library": 0.55}
    assert Rule("max_corr_with_library", 0, "global", "<=", 0.7).check(m)[0]
    ok, why = Rule("max_corr_with_library", 0, "global", "<=", 0.5).check(m)
    assert not ok and "相关性" in why
    # 相关性属全局规则: check_metrics 跳过, check_global 负责
    rs = RuleSet(rules=[Rule("max_corr_with_library", 0, "global", "<=", 0.5)],
                 min_coverage=0.5, require_same_sign=False)
    assert rs.check_metrics(m)[0]
    assert not rs.check_global(m)[0]
    assert not rs.check_all(m)[0]


def test_legacy_migration(tmp_path):
    class FakeCfg:
        artifacts_dir = tmp_path
    RuleSet.path(FakeCfg).write_text(
        '{"rules": [], "max_corr_with_library": 0.65, "min_coverage": 0.6, '
        '"require_same_sign": true, "primary_horizon": 10}', "utf-8")
    rs = RuleSet.load(FakeCfg)
    corr_rules = [r for r in rs.rules if r.metric == "max_corr_with_library"]
    assert len(corr_rules) == 1 and corr_rules[0].threshold == 0.65
    assert corr_rules[0].op == "<="


def test_save_load_roundtrip(tmp_path):
    class FakeCfg:
        artifacts_dir = tmp_path
    rs = RuleSet(rules=[Rule("ic_mean", 10, "train", "abs>=", 0.02, note="测试"),
                        Rule("max_corr_with_library", 0, "global", "<=", 0.55)],
                 min_coverage=0.7, require_same_sign=False, primary_horizon=10)
    rs.save(FakeCfg)
    rs2 = RuleSet.load(FakeCfg)
    assert not rs2.require_same_sign and rs2.min_coverage == 0.7
    assert rs2.rules[0].note == "测试" and rs2.rules[0].threshold == 0.02
    assert rs2.rules[1].metric == "max_corr_with_library"
    assert rs2.threshold_of("ic_mean") == 0.02
    assert rs2.threshold_of("icir") is None
    # 文件为可读JSON且不再写旧版顶层corr字段
    d = json.loads(RuleSet.path(FakeCfg).read_text("utf-8"))
    assert "max_corr_with_library" not in d and len(d["rules"]) == 2
