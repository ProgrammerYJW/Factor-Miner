"""经典因子 sanity check: 验证评估管道方向与量级 (方案§14).

20日反转/低换手/EP价值 三个教科书因子应给出符合共识的IC方向。
用法: python sanity_check.py (需先完成数据同步与特征构建)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CLASSICS = {
    "反转20日(预期IC>0)": "neg(ts_ret(close, 20))",
    "低换手20日(预期IC>0)": "neg(ts_mean(turnover, 20))",
    "EP价值(预期IC>0)": "ep_ttm",
    "20日动量(预期IC<0,A股反转市)": "ts_ret(close, 20)",
}


def main() -> None:
    from factor_miner.evaluation import Evaluator
    from factor_miner.expression.parser import parse

    ev = Evaluator()
    print("=" * 100)
    print(f"{'因子':<28}{'段':<8}{'RankIC':>9}{'IR':>8}{'年化IR':>9}"
          f"{'t值':>8}{'胜率':>7}{'多空年化':>9}{'夏普':>7}")
    print("-" * 100)
    for name, s in CLASSICS.items():
        rep = ev.evaluate_full(parse(s))
        for seg in ("train", "valid"):
            m = rep["metrics"][f"h{ev.primary_h}_{seg}"]
            print(f"{name:<28}{seg:<8}{m['ic_mean']:>9.4f}{m['icir']:>8.3f}"
                  f"{m['icir_ann']:>9.3f}{m['t_stat']:>8.2f}{m['win_rate']:>7.3f}"
                  f"{m.get('ls_ann_ret', float('nan')):>9.4f}"
                  f"{m.get('ls_sharpe', float('nan')):>7.2f}")
    print("=" * 100)
    print("判定: 反转/低换手/EP 训练段 RankIC 应为正且|ICIR|可观; 动量应为负 => 管道方向正确")


if __name__ == "__main__":
    main()
