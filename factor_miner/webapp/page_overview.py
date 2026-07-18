"""页面①: 因子库总览 — 默认按10日IC/IR(训练段)降序, 表达式以课本格式渲染."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from factor_miner.expression.parser import parse as _parse
from factor_miner.expression.pretty import to_textbook
from factor_miner.webapp.common import fmt_summary, lib

st.title("📚 因子库总览")
st.caption("**IC/IR 口径(教材标准)**: IC = 日度RankIC均值; IR(信息比率) = IC均值 ÷ IC标准差 "
           "(Grinold & Kahn《主动组合管理》与券商金工研报通用定义, 亦称ICIR)。两列均置顶显示。")

df = lib().list()
if not len(df):
    st.info("因子库为空。运行 scripts/run_gp.py 或 scripts/run_rl.py 开始挖掘, "
            "或在下方手工添加因子表达式。")
else:
    c1, c2, c3, c4 = st.columns(4)
    engines = c1.multiselect("引擎", sorted(df["engine"].unique().tolist()))
    statuses = c2.multiselect("状态", sorted(df["status"].unique().tolist()),
                              default=["active"] if "active" in set(df["status"]) else [])
    min_icir = c3.number_input("最小|IC/IR|(10日,训练)", value=0.0, step=0.05)
    kw = c4.text_input("表达式/名称搜索")
    view = df.copy()
    if engines:
        view = view[view["engine"].isin(engines)]
    if statuses:
        view = view[view["status"].isin(statuses)]
    if min_icir > 0:
        view = view[view["icir10_train"].abs() >= min_icir]
    if kw:
        m = view["name"].str.contains(kw, case=False, na=False) | \
            view["expression"].str.contains(kw, case=False, na=False)
        view = view[m]
    st.caption(f"共 {len(view)} 个因子 (默认按 ⭐IC/IR(10日,训练) 绝对值降序)")
    # 渲染表格: 每行用 st.latex 替换原始表达式列
    s = fmt_summary(view)
    table_cols = [c for c in s.columns if c != "📐表达式(课本格式)"]
    header = st.columns([1.5 if c == "表达式" or c.startswith("📐") else
                         0.6 if c in ("ID","状态") else 1.0
                         for c in s.columns])
    for h, cn in zip(header, s.columns):
        h.markdown(f"**{cn}**" if len(cn) < 20 else f"**{cn}**", help="")
    for _, row in s.iterrows():
        cols = st.columns(len(s.columns))
        for i, (c, cn) in enumerate(zip(cols, s.columns)):
            v = row[cn]
            if cn == "📐表达式(课本格式)":
                c.latex(r"\displaystyle " + v)
            else:
                c.write(v)
    st.caption("→ 到『因子详情』页查看单因子完整指标并执行删改操作")

with st.expander("➕ 手工添加因子(走同一评估+准入管道)"):
    ex = st.text_input("表达式", placeholder="例: cs_rank(ts_corr(cs_rank(close), cs_rank(volume), 10))")
    force = st.checkbox("跳过准入门槛强制入库(status=candidate)", value=False)
    if st.button("评估并入库", type="primary") and ex.strip():
        from factor_miner.expression.parser import parse
        from factor_miner.library import Admission
        from factor_miner.webapp.common import evaluator

        try:
            expr = parse(ex.strip())
        except Exception as e:  # noqa: BLE001
            st.error(f"表达式解析失败: {e}")
            st.stop()
        with st.spinner("全量评估中(约数十秒)..."):
            if force:
                rep = evaluator().evaluate_full(expr)
                fid = lib().add(expr.to_string(), expr.key(), "manual",
                                rep["metrics"], rep["factor"], rep["ic_series"],
                                status="candidate")
                st.success(f"已强制入库 #{fid} (candidate)")
            else:
                ok, reason, fid = Admission(evaluator(), lib()).submit(expr, "manual")
                st.success(f"入库成功 #{fid}") if ok else st.warning(f"未通过准入: {reason}")
