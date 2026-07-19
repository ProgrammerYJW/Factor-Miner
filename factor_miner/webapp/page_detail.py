"""页面②: 因子详情 — 全周期x全时段指标(⭐ICIR突出) + 图表 + 删改操作."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from factor_miner.webapp.common import SEG_NAMES, cfg, lib, metrics_matrix

st.title("🔍 因子详情")

df = lib().list()
if not len(df):
    st.info("因子库为空")
    st.stop()

label = st.selectbox("选择因子", [f"#{r.id} {r['name']} ({r.engine}/{r.status})"
                                  for _, r in df.iterrows()])
fid = int(label.split()[0][1:])
fac = lib().get(fid)
m = fac["metrics"]
horizons = list(cfg()["label"]["horizons"])

st.subheader("📐 因子定义式")
from factor_miner.expression.parser import parse as _parse
from factor_miner.expression.pretty import VARIABLE_LEGEND, to_textbook

try:
    st.latex(to_textbook(_parse(fac["expression"])))
    st.caption(VARIABLE_LEGEND)
except Exception:  # noqa: BLE001 渲染失败时退回原始式
    st.warning("课本格式渲染失败, 显示原始表达式")
with st.expander("原始表达式(供复制/手工添加/重算)"):
    st.code(fac["expression"], language="text")
st.caption("IC/IR = IC均值 ÷ IC标准差")
h10t = m.get("h10_train", {})
c = st.columns(6)
c[0].metric("⭐IC均值 (10日,训练)", h10t.get("ic_mean"))
c[1].metric("⭐IC/IR (10日,训练)", h10t.get("icir"))
c[2].metric("⭐IC/IR (10日,验证)", m.get("h10_valid", {}).get("icir"))
c[3].metric("多空年化(训练)", h10t.get("ls_ann_ret"))
c[4].metric("多空夏普(训练)", h10t.get("ls_sharpe"))
c[5].metric("与库内最大相关", m.get("max_corr_with_library"))

tabs = st.tabs([SEG_NAMES[s] for s in ("train", "valid", "observe")])
for tab, seg in zip(tabs, ("train", "valid", "observe")):
    with tab:
        st.dataframe(metrics_matrix(m, horizons, seg), width="stretch")

# ---------- 图表 ----------
ic = lib().load_ic(fid)
ic.index = pd.to_datetime(ic.index)
h_sel = st.selectbox("IC曲线周期", [str(h) for h in horizons],
                     index=horizons.index(int(cfg()["label"]["primary_horizon"])))
s = ic[h_sel].dropna()
g1, g2 = st.columns(2)
with g1:
    fig = go.Figure(go.Scatter(x=s.index, y=s.cumsum(), mode="lines",
                               name=f"累计RankIC({h_sel}日)"))
    fig.update_layout(title=f"累计RankIC({h_sel}日)", height=340,
                      margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")
with g2:
    mon = s.resample("ME").mean()
    heat = pd.DataFrame({"year": mon.index.year, "month": mon.index.month, "ic": mon.values})
    piv = heat.pivot(index="year", columns="month", values="ic")
    fig2 = px.imshow(piv, color_continuous_scale="RdBu_r", origin="lower",
                     aspect="auto", title=f"月均RankIC热力图({h_sel}日)")
    fig2.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig2, width="stretch")

ga = h10t.get("group_ann") or []
if ga:
    figg = go.Figure(go.Bar(x=[f"G{i + 1}" for i in range(len(ga))], y=ga))
    figg.update_layout(title="10分组年化收益(训练段, G10=因子值最高组)", height=300,
                       margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(figg, width="stretch")

# ---------- 删改操作 ----------
st.divider()
st.subheader("✏️ 编辑 / 🗑️ 删除")
e1, e2, e3 = st.columns(3)
new_name = e1.text_input("重命名", value=fac["name"])
new_tags = e2.text_input("标签(逗号分隔)", value=fac["tags"])
new_status = e3.selectbox("状态", ["active", "candidate", "archived"],
                          index=["active", "candidate", "archived"].index(fac["status"]))
new_notes = st.text_area("备注", value=fac["notes"], height=80)
b1, b2, b3, _ = st.columns([1, 1, 1, 2])
if b1.button("保存修改", type="primary"):
    lib().update(fid, name=new_name, tags=new_tags, notes=new_notes, status=new_status)
    st.success("已保存")
    st.rerun()
if b2.button("重新评估(全量)"):
    from factor_miner.expression.parser import parse
    from factor_miner.webapp.common import evaluator

    with st.spinner("重新评估中..."):
        rep = evaluator().evaluate_full(parse(fac["expression"]))
        lib().update(fid, metrics=rep["metrics"])
        rep["factor"].astype("float32").to_parquet(
            lib().values_dir / f"{fid}.parquet", compression="zstd")
        pd.DataFrame(rep["ic_series"]).to_parquet(
            lib().ic_dir / f"{fid}.parquet", compression="zstd")
    st.success("已更新指标")
    st.rerun()
with b3.popover("删除因子"):
    hard = st.checkbox("硬删除(连同数据文件, 不可恢复)")
    st.warning(f"确认删除 #{fid} {fac['name']} ?")
    if st.button("确认删除", type="primary"):
        lib().delete(fid, hard=hard)
        st.success("已删除(硬)" if hard else "已归档(软删, 状态=archived)")
        st.rerun()
