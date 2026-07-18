"""页面③: 挖掘监控 — GP各代/RL各update进度曲线与近期入库流水."""
from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from factor_miner.webapp.common import cfg, lib

st.title("⛏️ 挖掘监控")
if st.button("🔄 刷新"):
    st.rerun()

log_dir = cfg().artifacts_dir / "logs"


def _read_jsonl(name: str) -> pd.DataFrame:
    p = log_dir / name
    if not p.exists():
        return pd.DataFrame()
    rows = [json.loads(x) for x in p.read_text("utf-8").splitlines() if x.strip()]
    return pd.DataFrame(rows)


c1, c2 = st.columns(2)
with c1:
    st.subheader("GP 引擎")
    gp = _read_jsonl("gp_progress.jsonl")
    if len(gp):
        fig = go.Figure()
        fig.add_scatter(x=gp["gen"], y=gp["best_fitness"], name="最优适应度")
        fig.add_scatter(x=gp["gen"], y=gp["median_fitness"], name="中位适应度")
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, width="stretch")
        last = gp.iloc[-1]
        st.caption(f"Gen {last['gen']} | best_ic={last.get('best_ic')} "
                   f"best_icir={last.get('best_icir')} | {last['best_expr'][:80]}")
    else:
        st.info("尚无GP日志 (scripts/run_gp.py)")
with c2:
    st.subheader("RL 引擎 (PPO)")
    rl = _read_jsonl("rl_progress.jsonl")
    if len(rl):
        fig = go.Figure()
        fig.add_scatter(x=rl["update"], y=rl["reward_mean"], name="平均奖励")
        fig.add_scatter(x=rl["update"], y=rl["reward_best"], name="最优奖励")
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, width="stretch")
        last = rl.iloc[-1]
        st.caption(f"Update {last['update']} | uniq={last.get('n_unique')} | "
                   f"entropy={last.get('entropy'):.3f} | {last.get('best_expr', '')[:80]}")
    else:
        st.info("尚无RL日志 (scripts/run_rl.py)")

st.subheader("🆕 近期入库因子")
df = lib().list()
if len(df):
    recent = df.sort_values("created_at", ascending=False).head(10)
    from factor_miner.webapp.common import fmt_summary

    st.dataframe(fmt_summary(recent), width="stretch")
else:
    st.info("因子库为空")
