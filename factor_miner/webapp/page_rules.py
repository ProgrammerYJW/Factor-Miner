"""页面⑤: 筛选标准设置 — 在Web里编辑因子准入规则, 保存即对挖掘进程即时生效."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from factor_miner.library.rules import (
    GLOBAL_METRICS, METRIC_CHOICES, OP_CHOICES, SEG_CHOICES, Rule, RuleSet,
)
from factor_miner.webapp.common import cfg, lib

st.title("🎛️ 因子筛选标准")
st.caption("规则之间为**与**关系(全部满足才入库)。保存后**即时生效**——正在跑的GP/RL挖掘进程"
           "在下一次提交因子时就按新标准执行, 无需重启。日后咨询专家后可随时在本页调整。")

rs = RuleSet.load()
horizons = [int(h) for h in cfg()["label"]["horizons"]]
H_OPTS = ["全局"] + [f"{h}日" for h in horizons]
S_OPTS = ["全局"] + list(SEG_CHOICES.values())

# ---------- 专项标准 ----------
st.subheader("专项标准")
c1, c2 = st.columns(2)
min_cov = c1.number_input(
    "覆盖率下限", min_value=0.0, max_value=1.0, value=float(rs.min_coverage), step=0.05,
    help="因子有效值占股票池的比例下限")
same_sign = c2.toggle("要求验证段IC与训练段同号", value=rs.require_same_sign,
                      help="防止过拟合: 样本外方向反转的因子拒绝入库")

# ---------- 指标规则表 ----------
st.subheader("指标规则(可增删改行)")


def _h_label(r: Rule) -> str:
    return "全局" if r.metric in GLOBAL_METRICS else f"{r.horizon}日"


def _s_label(r: Rule) -> str:
    return "全局" if r.metric in GLOBAL_METRICS else SEG_CHOICES.get(r.segment, r.segment)


rows = [{"启用": r.enabled,
         "指标": METRIC_CHOICES.get(r.metric, r.metric),
         "周期": _h_label(r),
         "时段": _s_label(r),
         "比较": OP_CHOICES.get(r.op, r.op),
         "阈值": r.threshold,
         "备注": r.note} for r in rs.rules]
edited = st.data_editor(
    pd.DataFrame(rows, columns=["启用", "指标", "周期", "时段", "比较", "阈值", "备注"]),
    num_rows="dynamic", width="stretch", hide_index=True,
    column_config={
        "启用": st.column_config.CheckboxColumn(default=True),
        "指标": st.column_config.SelectboxColumn(
            options=list(METRIC_CHOICES.values()), required=True),
        "周期": st.column_config.SelectboxColumn(options=H_OPTS, required=True),
        "时段": st.column_config.SelectboxColumn(options=S_OPTS, required=True),
        "比较": st.column_config.SelectboxColumn(
            options=list(OP_CHOICES.values()), required=True),
        "阈值": st.column_config.NumberColumn(required=True, format="%.4f"),
    },
)
st.caption("说明: **与已有因子相关性**为全局指标(周期/时段选『全局』), 值为候选因子与库内"
           "活跃因子相关系数绝对值的最大者, 通常设『≤ 某上限』; "
           "多空最大回撤/Top组换手率等『越小越好』的指标请选 ≤ 或 |值|≤; "
           "分层类指标(多空收益/夏普/回撤/换手)仅在10日周期计算。")

_rev_metric = {v: k for k, v in METRIC_CHOICES.items()}
_rev_op = {v: k for k, v in OP_CHOICES.items()}
_rev_seg = {v: k for k, v in SEG_CHOICES.items()}

b1, b2, _ = st.columns([1, 1, 3])
if b1.button("💾 保存标准", type="primary"):
    try:
        new_rules = []
        for _, r in edited.iterrows():
            if pd.isna(r["指标"]) or pd.isna(r["阈值"]):
                continue
            metric = _rev_metric[str(r["指标"])]
            if metric in GLOBAL_METRICS:
                h, seg = 0, "global"
            else:
                if str(r["周期"]) == "全局" or str(r["时段"]) == "全局":
                    raise ValueError(f"{r['指标']} 不是全局指标, 周期/时段不能选『全局』")
                h = int(str(r["周期"]).replace("日", ""))
                seg = _rev_seg[str(r["时段"])]
            new_rules.append(Rule(
                metric=metric, horizon=h, segment=seg,
                op=_rev_op[str(r["比较"])], threshold=float(r["阈值"]),
                enabled=bool(r["启用"]),
                note="" if pd.isna(r["备注"]) else str(r["备注"]),
            ))
        new_rs = RuleSet(rules=new_rules, min_coverage=float(min_cov),
                         require_same_sign=bool(same_sign),
                         primary_horizon=rs.primary_horizon)
        new_rs.save()
        st.success(f"已保存 {len(new_rules)} 条指标规则 + 专项标准, 对后续提交即时生效")
    except Exception as e:  # noqa: BLE001
        st.error(f"保存失败, 请检查表格填写: {e}")
if b2.button("↩️ 恢复默认标准"):
    RuleSet.default().save()
    st.success("已恢复默认标准")
    st.rerun()

# ---------- 试算: 现行标准对库内因子的判定 ----------
st.divider()
st.subheader("🧪 试算: 当前标准对库内因子的判定")
df = lib().list()
if not len(df):
    st.info("因子库为空")
else:
    cur = RuleSet.load()
    out = []
    for _, row in df.iterrows():
        fac = lib().get(int(row["id"]))
        ok, fails = cur.check_all(fac["metrics"])
        out.append({"ID": row["id"], "名称": row["name"], "状态": row["status"],
                    "判定": "✅ 通过" if ok else "❌ 不通过",
                    "未通过原因": "" if ok else "; ".join(fails[:3])})
    st.dataframe(pd.DataFrame(out), width="stretch", hide_index=True)
    st.caption("注: 试算仅供参考, 不会自动移除已入库因子; 如需清理请到详情页删除/归档。"
               "库内因子的『与已有因子相关性』为其入库时点的值。")
