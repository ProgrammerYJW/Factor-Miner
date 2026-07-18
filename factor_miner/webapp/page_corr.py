"""页面④: 库内活跃因子相关性热力图."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from factor_miner.evaluation import metrics as M
from factor_miner.webapp.common import lib

st.title("🔗 因子相关性矩阵")

df = lib().list(status="active")
if len(df) < 2:
    st.info("活跃因子不足2个")
    st.stop()

top_n = st.slider("纳入因子数(按IC/IR排序)", 2, min(30, len(df)), min(15, len(df)))
ids = df["id"].head(top_n).tolist()
names = df.set_index("id")["name"].to_dict()


@st.cache_data(show_spinner="计算相关性矩阵...")
def corr_matrix(ids: tuple[int, ...]) -> pd.DataFrame:
    mats = {i: lib().load_values(i) for i in ids}
    n = len(ids)
    out = np.eye(n)
    for a in range(n):
        for b in range(a + 1, n):
            c = M.value_corr(mats[ids[a]], mats[ids[b]], step=10)
            out[a, b] = out[b, a] = c if np.isfinite(c) else np.nan
    lbl = [names[i] for i in ids]
    return pd.DataFrame(out, index=lbl, columns=lbl)


cm = corr_matrix(tuple(ids))
fig = px.imshow(cm, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
fig.update_layout(height=650, margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, width="stretch")
st.caption("值为因子暴露截面Pearson相关(每10日采样平均)。|corr|>0.7 建议只保留ICIR更高者。")
