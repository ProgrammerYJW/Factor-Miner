"""FactorMiner Web界面入口: streamlit run factor_miner/webapp/app.py"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

st.set_page_config(page_title="FactorMiner 因子库", page_icon="📈", layout="wide")

base = Path(__file__).parent
pages = [
    st.Page(str(base / "page_overview.py"), title="因子库总览", icon="📚", default=True),
    st.Page(str(base / "page_detail.py"), title="因子详情", icon="🔍"),
    st.Page(str(base / "page_rules.py"), title="筛选标准设置", icon="🎛️"),
    st.Page(str(base / "page_mining.py"), title="挖掘监控", icon="⛏️"),
    st.Page(str(base / "page_corr.py"), title="相关性矩阵", icon="🔗"),
]
st.navigation(pages).run()
