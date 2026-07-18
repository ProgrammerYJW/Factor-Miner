"""特征矩阵构建: data_cache/raw/ -> data_cache/features/ 宽表 parquet.

每个特征为 (T交易日 x N股票) float32 宽表, index=TradingDay, columns=str(InnerCode)。
v1 特征端子(方案§4): open/high/low/close(后复权) vwap volume amount turnover
free_turnover total_mv neg_mv ep_ttm bp sp_ttm
掩码: universe(非ST/非停牌/上市>=120交易日) suspend limit_up_oneline limit_down_oneline
复权: 精确复权线性变换 price_adj = price*A + B (A,B 按 ExDiviDate as-of 前向填充),
      并与 QT_PerformanceData.BackwardPrice 交叉核对(QA)。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from factor_miner.config import Config, get_config

log = logging.getLogger(__name__)

# LC_SpecialTrade.SpecialTradeType -> 风险警示状态机 (CT_SystemConst LB=1185)
ST_ON = {1, 3, 5, 7, 8, 9, 10}   # 实施ST/PT/*ST/ST转*ST/退市整理/高风险警示
ST_OFF = {2, 4, 6, 12}           # 各类撤销


class FeatureBuilder:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or get_config()
        self.raw = self.cfg.raw_dir
        self.out = self.cfg.features_dir
        self.out.mkdir(parents=True, exist_ok=True)
        cal = pd.read_parquet(self.raw / "trading_days.parquet")
        start = pd.Timestamp(self.cfg["data"]["start_date"])
        self.calendar = pd.DatetimeIndex(
            pd.to_datetime(cal["TradingDate"]).sort_values().unique()
        )
        self.calendar = self.calendar[self.calendar >= start]
        self.secu = pd.read_parquet(self.raw / "secu_main.parquet")
        self.secu["ListedDate"] = pd.to_datetime(self.secu["ListedDate"])
        log.info("日历 %d 天(%s~%s), 证券 %d 只",
                 len(self.calendar), self.calendar[0].date(), self.calendar[-1].date(),
                 len(self.secu))

    # ---------- io ----------
    def _load_daily(self, dataset: str) -> pd.DataFrame:
        parts = sorted((self.raw / dataset).glob("*.parquet"))
        if not parts:
            raise FileNotFoundError(f"数据集未同步: {dataset}")
        df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        df["TradingDay"] = pd.to_datetime(df["TradingDay"])
        return df

    def _pivot(self, df: pd.DataFrame, value: str) -> pd.DataFrame:
        w = df.pivot_table(index="TradingDay", columns="InnerCode", values=value,
                           aggfunc="last")
        w = w.reindex(self.calendar)
        w.columns = w.columns.astype(str)
        return w.astype(np.float32)

    def _save(self, name: str, w: pd.DataFrame) -> None:
        w.to_parquet(self.out / f"{name}.parquet", compression="zstd")
        log.info("特征 %-18s shape=%s 覆盖率=%.1f%%", name, w.shape,
                 100 * float(w.notna().mean().mean()))

    # ---------- 复权 ----------
    def _adj_ab(self, columns: pd.Index) -> tuple[pd.DataFrame, pd.DataFrame]:
        """按 ExDiviDate as-of 展开精确复权系数 A/B 到全日历。无记录处 A=1,B=0。"""
        f = pd.read_parquet(self.raw / "adj_factor.parquet")
        f["ExDiviDate"] = pd.to_datetime(f["ExDiviDate"])
        f = f.dropna(subset=["ExDiviDate"]).sort_values("ExDiviDate")
        f["InnerCode"] = f["InnerCode"].astype(str)
        a_ev = f.pivot_table(index="ExDiviDate", columns="InnerCode",
                             values="AdjustingFactor", aggfunc="last")
        b_ev = f.pivot_table(index="ExDiviDate", columns="InnerCode",
                             values="AdjustingConst", aggfunc="last")
        idx = self.calendar.union(a_ev.index)
        A = a_ev.reindex(idx).ffill().reindex(self.calendar).reindex(columns=columns)
        B = b_ev.reindex(idx).ffill().reindex(self.calendar).reindex(columns=columns)
        return A.fillna(1.0).astype(np.float32), B.fillna(0.0).astype(np.float32)

    # ---------- 主流程 ----------
    def build_all(self) -> None:
        q = self._load_daily("daily_quote")
        close = self._pivot(q, "ClosePrice")
        cols = close.columns
        open_, high, low = (self._pivot(q, c) for c in ("OpenPrice", "HighPrice", "LowPrice"))
        prev_close = self._pivot(q, "PrevClosePrice")
        volume = self._pivot(q, "TurnoverVolume")
        amount = self._pivot(q, "TurnoverValue")
        vwap = (amount / volume.replace(0, np.nan)).astype(np.float32)
        del q

        A, B = self._adj_ab(cols)
        A, B = A.reindex(columns=cols), B.reindex(columns=cols)
        feats = {
            "open": open_ * A + B, "high": high * A + B, "low": low * A + B,
            "close": close * A + B, "vwap": vwap * A + B,
            "volume": volume, "amount": amount,
        }

        # QA: 与官方后复权收盘价核对(QT_PerformanceData不含科创板, 须先对齐全网格)
        pdta = self._load_daily("perf_data")
        bwd = self._pivot(pdta, "BackwardPrice").reindex(columns=cols)
        both = feats["close"].notna() & bwd.notna()
        if both.to_numpy().any():
            rel = ((feats["close"] - bwd).abs() / bwd.abs().clip(lower=1e-9))[both]
            bad = float((rel > 0.001).sum().sum()) / float(both.sum().sum())
            log.info("复权QA: 与BackwardPrice相对误差>0.1%%的占比=%.4f%%", 100 * bad)
            if bad > 0.01:
                log.warning("复权QA偏差偏高, 请检查 adj_factor 数据!")
            # 官方值优先: 有 BackwardPrice 处用官方值(科创板等缺失处用自算值)
            feats["close"] = bwd.where(bwd.notna(), feats["close"]).astype(np.float32)

        # 涨跌停一字板(不可成交): 主板取官方标志, 缺失处(科创板)按 high==low 且方向自算
        srg = self._pivot(pdta, "StockBoard").reindex(columns=cols)     # 一字涨停
        dcl = self._pivot(pdta, "LimitBoard").reindex(columns=cols)     # 一字跌停
        oneline = (high == low) & high.notna()
        up_fallback = (oneline & (close > prev_close)).astype(np.float32)
        dn_fallback = (oneline & (close < prev_close)).astype(np.float32)
        self._save("limit_up_oneline", srg.where(srg.notna(), up_fallback))
        self._save("limit_down_oneline", dcl.where(dcl.notna(), dn_fallback))
        del pdta, bwd, srg, dcl

        perf = self._load_daily("stock_perf")
        suspend = self._pivot(perf, "Ifsuspend")
        feats["turnover"] = self._pivot(perf, "TurnoverRate")
        feats["free_turnover"] = self._pivot(perf, "TurnoverRateFreeFloat")
        feats["total_mv"] = self._pivot(perf, "TotalMV")
        feats["neg_mv"] = self._pivot(perf, "NegotiableMV")
        del perf

        val = self._load_daily("valuation")
        for name, col in (("ep_ttm", "PE"), ("bp", "PB"), ("sp_ttm", "PS")):
            x = self._pivot(val, col)
            feats[name] = (1.0 / x.replace(0, np.nan)).astype(np.float32)
        del val

        for name, w in feats.items():
            self._save(name, w.astype(np.float32))
        self._save("suspend", suspend.fillna(0.0))

        universe = self._build_universe(cols, close, suspend)
        self._save("universe", universe.astype(np.float32))
        self._save("industry", self._build_industry(cols))
        meta = {
            "n_days": len(self.calendar), "n_stocks": len(cols),
            "start": str(self.calendar[0].date()), "end": str(self.calendar[-1].date()),
            "features": sorted([*feats, "suspend", "universe",
                                "limit_up_oneline", "limit_down_oneline"]),
        }
        (self.out / "_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), "utf-8")
        log.info("特征构建完成: %s", meta)

    # ---------- 股票池掩码 ----------
    def _build_industry(self, cols: pd.Index) -> pd.DataFrame:
        """申万一级行业代码矩阵(as-of InfoPublDate 前向填充), 供行业中性化。"""
        ind = pd.read_parquet(self.raw / "industry.parquet")
        ind["InfoPublDate"] = pd.to_datetime(ind["InfoPublDate"])
        m = self.secu[["InnerCode", "CompanyCode"]].drop_duplicates()
        ind = ind.merge(m, on="CompanyCode", how="inner")
        ind["InnerCode"] = ind["InnerCode"].astype(str)
        ind["code"] = pd.to_numeric(ind["FirstIndustryCode"], errors="coerce")
        ind = ind.dropna(subset=["InfoPublDate", "code"]).sort_values("InfoPublDate")
        ev = ind.pivot_table(index="InfoPublDate", columns="InnerCode",
                             values="code", aggfunc="last")
        idx = self.calendar.union(ev.index)
        w = ev.reindex(idx).ffill().reindex(self.calendar).reindex(columns=cols)
        return w.astype(np.float32)

    def _st_mask(self, cols: pd.Index) -> pd.DataFrame:
        """风险警示状态机 -> (T,N) bool: True=当日处于ST/PT/退市整理/高风险警示。"""
        ev = pd.read_parquet(self.raw / "special_trade.parquet")
        ev["SpecialTradeTime"] = pd.to_datetime(ev["SpecialTradeTime"])
        ev["InnerCode"] = ev["InnerCode"].astype(str)
        ev = ev.sort_values("SpecialTradeTime")
        ev["state"] = np.where(ev["SpecialTradeType"].isin(list(ST_ON)), 1.0,
                       np.where(ev["SpecialTradeType"].isin(list(ST_OFF)), 0.0, np.nan))
        ev = ev.dropna(subset=["state"])
        w = ev.pivot_table(index="SpecialTradeTime", columns="InnerCode",
                           values="state", aggfunc="last")
        idx = self.calendar.union(w.index)
        st = w.reindex(idx).ffill().reindex(self.calendar).reindex(columns=cols).fillna(0.0)
        return st.astype(bool)

    def _build_universe(self, cols: pd.Index, close: pd.DataFrame,
                        suspend: pd.DataFrame) -> pd.DataFrame:
        has_quote = close.notna()
        not_suspend = suspend.reindex(columns=cols).fillna(0.0) < 0.5
        not_st = ~self._st_mask(cols)
        # 上市满 N 个交易日: 以"有行情的累计天数"计, 规避 ListedDate 缺失
        min_days = int(self.cfg["data"]["min_listed_days"])
        seasoned = has_quote.cumsum() > min_days
        uni = has_quote & not_suspend & not_st & seasoned
        log.info("universe: 平均截面股票数=%.0f", float(uni.sum(axis=1).mean()))
        return uni


def build_features(cfg: Config | None = None) -> None:
    FeatureBuilder(cfg).build_all()
