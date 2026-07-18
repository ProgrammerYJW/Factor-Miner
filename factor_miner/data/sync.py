"""聚源数据增量同步器: JYDB -> data_cache/raw/ 分年 parquet (数据集间并行).

数据集(方案§4): 交易日历/证券主表/日行情(主板∪科创板)/行情表现/后复权与涨跌停/
估值/复权因子/ST状态/行业(申万新版Standard=38)/指数行情/指数成分。
增量键: TradingDay; 进度按数据集独立记录 data_cache/raw/_meta/{dataset}.txt
(并行安全, 兼容迁移旧版单文件 _meta.json)。
并行: 数据集间 ThreadPoolExecutor(默认4线程, 网络IO型); 数据集内按年顺序(增量语义)。
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import pandas as pd

from factor_miner.config import Config, get_config
from factor_miner.data.juyuan import JuyuanDB

log = logging.getLogger(__name__)

# 科创板并表: (主板表, 科创板表)。科创板表字段与主板同构(字典与实测确认), 缺列时置NaN。
PAIRED_SQL = {
    "daily_quote": (
        "dbo.QT_DailyQuote",
        "dbo.LC_STIBDailyQuote",
        ["TradingDay", "PrevClosePrice", "OpenPrice", "HighPrice", "LowPrice",
         "ClosePrice", "TurnoverVolume", "TurnoverValue"],
    ),
    "stock_perf": (
        "dbo.QT_StockPerformance",
        "dbo.LC_STIBPerformance",
        ["TradingDay", "Ifsuspend", "TurnoverRate", "TurnoverRateFreeFloat",
         "TotalMV", "NegotiableMV"],
    ),
    "valuation": (
        "dbo.LC_DIndicesForValuation",
        "dbo.LC_STIBDIndiForValue",
        ["TradingDay", "PE", "PB", "PS", "DividendRatio"],
    ),
}

# 主板独有(科创板缺失时后续自算): 后复权价与涨跌停标志
PERF_DATA_COLS = ["TradingDay", "BackwardPrice", "ChangePCT",
                  "SurgedLimit", "DeclineLimit", "StockBoard", "LimitBoard"]


class Syncer:
    def __init__(self, cfg: Config | None = None, db: JuyuanDB | None = None):
        self.cfg = cfg or get_config()
        self.db = db or JuyuanDB(self.cfg)
        self.raw_dir = self.cfg.raw_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir = self.raw_dir / "_meta"
        self.meta_dir.mkdir(exist_ok=True)
        legacy = self.raw_dir / "_meta.json"
        if legacy.exists():                            # 迁移旧版进度并移除
            for k, v in json.loads(legacy.read_text("utf-8")).items():
                p = self.meta_dir / f"{k}.txt"
                if not p.exists():
                    p.write_text(str(v), "utf-8")
            legacy.unlink()
        self.markets = tuple(self.cfg["data"]["markets"])
        self.n_workers = int(self.cfg["data"].get("sync_workers", 4))
        self.start = pd.Timestamp(self.cfg["data"]["start_date"])
        end_cfg = self.cfg["data"].get("end_date") or ""
        self.end = pd.Timestamp(end_cfg) if end_cfg else pd.Timestamp(self.db.data_end_date())
        log.info("同步区间: %s ~ %s, markets=%s, 并行度=%d",
                 self.start.date(), self.end.date(), self.markets, self.n_workers)

    # ---------- 基础设施 ----------
    def _last_day(self, dataset: str) -> pd.Timestamp | None:
        p = self.meta_dir / f"{dataset}.txt"
        return pd.Timestamp(p.read_text("utf-8").strip()) if p.exists() else None

    def _set_last_day(self, dataset: str, day: pd.Timestamp) -> None:
        (self.meta_dir / f"{dataset}.txt").write_text(str(day.date()), "utf-8")

    def _write_year(self, dataset: str, year: int, df: pd.DataFrame, keys: list[str]) -> None:
        d = self.raw_dir / dataset
        d.mkdir(exist_ok=True)
        p = d / f"{year}.parquet"
        if p.exists():
            old = pd.read_parquet(p)
            df = pd.concat([old, df], ignore_index=True)
        df = df.drop_duplicates(subset=keys, keep="last").sort_values(keys)
        df.to_parquet(p, index=False, compression="zstd")

    def _write_full(self, dataset: str, df: pd.DataFrame) -> None:
        p = self.raw_dir / f"{dataset}.parquet"
        df.to_parquet(p, index=False, compression="zstd")
        log.info("[%s] 全量 %d 行 -> %s", dataset, len(df), p.name)

    def _incremental_range(self, dataset: str, force: bool) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        lo = self.start if force or self._last_day(dataset) is None \
            else self._last_day(dataset) + pd.Timedelta(days=1)
        if lo > self.end:
            log.info("[%s] 已最新(%s)", dataset, self._last_day(dataset).date())
            return None
        return lo, self.end

    # ---------- 小表全量 ----------
    def sync_trading_days(self, force: bool = False) -> None:
        df = self.db.query(
            """
            SELECT TradingDate, IfWeekEnd, IfMonthEnd FROM dbo.QT_TradingDayNew
            WHERE SecuMarket = 83 AND IfTradingDay = 1
              AND TradingDate BETWEEN '2005-01-01' AND :e ORDER BY TradingDate
            """,
            {"e": self.end.date()},
        )
        self._write_full("trading_days", df)

    def sync_secu_main(self, force: bool = False) -> None:
        df = self.db.query(
            """
            SELECT InnerCode, CompanyCode, SecuCode, SecuAbbr, SecuMarket,
                   ListedSector, ListedDate, ListedState
            FROM dbo.SecuMain
            WHERE SecuCategory = 1 AND SecuMarket IN :m
            """,
            {"m": self.markets},
        )
        self._write_full("secu_main", df)

    def sync_adj_factor(self, force: bool = False) -> None:
        main = self.db.query(
            """
            SELECT q.InnerCode, q.ExDiviDate, q.AdjustingFactor, q.AdjustingConst
            FROM dbo.QT_AdjustingFactor q
            JOIN dbo.SecuMain s ON s.InnerCode = q.InnerCode
            WHERE s.SecuCategory = 1 AND s.SecuMarket IN :m
            """,
            {"m": self.markets},
        )
        parts = [main]
        try:
            parts.append(self.db.query(
                """
                SELECT q.InnerCode, q.ExDiviDate, q.AdjustingFactor, q.AdjustingConst
                FROM dbo.LC_STIBAdjustingFactor q
                JOIN dbo.SecuMain s ON s.InnerCode = q.InnerCode
                WHERE s.SecuCategory = 1 AND s.SecuMarket IN :m
                """,
                {"m": self.markets},
            ))
        except Exception as e:  # noqa: BLE001
            log.warning("科创板复权因子拉取失败(可能列名差异): %s", e)
        self._write_full("adj_factor", pd.concat(parts, ignore_index=True))

    def sync_special_trade(self, force: bool = False) -> None:
        df = self.db.query(
            """
            SELECT t.InnerCode, t.SpecialTradeType, t.SpecialTradeTime
            FROM dbo.LC_SpecialTrade t
            JOIN dbo.SecuMain s ON s.InnerCode = t.InnerCode
            WHERE s.SecuCategory = 1 AND s.SecuMarket IN :m
            """,
            {"m": self.markets},
        )
        self._write_full("special_trade", df)

    def sync_industry(self, force: bool = False) -> None:
        df = self.db.query(
            """
            SELECT i.CompanyCode, i.InfoPublDate, i.CancelDate, i.IfPerformed,
                   i.FirstIndustryCode, i.FirstIndustryName
            FROM dbo.LC_ExgIndustry i
            WHERE i.Standard = 38
            """
        )
        self._write_full("industry", df)

    def sync_index_quote(self, force: bool = False) -> None:
        df = self.db.query(
            """
            SELECT s.SecuCode, q.TradingDay, q.ClosePrice, q.ChangePCT
            FROM dbo.QT_IndexQuote q
            JOIN dbo.SecuMain s ON s.InnerCode = q.InnerCode
            WHERE s.SecuCategory = 4 AND s.SecuCode IN :codes
              AND q.TradingDay BETWEEN :a AND :b
            """,
            {"codes": tuple(self.cfg["data"]["benchmark_indexes"]),
             "a": self.start.date(), "b": self.end.date()},
        )
        self._write_full("index_quote", df)

    def sync_index_component(self, force: bool = False) -> None:
        df = self.db.query(
            """
            SELECT si.SecuCode AS IndexCode, c.SecuInnerCode AS InnerCode,
                   c.InDate, c.OutDate, c.Flag
            FROM dbo.LC_IndexComponent c
            JOIN dbo.SecuMain si ON si.InnerCode = c.IndexInnerCode
            WHERE si.SecuCategory = 4 AND si.SecuCode IN :codes
            """,
            {"codes": tuple(self.cfg["data"]["benchmark_indexes"])},
        )
        self._write_full("index_component", df)

    # ---------- 大表增量(分年, 主板∪科创板) ----------
    def _pull_paired_year(self, dataset: str, a: pd.Timestamp, b: pd.Timestamp) -> pd.DataFrame:
        main_tbl, stib_tbl, cols = PAIRED_SQL[dataset]
        col_sql = ", ".join(f"q.{c}" for c in cols)
        parts = []
        for tbl in (main_tbl, stib_tbl):
            try:
                parts.append(self.db.query(
                    f"""
                    SELECT q.InnerCode, {col_sql}
                    FROM {tbl} q
                    JOIN dbo.SecuMain s ON s.InnerCode = q.InnerCode
                    WHERE s.SecuCategory = 1 AND s.SecuMarket IN :m
                      AND q.TradingDay BETWEEN :a AND :b
                    """,
                    {"m": self.markets, "a": a.date(), "b": b.date()},
                ))
            except Exception as e:  # noqa: BLE001
                if tbl == main_tbl:
                    raise
                log.warning("[%s] 科创板表 %s 拉取失败, 跳过: %s", dataset, tbl, e)
        return pd.concat(parts, ignore_index=True)

    def _sync_daily_dataset(self, dataset: str, puller, force: bool) -> None:
        rng = self._incremental_range(dataset, force)
        if rng is None:
            return
        lo, hi = rng
        for year in range(lo.year, hi.year + 1):
            a = max(lo, pd.Timestamp(year, 1, 1))
            b = min(hi, pd.Timestamp(year, 12, 31))
            df = puller(a, b)
            if len(df):
                df["TradingDay"] = pd.to_datetime(df["TradingDay"])
                self._write_year(dataset, year, df, ["InnerCode", "TradingDay"])
            log.info("[%s] %d: +%d 行", dataset, year, len(df))
            self._set_last_day(dataset, b)

    def sync_daily_quote(self, force: bool = False) -> None:
        self._sync_daily_dataset(
            "daily_quote", lambda a, b: self._pull_paired_year("daily_quote", a, b), force)

    def sync_stock_perf(self, force: bool = False) -> None:
        self._sync_daily_dataset(
            "stock_perf", lambda a, b: self._pull_paired_year("stock_perf", a, b), force)

    def sync_valuation(self, force: bool = False) -> None:
        self._sync_daily_dataset(
            "valuation", lambda a, b: self._pull_paired_year("valuation", a, b), force)

    def sync_perf_data(self, force: bool = False) -> None:
        col_sql = ", ".join(f"q.{c}" for c in PERF_DATA_COLS)

        def pull(a: pd.Timestamp, b: pd.Timestamp) -> pd.DataFrame:
            return self.db.query(
                f"""
                SELECT q.InnerCode, {col_sql}
                FROM dbo.QT_PerformanceData q
                JOIN dbo.SecuMain s ON s.InnerCode = q.InnerCode
                WHERE s.SecuCategory = 1 AND s.SecuMarket IN :m
                  AND q.TradingDay BETWEEN :a AND :b
                """,
                {"m": self.markets, "a": a.date(), "b": b.date()},
            )

        self._sync_daily_dataset("perf_data", pull, force)

    # ---------- 编排 ----------
    SMALL = ["trading_days", "secu_main", "adj_factor", "special_trade",
             "industry", "index_quote", "index_component"]
    DAILY = ["daily_quote", "stock_perf", "valuation", "perf_data"]

    def sync(self, datasets: list[str] | None = None, force: bool = False) -> None:
        """数据集间并行同步(线程池, 网络IO型); 失败的数据集汇总后抛出。"""
        names = datasets or (self.SMALL + self.DAILY)
        errors: dict[str, Exception] = {}
        with ThreadPoolExecutor(max_workers=self.n_workers) as pool:
            futs = {pool.submit(getattr(self, f"sync_{n}"), force): n for n in names}
            for fut in as_completed(futs):
                n = futs[fut]
                try:
                    fut.result()
                    log.info("[%s] 完成", n)
                except Exception as e:  # noqa: BLE001
                    errors[n] = e
                    log.error("[%s] 失败: %s", n, e)
        if errors:
            raise RuntimeError(f"以下数据集同步失败: {list(errors)}")
        log.info("同步全部完成: %s", names)
