"""聚源 JYDB 只读客户端 (SQL Server, pymssql + SQLAlchemy).

安全约定(遵守技能包规定): 仅允许 SELECT/WITH 单语句查询, 严禁任何写操作。
连接失败时提示关闭代理(环境文档警告: 开梯子无法连接内网数据库)。
"""
from __future__ import annotations

import logging
import re
import time
from datetime import date
from urllib.parse import quote_plus

import pandas as pd
from sqlalchemy import bindparam, create_engine, text

from factor_miner.config import Config, get_config

log = logging.getLogger(__name__)

_READONLY_RE = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)


class ReadOnlyViolation(ValueError):
    pass


def _assert_readonly(sql: str) -> None:
    """只允许单条 SELECT/WITH。剥掉注释后校验首个关键字, 并禁止语句内分号。"""
    body = re.sub(r"--[^\n]*", "", sql)
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.DOTALL).strip()
    if not _READONLY_RE.match(body):
        raise ReadOnlyViolation(f"仅允许 SELECT/WITH 查询, 拒绝执行: {body[:80]!r}")
    if ";" in body.rstrip(";"):
        raise ReadOnlyViolation("禁止多语句查询")


class JuyuanDB:
    """线程安全的连接池客户端。用法: db = JuyuanDB(); df = db.query(sql, params)."""

    def __init__(self, cfg: Config | None = None):
        c = (cfg or get_config())["database"]
        url = (
            f"mssql+pymssql://{c['user']}:{quote_plus(c['password'])}"
            f"@{c['server']}:{c['port']}/{c['database']}?charset=utf8"
        )
        self.engine = create_engine(
            url, pool_pre_ping=True, pool_recycle=3600, pool_size=10, max_overflow=6
        )

    def query(self, sql: str, params: dict | None = None, retries: int = 3) -> pd.DataFrame:
        _assert_readonly(sql)
        stmt = text(sql)
        if params:
            # 序列参数自动做 expanding IN 展开: "IN :m" + (83,90) -> IN (83,90)
            expanding = [bindparam(k, expanding=True)
                         for k, v in params.items() if isinstance(v, (list, tuple, set))]
            if expanding:
                stmt = stmt.bindparams(*expanding)
        last_err: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                t0 = time.time()
                with self.engine.connect() as conn:
                    df = pd.read_sql(stmt, conn, params=params)
                log.debug("query %d rows in %.1fs", len(df), time.time() - t0)
                return df
            except ReadOnlyViolation:
                raise
            except Exception as e:  # noqa: BLE001 连接类错误重试
                last_err = e
                log.warning("query attempt %d/%d failed: %s", attempt, retries, e)
                if attempt < retries:
                    time.sleep(2 * attempt)
        raise RuntimeError(
            f"聚源查询失败(已重试{retries}次): {last_err}\n"
            "提示: 若为连接超时, 请确认 1)未开启代理/梯子 2)可达 192.168.219.222:1433"
        ) from last_err

    def scalar(self, sql: str, params: dict | None = None):
        df = self.query(sql, params)
        return None if df.empty else df.iloc[0, 0]

    # ---------- 元信息 ----------
    def table_exists(self, name: str) -> bool:
        return bool(
            self.scalar(
                "SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = :n",
                {"n": name},
            )
        )

    def existing_tables(self, names: list[str]) -> set[str]:
        df = self.query(
            "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME IN :ns",
            {"ns": tuple(names)},
        )
        return set(df["TABLE_NAME"])

    def data_end_date(self) -> date:
        """探测行情数据终点: 沪深300指数行情与浦发银行日行情最大日期取交集(min)。"""
        idx_end = self.scalar(
            """
            SELECT MAX(q.TradingDay) FROM dbo.QT_IndexQuote q
            JOIN dbo.SecuMain s ON s.InnerCode = q.InnerCode
            WHERE s.SecuCode = '000300' AND s.SecuCategory = 4
            """
        )
        stk_end = self.scalar(
            """
            SELECT MAX(q.TradingDay) FROM dbo.QT_DailyQuote q
            JOIN dbo.SecuMain s ON s.InnerCode = q.InnerCode
            WHERE s.SecuCode = '600000' AND s.SecuCategory = 1
            """
        )
        ends = [pd.Timestamp(x).date() for x in (idx_end, stk_end) if x is not None]
        if not ends:
            raise RuntimeError("无法探测数据终点: 行情表为空?")
        return min(ends)
