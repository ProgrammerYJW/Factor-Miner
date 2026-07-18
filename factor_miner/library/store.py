"""因子库: SQLite(WAL) 元数据 + parquet 因子值/IC序列. 增删改查 + 表达式级查重.

存储布局(artifacts/):
  factor_library.db            factors 表(元数据+指标JSON)
  factor_values/{id}.parquet   预处理后因子值宽表(相关性查重与UI画图用)
  ic_series/{id}.parquet       各周期日度RankIC序列
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from factor_miner.config import Config, get_config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS factors (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  name       TEXT UNIQUE NOT NULL,
  expression TEXT NOT NULL,
  expr_key   TEXT UNIQUE NOT NULL,
  engine     TEXT NOT NULL,
  status     TEXT NOT NULL DEFAULT 'active',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  tags       TEXT NOT NULL DEFAULT '',
  notes      TEXT NOT NULL DEFAULT '',
  metrics    TEXT NOT NULL
);
"""

# 列表页提取的核心指标(⭐ICIR置顶), 来源 metrics JSON
_SUMMARY_KEYS = [
    ("icir10_train", "h10_train", "icir"),
    ("icir10_valid", "h10_valid", "icir"),
    ("ic10_train", "h10_train", "ic_mean"),
    ("ic10_valid", "h10_valid", "ic_mean"),
    ("ls_ann_train", "h10_train", "ls_ann_ret"),
    ("ls_sharpe_train", "h10_train", "ls_sharpe"),
    ("rank_autocorr", "h10_train", "rank_autocorr"),
]


class FactorLibrary:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or get_config()
        self.dir = self.cfg.artifacts_dir
        self.values_dir = self.dir / "factor_values"
        self.ic_dir = self.dir / "ic_series"
        for d in (self.dir, self.values_dir, self.ic_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.db_path = self.dir / "factor_library.db"
        with self._conn() as c:
            c.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path, timeout=30)
        c.execute("PRAGMA journal_mode=WAL")
        c.row_factory = sqlite3.Row
        return c

    # ---------- 增 ----------
    def add(self, expression: str, expr_key: str, engine: str, metrics: dict,
            factor: pd.DataFrame, ic_series: dict[str, pd.Series],
            name: str | None = None, status: str = "active") -> int:
        now = datetime.now().isoformat(timespec="seconds")
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO factors(name,expression,expr_key,engine,status,"
                "created_at,updated_at,metrics) VALUES(?,?,?,?,?,?,?,?)",
                (name or f"__tmp__{expr_key}", expression, expr_key, engine,
                 status, now, now, json.dumps(metrics, ensure_ascii=False)),
            )
            fid = int(cur.lastrowid)
            if name is None:
                auto = f"{engine}_{datetime.now():%Y%m%d}_{fid:04d}"
                c.execute("UPDATE factors SET name=? WHERE id=?", (auto, fid))
        factor.astype("float32").to_parquet(self.values_dir / f"{fid}.parquet",
                                            compression="zstd")
        pd.DataFrame(ic_series).to_parquet(self.ic_dir / f"{fid}.parquet",
                                           compression="zstd")
        return fid

    # ---------- 查 ----------
    def exists(self, expr_key: str) -> bool:
        with self._conn() as c:
            r = c.execute("SELECT 1 FROM factors WHERE expr_key=?", (expr_key,)).fetchone()
        return r is not None

    def get(self, fid: int) -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM factors WHERE id=?", (fid,)).fetchone()
        if r is None:
            return None
        d = dict(r)
        d["metrics"] = json.loads(d["metrics"])
        return d

    def list(self, status: str | None = None) -> pd.DataFrame:
        sql = "SELECT * FROM factors"
        args: tuple = ()
        if status:
            sql += " WHERE status=?"
            args = (status,)
        with self._conn() as c:
            rows = [dict(r) for r in c.execute(sql, args).fetchall()]
        if not rows:
            return pd.DataFrame()
        recs = []
        for r in rows:
            m = json.loads(r.pop("metrics"))
            rec = {**r}
            for out_key, seg, k in _SUMMARY_KEYS:
                rec[out_key] = m.get(seg, {}).get(k)
            rec["coverage"] = m.get("coverage")
            rec["n_nodes"] = m.get("n_nodes")
            recs.append(rec)
        df = pd.DataFrame(recs)
        if "icir10_train" in df:
            df = df.sort_values("icir10_train", key=lambda s: s.abs(),
                                ascending=False, na_position="last")
        return df.reset_index(drop=True)

    def load_values(self, fid: int) -> pd.DataFrame:
        return pd.read_parquet(self.values_dir / f"{fid}.parquet")

    def load_ic(self, fid: int) -> pd.DataFrame:
        return pd.read_parquet(self.ic_dir / f"{fid}.parquet")

    def active_value_matrices(self, limit: int | None = None) -> dict[int, pd.DataFrame]:
        df = self.list(status="active")
        ids = df["id"].tolist()[: limit or None] if len(df) else []
        return {i: self.load_values(i) for i in ids
                if (self.values_dir / f"{i}.parquet").exists()}

    # ---------- 改 ----------
    def update(self, fid: int, *, name: str | None = None, tags: str | None = None,
               notes: str | None = None, status: str | None = None,
               metrics: dict | None = None) -> None:
        sets, args = [], []
        for col, v in (("name", name), ("tags", tags), ("notes", notes),
                       ("status", status)):
            if v is not None:
                sets.append(f"{col}=?")
                args.append(v)
        if metrics is not None:
            sets.append("metrics=?")
            args.append(json.dumps(metrics, ensure_ascii=False))
        if not sets:
            return
        sets.append("updated_at=?")
        args.append(datetime.now().isoformat(timespec="seconds"))
        args.append(fid)
        with self._conn() as c:
            c.execute(f"UPDATE factors SET {', '.join(sets)} WHERE id=?", args)

    # ---------- 删 ----------
    def delete(self, fid: int, hard: bool = False) -> None:
        """默认软删(归档); hard=True 连数据文件一并移除。"""
        if not hard:
            self.update(fid, status="archived")
            return
        with self._conn() as c:
            c.execute("DELETE FROM factors WHERE id=?", (fid,))
        for p in (self.values_dir / f"{fid}.parquet", self.ic_dir / f"{fid}.parquet"):
            Path(p).unlink(missing_ok=True)
