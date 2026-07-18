"""环境与数据库自检: python check_env.py"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

KEY_TABLES = [
    "SecuMain", "QT_TradingDayNew", "QT_DailyQuote", "LC_STIBDailyQuote",
    "QT_StockPerformance", "LC_STIBPerformance", "QT_PerformanceData",
    "QT_AdjustingFactor", "LC_STIBAdjustingFactor", "LC_DIndicesForValuation",
    "LC_STIBDIndiForValue", "LC_SpecialTrade", "LC_ExgIndustry",
    "QT_IndexQuote", "LC_IndexComponent", "CT_SystemConst",
]


def main() -> int:
    print("=" * 60)
    print("FactorMiner 环境自检")
    print("=" * 60)
    import numpy, pandas  # noqa: PLC0415

    print(f"python={sys.version.split()[0]} pandas={pandas.__version__} numpy={numpy.__version__}")
    try:
        import pymssql  # noqa: PLC0415
        print(f"pymssql={pymssql.__version__}")
    except ImportError:
        print("!! pymssql 未安装: pip install --no-cache-dir pymssql")
        return 1
    try:
        import pyarrow  # noqa: PLC0415
        print(f"pyarrow={pyarrow.__version__}")
    except ImportError:
        print("!! pyarrow 未安装: pip install --no-cache-dir pyarrow")
        return 1

    from factor_miner.config import get_config
    from factor_miner.data.juyuan import JuyuanDB

    cfg = get_config()
    print(f"config={cfg.path}")
    print(f"cache_dir={cfg.cache_dir}")

    db = JuyuanDB(cfg)
    ver = db.scalar("SELECT @@VERSION")
    print(f"连接成功: {str(ver).splitlines()[0]}")

    have = db.existing_tables(KEY_TABLES)
    missing = [t for t in KEY_TABLES if t not in have]
    print(f"关键表: {len(have)}/{len(KEY_TABLES)} 存在")
    if missing:
        print(f"!! 缺失: {missing}")

    end = db.data_end_date()
    print(f"数据终点(实测): {end}")

    n = db.scalar(
        "SELECT COUNT(*) FROM dbo.SecuMain WHERE SecuCategory=1 AND SecuMarket IN :m AND ListedState=1",
        {"m": tuple(cfg["data"]["markets"])},
    )
    print(f"股票池粗计(在市A股, markets={cfg['data']['markets']}): {n} 只")
    print("自检通过 [OK]" if not missing else "自检通过(有缺表警告)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
