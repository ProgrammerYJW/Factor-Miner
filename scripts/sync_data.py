"""数据同步入口: python sync_data.py [--datasets a,b] [--force]"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    ap = argparse.ArgumentParser(description="聚源JYDB -> 本地parquet 同步")
    ap.add_argument("--datasets", default="", help="逗号分隔; 空=全部")
    ap.add_argument("--force", action="store_true", help="忽略增量记录全量重拉")
    args = ap.parse_args()

    from factor_miner.data.sync import Syncer

    s = Syncer()
    names = [x.strip() for x in args.datasets.split(",") if x.strip()] or None
    s.sync(names, force=args.force)


if __name__ == "__main__":
    main()
