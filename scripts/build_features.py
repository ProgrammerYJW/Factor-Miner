"""特征构建入口: python build_features.py"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from factor_miner.data.features import build_features  # noqa: E402

if __name__ == "__main__":
    build_features()
