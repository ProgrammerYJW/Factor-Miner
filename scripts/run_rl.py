"""RL引擎入口: python run_rl.py [--updates N] [--resume]"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    from factor_miner.engines.rl import RLEngine

    RLEngine(seed=args.seed).run(total_updates=args.updates, resume=args.resume)


if __name__ == "__main__":
    main()
