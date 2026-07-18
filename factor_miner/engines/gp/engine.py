"""GP主引擎: 种群演化 -> 并行快评 -> 精英/锦标赛/交叉变异 -> 达标者提交准入.

进度落盘 artifacts/logs/gp_progress.jsonl (Web监控读取);
检查点 artifacts/checkpoints/gp_state.json 支持断点续跑。
"""
from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import numpy as np

from factor_miner.config import Config, get_config
from factor_miner.engines.gp import worker as W
from factor_miner.evaluation.evaluator import BASE_FEATURES, Evaluator
from factor_miner.expression.nodes import Expr
from factor_miner.expression.parser import parse
from factor_miner.expression.random_gen import ExprSampler
from factor_miner.library import Admission, FactorLibrary

log = logging.getLogger(__name__)


class GPEngine:
    def __init__(self, cfg: Config | None = None, seed: int = 42):
        self.cfg = cfg or get_config()
        g = self.cfg["gp"]
        self.pop_size = int(g["population_size"])
        self.generations = int(g["generations"])
        self.tour = int(g["tournament_size"])
        self.p_cx = float(g["p_crossover"])
        self.p_sub = float(g["p_subtree_mutation"])
        self.p_pt = float(g["p_point_mutation"])
        self.p_hoist = float(g["p_hoist_mutation"])
        self.elitism = int(g["elitism"])
        self.n_workers = int(g["n_workers"]) or max(1, (os.cpu_count() or 4) - 1)
        ex = self.cfg["expression"]
        self.sampler = ExprSampler(
            features=BASE_FEATURES, windows=list(ex["windows"]),
            max_depth=int(ex["max_depth"]), max_nodes=int(ex["max_nodes"]), seed=seed)
        self.rng = np.random.default_rng(seed)
        self.lib = FactorLibrary(self.cfg)
        self.evaluator: Evaluator | None = None       # 惰性加载(准入时才需要)
        self.fitness: dict[str, float] = {}           # key -> fitness
        self.info: dict[str, dict] = {}
        self.exprs: dict[str, Expr] = {}
        self.submitted: set[str] = set()
        self.log_dir = self.cfg.artifacts_dir / "logs"
        self.ckpt = self.cfg.artifacts_dir / "checkpoints" / "gp_state.json"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt.parent.mkdir(parents=True, exist_ok=True)

    # ---------- 池与准入 ----------
    def _pool_exprs(self, limit: int = 10) -> list[str]:
        df = self.lib.list(status="active")
        if not len(df):
            return []
        return df["expression"].head(limit).tolist()

    def _try_admit(self, cands: list[str]) -> int:
        from factor_miner.library.rules import RuleSet

        rs = RuleSet.load(self.cfg)
        ic_thr = rs.threshold_of("ic_mean") or 0.0
        ir_thr = rs.threshold_of("icir") or 0.0
        pre = [k for k in cands
               if k not in self.submitted and np.isfinite(self.fitness.get(k, -np.inf))
               and abs(self.info[k].get("ic_mean", 0)) >= ic_thr
               and abs(self.info[k].get("icir", 0)) >= 0.8 * ir_thr]
        n_ok = 0
        for k in pre[:3]:                              # 每代最多全评3个, 控制成本
            self.submitted.add(k)
            if self.evaluator is None:
                log.info("加载全量评估器(首次准入)...")
                self.evaluator = Evaluator(self.cfg)
            adm = Admission(self.evaluator, self.lib, self.cfg)
            ok, reason, fid = adm.submit(self.exprs[k], engine="GP")
            log.info("准入 %s: %s %s", self.exprs[k].to_string()[:60],
                     "OK#%s" % fid if ok else "拒绝", reason)
            n_ok += int(ok)
        return n_ok

    # ---------- 演化 ----------
    def _offspring(self, ranked: list[str]) -> Expr:
        def pick() -> Expr:
            idx = self.rng.integers(0, len(ranked), size=min(self.tour, len(ranked)))
            best = min(idx)                            # ranked 已按适应度降序
            return self.exprs[ranked[best]]

        r = self.rng.random()
        if r < self.p_cx:
            return self.sampler.crossover(pick(), pick())
        if r < self.p_cx + self.p_sub:
            return self.sampler.mutate_subtree(pick())
        if r < self.p_cx + self.p_sub + self.p_pt:
            return self.sampler.mutate_point(pick())
        if r < self.p_cx + self.p_sub + self.p_pt + self.p_hoist:
            return self.sampler.mutate_hoist(pick())
        return pick()

    def _evaluate(self, pool: ProcessPoolExecutor, keys: list[str]) -> None:
        todo = [k for k in keys if k not in self.fitness]
        if not todo:
            return
        fcfg, ev = self.cfg["fitness"], self.cfg["evaluation"]
        pool_exprs = self._pool_exprs()
        chunks = np.array_split(np.array(todo, dtype=object), self.n_workers * 4)
        futs = [pool.submit(W.eval_batch,
                            [self.exprs[k].to_string() for k in ch], pool_exprs,
                            float(fcfg["lambda_corr"]), float(fcfg["lambda_complexity"]),
                            float(ev["min_coverage"]))
                for ch in chunks if len(ch)]
        i = 0
        for fut, ch in zip(futs, [c for c in chunks if len(c)]):
            for k, (fit, info) in zip(ch, fut.result()):
                self.fitness[k] = fit
                self.info[k] = info
                i += 1
        log.info("评估 %d 个新个体", i)

    def _log_gen(self, gen: int, ranked: list[str], admitted: int) -> None:
        top = ranked[0]
        rec = {
            "ts": datetime.now().isoformat(timespec="seconds"), "gen": gen,
            "best_fitness": round(self.fitness[top], 5),
            "best_expr": self.exprs[top].to_string(),
            "best_ic": self.info[top].get("ic_mean"),
            "best_icir": self.info[top].get("icir"),
            "median_fitness": round(float(np.median(
                [self.fitness[k] for k in ranked if np.isfinite(self.fitness[k])])), 5),
            "n_valid": int(sum(np.isfinite(self.fitness[k]) for k in ranked)),
            "admitted": admitted,
        }
        with open(self.log_dir / "gp_progress.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log.info("Gen %d best=%.4f ic=%.4f icir=%.3f %s", gen, rec["best_fitness"],
                 rec["best_ic"] or 0, rec["best_icir"] or 0, rec["best_expr"][:70])

    # ---------- 检查点 ----------
    def _save_ckpt(self, gen: int, population: list[str]) -> None:
        state = {"gen": gen,
                 "population": [self.exprs[k].to_string() for k in population],
                 "submitted": [self.exprs[k].to_string() for k in self.submitted
                               if k in self.exprs]}
        self.ckpt.write_text(json.dumps(state, ensure_ascii=False), "utf-8")

    def _load_ckpt(self) -> tuple[int, list[str]] | None:
        if not self.ckpt.exists():
            return None
        st = json.loads(self.ckpt.read_text("utf-8"))
        pop = []
        for s in st["population"]:
            e = parse(s)
            self.exprs[e.key()] = e
            pop.append(e.key())
        for s in st.get("submitted", []):
            self.submitted.add(parse(s).key())
        return int(st["gen"]), pop

    # ---------- 主循环 ----------
    def run(self, generations: int | None = None, resume: bool = False) -> None:
        gens = generations or self.generations
        start_gen, population = 0, []
        if resume and (ck := self._load_ckpt()):
            start_gen, population = ck[0] + 1, ck[1]
            log.info("从检查点续跑: gen=%d, 种群=%d", start_gen, len(population))
        if not population:
            g = self.cfg["gp"]
            for e in self.sampler.ramped(self.pop_size * 2,
                                         int(g["init_depth_min"]), int(g["init_depth_max"])):
                if e.key() not in self.exprs:
                    self.exprs[e.key()] = e
                    population.append(e.key())
                if len(population) >= self.pop_size:
                    break
        with ProcessPoolExecutor(max_workers=self.n_workers,
                                 initializer=W.init_worker) as pool:
            for gen in range(start_gen, gens):
                self._evaluate(pool, population)
                ranked = sorted(population,
                                key=lambda k: self.fitness.get(k, -np.inf), reverse=True)
                admitted = self._try_admit(ranked[:10])
                self._log_gen(gen, ranked, admitted)
                self._save_ckpt(gen, ranked)
                nxt = list(ranked[: self.elitism])
                seen = set(nxt)
                tries = 0
                while len(nxt) < self.pop_size and tries < self.pop_size * 20:
                    tries += 1
                    child = self._offspring(ranked)
                    k = child.key()
                    if k in seen:
                        continue
                    seen.add(k)
                    self.exprs[k] = child
                    nxt.append(k)
                while len(nxt) < self.pop_size:
                    e = self.sampler.grow(4)
                    if e.key() not in seen:
                        seen.add(e.key())
                        self.exprs[e.key()] = e
                        nxt.append(e.key())
                population = nxt
        log.info("GP完成 %d 代", gens)
