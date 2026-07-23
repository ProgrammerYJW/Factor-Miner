"""RL引擎(AlphaGen式): PPO 逐token生成因子表达式.

回合: BOS -> token... -> END, 终端奖励 = |RankIC(训练段快评)| - λ1·池相关 - λ2·复杂度
(与GP适应度同构, 两引擎可比)。奖励做滑动标准化。已见表达式给予重复惩罚。
进度: artifacts/logs/rl_progress.jsonl; 检查点: artifacts/checkpoints/rl_policy.pt。

性能: 回合生成整批一次前向(替代逐回合单样本); 表达式评分复用GP工作进程池并行,
子进程只做纯计算, 日志/检查点/入库全部留在主进程。
"""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from factor_miner.config import Config, get_config
from factor_miner.engines.gp import worker as W
from factor_miner.engines.rl.policy import PolicyNet
from factor_miner.engines.rl.tokens import TokenSpace
from factor_miner.evaluation.evaluator import BASE_FEATURES, Evaluator
from factor_miner.library import Admission, FactorLibrary

log = logging.getLogger(__name__)


class RLEngine:
    def __init__(self, cfg: Config | None = None, seed: int = 7):
        self.cfg = cfg or get_config()
        r = self.cfg["rl"]
        ex = self.cfg["expression"]
        self.space = TokenSpace(BASE_FEATURES, list(ex["windows"]),
                                max_tokens=int(r["max_tokens"]))
        self.device = torch.device(str(r.get("device", "cpu")))
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.set_num_threads(4)                    # 小网络吃不满核, 别和评分进程抢
        self.net = PolicyNet(self.space.n_actions, self.space.max_tokens + 1,
                             d_model=int(r["d_model"]), n_layers=int(r["n_layers"]),
                             n_heads=int(r["n_heads"])).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=float(r["lr"]))
        self.clip = float(r["clip_ratio"])
        self.ent_coef = float(r["entropy_coef"])
        self.batch_episodes = int(r["batch_episodes"])
        self.total_updates = int(r["total_updates"])
        self.eval_workers = int(r.get("eval_workers", 4))
        self.lib = FactorLibrary(self.cfg)
        self.evaluator: Evaluator | None = None
        self.seen: dict[str, float] = {}              # key -> reward
        self.submitted: set[str] = set()
        self.rew_mean, self.rew_std, self.rew_n = 0.0, 1.0, 0
        self.log_path = self.cfg.artifacts_dir / "logs" / "rl_progress.jsonl"
        self.ckpt = self.cfg.artifacts_dir / "checkpoints" / "rl_policy.pt"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.ckpt.parent.mkdir(parents=True, exist_ok=True)

    # ---------- 因子池 ----------
    def _pool_exprs(self, limit: int = 10) -> list[str]:
        df = self.lib.list(status="active")
        return df["expression"].head(limit).tolist() if len(df) else []

    # ---------- 奖励 ----------
    def _norm(self, r: float) -> float:
        self.rew_n += 1
        a = 1.0 / min(self.rew_n, 1000)
        self.rew_mean = (1 - a) * self.rew_mean + a * r
        self.rew_std = (1 - a) * self.rew_std + a * abs(r - self.rew_mean)
        return float(np.clip((r - self.rew_mean) / max(self.rew_std, 1e-4), -5, 5))

    def _score(self, pool: ProcessPoolExecutor, pool_exprs: list[str],
               steps: list[list[dict]]) -> tuple[list[float], list[dict]]:
        """并行评分: 主进程解码/查重, 新表达式送工作进程评估(奖励=GP适应度同公式)。

        子进程只返回数字; 覆盖率不足/IC无效/求值异常 -> -0.5; 批内/历史重复 -> 旧奖励×0.3。
        """
        B = len(steps)
        raws: list[float] = [0.0] * B
        infos: list[dict] = [{} for _ in range(B)]
        keys: list[str | None] = [None] * B
        expr_strs: list[str] = [""] * B
        pre_seen = set(self.seen)
        todo: dict[str, str] = {}                      # key -> expr_str (保序)
        for i in range(B):
            rseq = [s["action"] for s in steps[i] if s["action"] != self.space.END]
            try:
                expr = self.space.decode(rseq)
            except Exception as e:  # noqa: BLE001
                raws[i] = -1.0
                infos[i] = {"error": str(e)[:80]}
                continue
            k = expr.key()
            keys[i] = k
            expr_strs[i] = expr.to_string()
            if k not in pre_seen and k not in todo:
                todo[k] = expr_strs[i]
        results: dict[str, tuple[float, dict]] = {}
        if todo:
            fcfg, ev = self.cfg["fitness"], self.cfg["evaluation"]
            items = list(todo.items())
            chunks = np.array_split(np.array(items, dtype=object),
                                    min(len(items), self.eval_workers * 4))
            futs = [pool.submit(W.eval_batch,
                                [s for _, s in ch], pool_exprs,
                                float(fcfg["lambda_corr"]), float(fcfg["lambda_complexity"]),
                                float(ev["min_coverage"]))
                    for ch in chunks if len(ch)]
            for fut, ch in zip(futs, [c for c in chunks if len(c)]):
                for (k, _s), (fit, info) in zip(ch, fut.result()):
                    results[k] = (fit, info)
        count: dict[str, int] = {}
        for i in range(B):
            k = keys[i]
            if k is None:
                continue
            count[k] = count.get(k, 0) + 1
            if k in pre_seen or count[k] > 1:          # 重复: 给旧奖励再打折, 抑制刷重复
                raws[i] = self.seen[k] * 0.3
                infos[i] = {"dup": True, "expr": expr_strs[i]}
                continue
            fit, info = results[k]
            if not np.isfinite(fit):                   # 覆盖率不足/IC无效/求值异常
                self.seen[k] = -0.5
                raws[i] = -0.5
                infos[i] = {"error": info.get("error", ""), "expr": todo[k]}
            else:
                self.seen[k] = fit
                raws[i] = fit
                infos[i] = {**info, "expr": todo[k], "key": k}
        return raws, infos

    # ---------- 采样 ----------
    @torch.no_grad()
    def _rollout(self, pool: ProcessPoolExecutor,
                 pool_exprs: list[str]) -> tuple[list[dict], list[dict]]:
        B = self.batch_episodes
        seqs: list[list[int]] = [[] for _ in range(B)]
        steps: list[list[dict]] = [[] for _ in range(B)]
        active = list(range(B))
        while active:                                  # 整批齐步走, 每步一次批量前向
            L = max(len(seqs[i]) for i in active) + 1
            inp = torch.zeros((len(active), L), dtype=torch.long, device=self.device)
            lens = torch.tensor([len(seqs[i]) + 1 for i in active], device=self.device)
            masks_np = np.stack([self.space.valid_mask(seqs[i]) for i in active])
            for r, i in enumerate(active):
                inp[r, 0] = self.net.bos
                if seqs[i]:
                    inp[r, 1: len(seqs[i]) + 1] = torch.tensor(seqs[i], dtype=torch.long)
            logits, values = self.net(inp, lens, torch.tensor(masks_np, device=self.device))
            dist = torch.distributions.Categorical(logits=logits)
            acts = dist.sample()
            logps = dist.log_prob(acts)
            nxt = []
            for r, i in enumerate(active):
                a = int(acts[r])
                steps[i].append({"mask": masks_np[r], "action": a,
                                 "logp": float(logps[r]), "value": float(values[r])})
                if a == self.space.END or len(seqs[i]) + 1 >= self.space.max_tokens:
                    if a != self.space.END:
                        steps[i][-1]["forced_end"] = True
                else:
                    seqs[i].append(a)
                    nxt.append(i)
            active = nxt
        raws, infos = self._score(pool, pool_exprs, steps)
        episodes = []
        for i in range(B):
            raw = raws[i]
            if steps[i][-1].get("forced_end"):
                raw = min(raw, -0.5)                    # 未正常收敛: 惩罚
            episodes.append({"steps": steps[i], "reward": self._norm(raw), "raw": raw})
        return episodes, infos

    # ---------- PPO更新 ----------
    def _update(self, episodes: list[dict]) -> dict:
        obs_seq, obs_len, obs_mask, acts, logps, advs, rets = [], [], [], [], [], [], []
        for ep in episodes:
            R = ep["reward"]
            prefix: list[int] = []
            for s in ep["steps"]:
                obs_seq.append([self.net.bos, *prefix])
                obs_len.append(len(prefix) + 1)
                obs_mask.append(s["mask"])
                acts.append(s["action"])
                logps.append(s["logp"])
                advs.append(R - s["value"])            # γ=1, 终端奖励
                rets.append(R)
                if s["action"] != self.space.END:
                    prefix.append(s["action"])
        L = max(obs_len)
        seqs = torch.full((len(obs_seq), L), 0, dtype=torch.long, device=self.device)
        for i, s in enumerate(obs_seq):
            seqs[i, : len(s)] = torch.tensor(s, device=self.device)
        lens = torch.tensor(obs_len, device=self.device)
        masks = torch.tensor(np.array(obs_mask), device=self.device)
        acts_t = torch.tensor(acts, device=self.device)
        old_logp = torch.tensor(logps, device=self.device)
        adv = torch.tensor(advs, dtype=torch.float32, device=self.device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        ret = torch.tensor(rets, dtype=torch.float32, device=self.device)

        stats = {}
        idx_all = np.arange(len(acts))
        for _ in range(4):                             # epochs
            np.random.shuffle(idx_all)
            for chunk in np.array_split(idx_all, max(1, len(idx_all) // 512)):
                b = torch.tensor(chunk, device=self.device)
                logits, value = self.net(seqs[b], lens[b], masks[b])
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(acts_t[b])
                ratio = torch.exp(lp - old_logp[b])
                s1 = ratio * adv[b]
                s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv[b]
                pi_loss = -torch.min(s1, s2).mean()
                v_loss = F.mse_loss(value, ret[b])
                ent = dist.entropy().mean()
                loss = pi_loss + 0.5 * v_loss - self.ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.opt.step()
                stats = {"pi_loss": float(pi_loss), "v_loss": float(v_loss),
                         "entropy": float(ent)}
        return stats

    # ---------- 准入 ----------
    def _try_admit(self, infos: list[dict]) -> int:
        from factor_miner.library.rules import RuleSet

        rs = RuleSet.load(self.cfg)
        ic_thr = rs.threshold_of("ic_mean") or rs.threshold_of("rank_ic", op_prefix="") or 0.0
        ir_thr = rs.threshold_of("icir") or 0.0
        cands = sorted(
            (i for i in infos
             if i.get("key") and i["key"] not in self.submitted
             and abs(i.get("ic_mean", 0) or 0) >= ic_thr
             and abs(i.get("icir", 0) or 0) >= 0.8 * ir_thr),
            key=lambda x: abs(x.get("ic_mean", 0) or 0), reverse=True)
        n_ok = 0
        for i in cands[:2]:
            self.submitted.add(i["key"])
            if self.evaluator is None:
                log.info("加载全量评估器(首次准入)...")
                self.evaluator = Evaluator(self.cfg)
            from factor_miner.expression.parser import parse
            ok, reason, fid = Admission(self.evaluator, self.lib, self.cfg).submit(
                parse(i["expr"]), engine="RL")
            log.info("准入 %s: %s %s", i["expr"][:60],
                     "OK#%s" % fid if ok else "拒绝", reason)
            n_ok += int(ok)
        return n_ok

    # ---------- 主循环 ----------
    def run(self, total_updates: int | None = None, resume: bool = False) -> None:
        start = 0
        if resume and self.ckpt.exists():
            st = torch.load(self.ckpt, map_location=self.device, weights_only=False)
            self.net.load_state_dict(st["net"])
            self.opt.load_state_dict(st["opt"])
            start = int(st.get("update", 0)) + 1
            self.seen = st.get("seen", {})
            self.submitted = set(st.get("submitted", []))
            log.info("从检查点续训: update=%d, seen=%d", start, len(self.seen))
        n_up = total_updates or self.total_updates
        with ProcessPoolExecutor(max_workers=self.eval_workers,
                                 initializer=W.init_worker) as pool:
            pool_exprs = self._pool_exprs()
            for u in range(start, n_up):
                t0 = time.perf_counter()
                if u % 10 == 0:
                    pool_exprs = self._pool_exprs()
                episodes, infos = self._rollout(pool, pool_exprs)
                t_ro = time.perf_counter() - t0
                stats = self._update(episodes)
                t_up = time.perf_counter() - t0 - t_ro
                admitted = self._try_admit(infos)
                t_ad = time.perf_counter() - t0 - t_ro - t_up
                log.info("upd %d 耗时: 采样评分 %.1fs / PPO %.1fs / 准入 %.1fs",
                         u, t_ro, t_up, t_ad)
                raws = [e["raw"] for e in episodes]
                best_i = int(np.argmax(raws))
                rec = {"ts": datetime.now().isoformat(timespec="seconds"), "update": u,
                       "reward_mean": round(float(np.mean(raws)), 5),
                       "reward_best": round(float(raws[best_i]), 5),
                       "best_expr": infos[best_i].get("expr", ""),
                       "best_ic": infos[best_i].get("ic_mean"),
                       "best_icir": infos[best_i].get("icir"),
                       "n_unique": len(self.seen), "admitted": admitted, **stats}
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if u % 5 == 0:
                    log.info("upd %d rew=%.4f best=%.4f uniq=%d %s", u, rec["reward_mean"],
                             rec["reward_best"], rec["n_unique"], rec["best_expr"][:60])
                    torch.save({"net": self.net.state_dict(), "opt": self.opt.state_dict(),
                                "update": u, "seen": self.seen,
                                "submitted": list(self.submitted)}, self.ckpt)
        log.info("RL完成 %d updates", n_up)
