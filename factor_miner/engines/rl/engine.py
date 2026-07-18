"""RL引擎(AlphaGen式): PPO 逐token生成因子表达式.

回合: BOS -> token... -> END, 终端奖励 = |RankIC(训练段快评)| - λ1·池相关 - λ2·复杂度
(与GP适应度同构, 两引擎可比)。奖励做滑动标准化。已见表达式给予重复惩罚。
进度: artifacts/logs/rl_progress.jsonl; 检查点: artifacts/checkpoints/rl_policy.pt。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from factor_miner.config import Config, get_config
from factor_miner.engines.rl.policy import PolicyNet
from factor_miner.engines.rl.tokens import TokenSpace
from factor_miner.evaluation import metrics as M
from factor_miner.evaluation.evaluator import BASE_FEATURES, Evaluator
from factor_miner.evaluation.fast_ctx import FastContext
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
        self.net = PolicyNet(self.space.n_actions, self.space.max_tokens + 1,
                             d_model=int(r["d_model"]), n_layers=int(r["n_layers"]),
                             n_heads=int(r["n_heads"])).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=float(r["lr"]))
        self.clip = float(r["clip_ratio"])
        self.ent_coef = float(r["entropy_coef"])
        self.batch_episodes = int(r["batch_episodes"])
        self.total_updates = int(r["total_updates"])
        self.fast = FastContext(self.cfg)
        self.lib = FactorLibrary(self.cfg)
        self.evaluator: Evaluator | None = None
        self.seen: dict[str, float] = {}              # key -> reward
        self.submitted: set[str] = set()
        self.rew_mean, self.rew_std, self.rew_n = 0.0, 1.0, 0
        self.log_path = self.cfg.artifacts_dir / "logs" / "rl_progress.jsonl"
        self.ckpt = self.cfg.artifacts_dir / "checkpoints" / "rl_policy.pt"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.ckpt.parent.mkdir(parents=True, exist_ok=True)

    # ---------- 奖励 ----------
    def _pool_grids(self, limit: int = 10) -> list:
        df = self.lib.list(status="active")
        grids = []
        if len(df):
            from factor_miner.expression.parser import parse
            for s in df["expression"].head(limit):
                try:
                    grids.append(self.fast.factor_on_grid(parse(s)))
                except Exception:  # noqa: BLE001
                    continue
        return grids

    def _terminal_reward(self, seq: list[int], pool_grids: list) -> tuple[float, dict]:
        fcfg, ev = self.cfg["fitness"], self.cfg["evaluation"]
        try:
            expr = self.space.decode(seq)
        except Exception as e:  # noqa: BLE001
            return -1.0, {"error": str(e)[:80]}
        k = expr.key()
        if k in self.seen:                             # 重复: 给旧奖励再打折, 抑制刷重复
            return self.seen[k] * 0.3, {"dup": True, "expr": expr.to_string()}
        try:
            f = self.fast.factor_on_grid(expr)
            uni = self.fast.universe.iloc[:: self.fast.step]
            cov = M.coverage(f, uni)
            if not np.isfinite(cov) or cov < float(ev["min_coverage"]):
                self.seen[k] = -0.5
                return -0.5, {"error": f"cov={cov}", "expr": expr.to_string()}
            ic = M.daily_rank_ic(f, self.fast.label_on_grid())
            st = M.ic_stats(ic, self.fast.horizon)
            if not np.isfinite(st["ic_mean"]):
                self.seen[k] = -0.5
                return -0.5, {"error": "ic nan", "expr": expr.to_string()}
            max_corr = 0.0
            for g in pool_grids:
                c = M.value_corr(f, g, step=1)
                if np.isfinite(c):
                    max_corr = max(max_corr, abs(c))
            rew = abs(st["ic_mean"]) \
                - float(fcfg["lambda_corr"]) * max_corr \
                - float(fcfg["lambda_complexity"]) * expr.n_nodes()
            self.seen[k] = rew
            return rew, {**st, "expr": expr.to_string(), "key": k,
                         "max_corr_pool": round(max_corr, 4)}
        except Exception as e:  # noqa: BLE001
            self.seen[k] = -0.5
            return -0.5, {"error": str(e)[:80], "expr": expr.to_string()}

    def _norm(self, r: float) -> float:
        self.rew_n += 1
        a = 1.0 / min(self.rew_n, 1000)
        self.rew_mean = (1 - a) * self.rew_mean + a * r
        self.rew_std = (1 - a) * self.rew_std + a * abs(r - self.rew_mean)
        return float(np.clip((r - self.rew_mean) / max(self.rew_std, 1e-4), -5, 5))

    # ---------- 采样 ----------
    @torch.no_grad()
    def _rollout(self, pool_grids: list) -> tuple[list[dict], list[dict]]:
        episodes, infos = [], []
        for _ in range(self.batch_episodes):
            seq: list[int] = []
            steps = []
            while True:
                mask = self.space.valid_mask(seq)
                inp = torch.tensor([[self.net.bos, *seq]], device=self.device)
                ln = torch.tensor([len(seq) + 1], device=self.device)
                mk = torch.tensor(mask[None], device=self.device)
                logits, value = self.net(inp, ln, mk)
                dist = torch.distributions.Categorical(logits=logits[0])
                a = int(dist.sample())
                steps.append({"mask": mask, "action": a,
                              "logp": float(dist.log_prob(torch.tensor(a, device=self.device))),
                              "value": float(value[0])})
                if a == self.space.END or len(seq) + 1 >= self.space.max_tokens:
                    if a != self.space.END:
                        steps[-1]["forced_end"] = True
                    break
                seq.append(a)
            raw, info = self._terminal_reward(
                [s["action"] for s in steps if s["action"] != self.space.END], pool_grids)
            if steps[-1].get("forced_end"):
                raw = min(raw, -0.5)                    # 未正常收敛: 惩罚
            episodes.append({"steps": steps, "reward": self._norm(raw), "raw": raw})
            infos.append(info)
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
        ic_thr = rs.threshold_of("ic_mean") or 0.0
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
        for u in range(start, n_up):
            pool_grids = self._pool_grids() if u % 10 == 0 or u == start else pool_grids
            episodes, infos = self._rollout(pool_grids)
            stats = self._update(episodes)
            admitted = self._try_admit(infos)
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
