#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
okg_from_preds.py

读取由 `inference.py` 生成的 preds.jsonl（每题包含多次 LLM 输出的候选答案列表），
运行 OKG（Optimistic Knowledge Gradient）式的分配策略，在 “无放回” 的前提下
逐步从各题的候选池中抽样答案；同时为每题累积一个投票列表，并在最终导出 jsonl
（同原格式，含 "final" 多数投票）以及可选的 "预算-准确率" 曲线 CSV。

Usage:
  # 十选（默认 A-J），输出 OKG 结果
  python curve_okg_from_preds.py --preds preds_pro.jsonl --out preds_pro_okg.jsonl --T 10000

  # 四选（A-D），同时记录随预算增长的准确率曲线到 curve_okg.csv
  python curve_okg_from_preds.py \
    --preds preds.jsonl \
    --out preds_okg.jsonl \
    --choices A,B,C,D \
    --labels labels.jsonl \
    --curve_csv curve_okg.csv \
    --T 2000
"""

from __future__ import annotations
import json
import argparse
from typing import Sequence, Dict, List, Tuple, Optional
import numpy as np


# ----------------- helpers -----------------

def vote_majority(answers: List[str]) -> str:
    """
    多数投票；显式平票规则：若出现次数并列，选择在序列中“先出现”的那个选项。
    这样不依赖 Counter.most_common 的内部实现细节。
    """
    if not answers:
        return ""
    from collections import Counter
    cnt = Counter(answers)
    maxc = max(cnt.values())
    tied = {a for a, c in cnt.items() if c == maxc}
    firstpos = {}
    for i, a in enumerate(answers):
        if a in tied and a not in firstpos:
            firstpos[a] = i
    return min(tied, key=lambda a: firstpos[a])


# ----------------- OKG Allocator -----------------

class OKGAllocator:
    """
    与仓库相同思路：
    - 以 Dirichlet-Gamma 采样近似“每个类别成为最大值的概率向量 I(alpha)”
    - 计算对每个题 i，在所有类别 m 中“+1 观测”后 h(I) 的提升，取最大作为该题的 R^+(i)
    - 每轮选择 R^+(i) 最大的题 i
    """
    def __init__(self, M: int, nsamples: int = 500, seed: Optional[int] = None):
        self.M = int(M)
        self.nsamples = int(nsamples)
        self.rng = np.random.default_rng(seed)

    def _estimate_I_vector(self, alpha: Sequence[float]) -> np.ndarray:
        alpha = np.asarray(alpha, dtype=float)
        if alpha.ndim != 1 or alpha.size != self.M:
            raise ValueError("alpha must be 1-D with length M")
        # Gamma 采样 + argmax 统计
        X = self.rng.gamma(shape=alpha, scale=1.0, size=(self.nsamples, self.M))
        argmax = np.argmax(X, axis=1)
        counts = np.bincount(argmax, minlength=self.M)
        probs = counts / self.nsamples
        return probs

    @staticmethod
    def _compute_h(I_vector: np.ndarray) -> float:
        # 目标函数：max_k I_k(alpha)
        return float(np.max(I_vector))

    def select_next(self, alpha_list: Sequence[Sequence[float]], c: float = 1.0) -> int:
        A = np.asarray(alpha_list, dtype=float)
        if A.ndim != 2 or A.shape[1] != self.M:
            raise ValueError("alpha_list must be shape (K, M)")
        K = A.shape[0]
        R_plus_vals = np.empty(K, dtype=float)
        for i in range(K):
            alpha_i = A[i]
            r_vals = np.empty(self.M, dtype=float)
            I_base = self._estimate_I_vector(alpha_i)
            h_base = self._compute_h(I_base)
            for m in range(self.M):
                alpha_plus = alpha_i.copy()
                alpha_plus[m] += c
                I_plus = self._estimate_I_vector(alpha_plus)
                h_plus = self._compute_h(I_plus)
                r_vals[m] = h_plus - h_base
            R_plus_vals[i] = np.max(r_vals)
        return int(np.argmax(R_plus_vals))

    @staticmethod
    def update(alpha_list: np.ndarray, i: int, y: int, c: float = 1.0) -> None:
        alpha_list[i, y] += c


# ----------------- main run -----------------

def run(preds_path: str,
        out_path: str = "preds_okg.jsonl",
        T: int = 1400,
        nsamples: int = 500,
        seed: Optional[int] = 2025,
        choices: str = "A,B,C,D,E,F,G,H,I,J",
        labels_path: Optional[str] = None,
        curve_csv: Optional[str] = None,
        eval_every: int = 1) -> Tuple[list, np.ndarray]:
    """
    运行 OKG，并可选记录 (预算t, acc) 曲线。
    - preds_path: 输入 preds.jsonl（每行 {"id", "question", "answers": [...] }）
    - out_path  : 输出 OKG 收集后的 jsonl（同格式，"answers" 为抽样轨迹，"final" 为投票结果）
    - T         : 总预算（步数）；OKG 每步抽取一次（一次预算）
    - nsamples  : 蒙特卡洛近似 I(alpha) 的采样次数
    - seed      : 随机种子（控制 OKG 与无放回抽样）
    - choices   : 选项集合，用逗号分隔（例如 "A,B,C,D" 或 "A,B,C,...,J"）
    - labels_path: 若提供，会在循环内按 eval_every 步评估一次准确率
    - curve_csv : 若提供，把 (t, acc) 记到 CSV
    - eval_every: 曲线评估步频（为 1 表示每步评估）
    """
    rng = np.random.default_rng(seed)

    # 1) 解析选项集合，并构造映射
    label_list = [s.strip() for s in choices.split(",") if s.strip()]
    if not label_list:
        raise ValueError("--choices 解析为空")
    answer_map = {lab: i for i, lab in enumerate(label_list)}
    rev_map = {i: lab for lab, i in answer_map.items()}
    M = len(label_list)

    # 2) 读取 preds，构建题目列表与答案池（映射为 int 类别）
    ordered_qids: List = []
    questions_map: Dict = {}
    answer_pools: Dict = {}
    skipped_unmapped = 0

    with open(preds_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            qid = item["id"]
            ordered_qids.append(qid)
            questions_map[qid] = item.get("question", "")

            pool = []
            for ans in item.get("answers", []):
                if ans in answer_map:
                    pool.append(answer_map[ans])
                else:
                    skipped_unmapped += 1
            answer_pools[qid] = pool

    K = len(ordered_qids)
    if K == 0:
        raise RuntimeError("No questions found in preds file")

    print(f"[load] K={K}, M={M}, skipped_unmapped_answers={skipped_unmapped}")

    # 3) 初始化 OKG 状态
    alphas = np.ones((K, M), dtype=float)
    alloc = OKGAllocator(M=M, nsamples=nsamples, seed=seed)
    used_indices = {qid: [] for qid in ordered_qids}     # 无放回索引
    collected_answers = {qid: [] for qid in ordered_qids}  # 累计抽样的"字母"答案轨迹
    exhausted_questions = set()  # 记录已用尽答案池的题目

    # 4) 可选读取 labels 用于在线评估曲线
    labels: Dict = {}
    if labels_path:
        with open(labels_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    labels[obj["id"]] = obj["label"]
        print(f"[labels] loaded {len(labels)} items")

    # 5) 主循环
    selections: List[int] = []
    curve: List[Tuple[int, float]] = []  # (t, acc)

    for t in range(T):
        # 如果所有题目都已用尽，停止采样
        if len(exhausted_questions) >= K:
            print(f"[t={t}] All questions exhausted, stopping early", flush=True)
            break

        # 选择题目（排除已用尽的题目）
        available_indices = [i for i in range(K) if ordered_qids[i] not in exhausted_questions]
        if not available_indices:
            print(f"[t={t}] No available questions, stopping early", flush=True)
            break

        # 从可用题目中选择
        temp_alphas = alphas[available_indices]
        temp_idx = alloc.select_next(temp_alphas, c=1.0)
        idx = available_indices[temp_idx]
        selections.append(int(idx))
        qid = ordered_qids[idx]

        pool = answer_pools.get(qid, [])
        if not pool:
            print(f"[t={t}] selected idx={idx} qid={qid} but empty pool, skip", flush=True)
            continue

        # 无放回：检查是否有可用答案
        available = [i for i in range(len(pool)) if i not in used_indices[qid]]
        if not available:
            # 该题已用尽，不再重置，标记为已用尽
            exhausted_questions.add(qid)
            print(f"[t={t}] qid={qid} exhausted, will not be selected again", flush=True)
            continue

        # 从可用索引中随机抽一个答案
        ans_idx = int(rng.choice(available))
        used_indices[qid].append(ans_idx)
        y_obs = int(pool[ans_idx])           # 数字类别
        sampled = rev_map[y_obs]             # 映射回“字母/自定义标签”
        collected_answers[qid].append(sampled)
        OKGAllocator.update(alphas, idx, y_obs, c=1.0)

        # 记录日志（可注释）
        print(f"[t={t}] selected idx={idx} qid={qid} sampled_ans={sampled} (ans_idx={ans_idx})", flush=True)

        # 在线评估：按 eval_every 步评估一次
        if labels and ((t + 1) % max(1, eval_every) == 0):
            correct = 0
            total = 0
            for q in ordered_qids:
                if q not in labels:
                    continue
                final = vote_majority(collected_answers[q])
                if final == labels[q]:
                    correct += 1
                total += 1
            acc = correct / max(1, total)
            curve.append((t + 1, acc))
            # 可选打印
            # print(f"[curve] t={t+1} acc={acc:.4f}")

    # 6) 输出曲线 CSV（若需要）
    if curve_csv and curve:
        with open(curve_csv, "w", encoding="utf-8") as f:
            f.write("budget_t,acc\n")
            for t_val, acc_val in curve:
                f.write(f"{t_val},{acc_val:.6f}\n")
        print(f"[OKG] curve saved: {curve_csv}")

    # 7) 写出 OKG 收集后的 jsonl
    with open(out_path, "w", encoding="utf-8") as f:
        for qid in ordered_qids:
            answers_list = collected_answers.get(qid, [])
            final = vote_majority(answers_list)
            obj = {
                "id": qid,
                "question": questions_map.get(qid, ""),
                "answers": answers_list,
                "final": final
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[OKG] Saved OKG outputs to: {out_path}")
    return selections, alphas


# ----------------- CLI -----------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="preds_pro.jsonl", help="输入 preds.jsonl")
    ap.add_argument("--out", default="preds_pro_okg.jsonl", help="输出 OKG 收集结果 jsonl")
    ap.add_argument("--T", type=int, default=7000, help="总预算（步数）")
    ap.add_argument("--nsamples", type=int, default=500, help="I(alpha) MC 采样次数")
    ap.add_argument("--seed", type=int, default=2025, help="随机种子")
    ap.add_argument("--choices", default="A,B,C,D,E,F,G,H,I,J",
                    help="选项集合，逗号分隔（如 A,B,C,D）")
    ap.add_argument("--labels", default=None,
                    help="labels.jsonl 路径；若提供则记录 acc-预算 曲线")
    ap.add_argument("--curve_csv", default=None,
                    help="把 (t, acc) 写入 CSV 文件")
    ap.add_argument("--eval_every", type=int, default=100,
                    help="每隔多少步评估一次曲线（默认每步）")
    args = ap.parse_args()

    run(preds_path=args.preds,
        out_path=args.out,
        T=args.T,
        nsamples=args.nsamples,
        seed=args.seed,
        choices=args.choices,
        labels_path=args.labels,
        curve_csv=args.curve_csv,
        eval_every=args.eval_every)
