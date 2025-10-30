#!/usr/bin/env python3
"""
curve_filtered_from_preds.py
从 preds_pro.jsonl（每题一串候选答案）与 labels.jsonl 生成 filtered 曲线：
x 轴 = 总预算 budget = K * b   (K=题目数, b=每题使用的前 b 个答案)
y 轴 = accuracy(b)
输出：CSV（--out_csv）和 npz（--out_npz，可选）
"""

import json, argparse, math, numpy as np
from collections import Counter, defaultdict

def vote_majority(answers):
    if not answers: return ""
    cnt = Counter(answers)
    maxc = max(cnt.values())
    tied = {a for a,c in cnt.items() if c==maxc}
    firstpos = {}
    for i,a in enumerate(answers):
        if a in tied and a not in firstpos: firstpos[a] = i
    return min(tied, key=lambda a: firstpos[a])

def load_labels(path):
    y = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                y[obj["id"]] = obj["label"]
    return y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="preds_pro.jsonl")
    ap.add_argument("--labels", default="labels.jsonl")
    ap.add_argument("--choices", default="A,B,C,D,E,F,G,H,I,J")
    ap.add_argument("--out_csv", default="curve_filtered.csv")
    ap.add_argument("--out_npz", default=None)
    args = ap.parse_args()

    labels = load_labels(args.labels)
    order = []
    pools = {}
    with open(args.preds, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            qid = obj["id"]
            order.append(qid)
            pools[qid] = obj.get("answers", [])

    K = len(order)
    assert K>0, "no questions in preds"
    # 最大可用 B（按所有题的最短长度来保证每题都有 b 个）
    max_B = min(len(pools[qid]) for qid in order if qid in labels)
    xs_budget = []
    ys_acc = []

    for b in range(1, max_B+1):
        correct = 0; total = 0
        for qid in order:
            if qid not in labels: continue
            ans = pools[qid][:b]
            final = vote_majority(ans)
            if final == labels[qid]:
                correct += 1
            total += 1
        acc = correct / max(1,total)
        budget = K * b
        xs_budget.append(budget)
        ys_acc.append(acc)
        print(f"[filtered] b={b} budget={budget} acc={acc:.4f}")

    # 写 CSV
    with open(args.out_csv, "w", encoding="utf-8") as f:
        f.write("budget,b_per_q,acc\n")
        for b,(bud,acc) in enumerate(zip(xs_budget, ys_acc), start=1):
            f.write(f"{bud},{b},{acc:.6f}\n")

    if args.out_npz:
        np.savez(args.out_npz, budget=np.array(xs_budget), acc=np.array(ys_acc))

    print(f"Saved: {args.out_csv}")

if __name__ == "__main__":
    main()
