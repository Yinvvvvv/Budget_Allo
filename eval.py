# eval.py
import json, argparse

def load_kv(path, key="id", val="final", normalize=True):
    d = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): 
                continue
            j = json.loads(line)
            v = j[val]
            if normalize and isinstance(v, str):
                v = v.strip().upper()
            d[j[key]] = v
    return d

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="preds.jsonl",
                    help="推理输出（含 id, final）")
    ap.add_argument("--label", default="labels.jsonl",
                    help="标准答案（含 id, label）")
    args = ap.parse_args()

    pred = load_kv(args.pred,  val="final")
    gold = load_kv(args.label, val="label")

    ids = set(pred) & set(gold)
    if not ids:
        raise SystemExit("No overlapping ids between preds and labels.")

    correct = sum(1 for i in ids if pred[i] == gold[i])
    total = len(ids)
    print(f"Accuracy: {correct}/{total} = {correct/total:.2%}")

    # 诊断信息
    miss_pred = len(set(gold) - set(pred))
    miss_gold = len(set(pred) - set(gold))
    if miss_pred or miss_gold:
        print(f"(Info) unmatched ids -> missing_pred:{miss_pred}, missing_gold:{miss_gold}")
