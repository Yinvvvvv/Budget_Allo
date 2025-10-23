from datasets import load_dataset
import json, argparse, os, random



SUBJECTS_ALL = None  # 或者用一个列表限定科目，例如 ["abstract_algebra","astronomy"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation", help="validation 或 test（一般用 validation 调参）")
    ap.add_argument("--subjects", nargs="*", default=None, help="科目子集；默认全部")
    ap.add_argument("--max_examples", type=int, default=0, help="最多导出多少道题；0 表示全部")
    ap.add_argument("--out_questions", default="questions.jsonl")
    ap.add_argument("--out_labels", default="labels.jsonl")
    args = ap.parse_args()

    # 载入整个 MMLU（多科目）
    ds_all = load_dataset("cais/mmlu","all")
    if args.split not in ds_all:
        raise ValueError(f"Split {args.split} 不存在，包含这些：{list(ds_all.keys())}")

    # MMLU 的每个科目在一个二级 split 里，datasets 会把它们组合成一个 DatasetDict
    # Hugging Face 的这个版本把所有科目拼在一起了（有 subject 字段）
    ds = ds_all[args.split]

    # 科目筛选
    subjects = set(args.subjects) if args.subjects else None
    if subjects:
        ds = ds.filter(lambda x: x.get("subject") in subjects)

    # （可选）限制数量
    if args.max_examples and args.max_examples > 0:
        ds = ds.select(range(min(args.max_examples, len(ds))))

    # 生成 questions.jsonl / labels.jsonl
    # 字段假设：question: str, choices: list[str] 长度为4, answer: int(0..3), subject: str
    # 一些版本字段名可能略不同，如果出错把样本 print 出来看看 keys
    qf = open(args.out_questions, "w", encoding="utf-8")
    lf = open(args.out_labels, "w", encoding="utf-8")

    id_counter = 1
    for row in ds:
        q = row["question"]
        choices = row["choices"]  # ['optA','optB','optC','optD']
        ans_idx = row.get("answer", None)  # 可能有版本没有 test 答案
        # 构造我们脚本的 question 文本（带 A-D 选项）
        question_text = (
            q.strip()
            + " "
            + " ".join([f"{chr(65+i)}) {choices[i]}" for i in range(len(choices))])
        )

        qline = {"id": id_counter, "question": question_text, "subject": row.get("subject","")}
        qf.write(json.dumps(qline, ensure_ascii=False) + "\n")

        if ans_idx is not None:
            gold_letter = "ABCD"[ans_idx]
            lf.write(json.dumps({"id": id_counter, "label": gold_letter}, ensure_ascii=False) + "\n")

        id_counter += 1

    qf.close(); lf.close()
    print(f"Saved {args.out_questions} and {args.out_labels} (若 test split 无答案，labels 可能为空)")

if __name__ == "__main__":
    main()
