#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
准备 MMLU-Pro 数据集（兼容不同数量选项）
Usage:
    # 默认验证集；不填充，按实际选项数导出
    python prepare_mmlu_pro.py --split validation

    # 指定科目；最多导出 200 道
    python prepare_mmlu_pro.py --split test --subjects mathematics physics --max_examples 200

    # 若下游强依赖 A–J，可逻辑上将标签空间补到 10（题干中不写空项）
    python prepare_mmlu_pro.py --split test --pad_to 10 \
        --out_questions questions_pro_test.jsonl --out_labels labels_pro_test.jsonl

    python prepare_mmlu_pro.py --split test 
        --out_questions questions_try.jsonl --out_labels labels_try.jsonl --max_examples 200
"""

from datasets import load_dataset
import json, argparse, os, random, sys

LETTERS = [chr(ord('A') + i) for i in range(26)]  # A-Z

def build_option_labels(n: int, pad_to: int = 0):
    """
    返回 (labels_for_text, labels_for_space) 两组标签：
    - labels_for_text: 只用于题干展示，长度 = 实际选项数
    - labels_for_space: 用于下游标签空间（若 pad_to>0，会扩到 pad_to）
    """
    if n > 26:
        return None, None
    labels_actual = LETTERS[:n]
    if pad_to and pad_to > 0:
        if pad_to > 26:
            pad_to = 26
        labels_space = LETTERS[:max(n, pad_to)]
    else:
        labels_space = labels_actual
    return labels_actual, labels_space

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation", help="validation 或 test")
    ap.add_argument("--subjects", nargs="*", default=None, help="科目子集；默认全部")
    ap.add_argument("--max_examples", type=int, default=0, help="最多导出多少道题；0 表示全部")
    ap.add_argument("--pad_to", type=int, default=0, help="将标签空间补到指定个数（仅标签空间，题干不写空项）；0 表示不补")
    ap.add_argument("--out_questions", default="questions_pro.jsonl")
    ap.add_argument("--out_labels", default="labels_pro.jsonl")
    args = ap.parse_args()

    # 载入 MMLU-Pro
    try:
        ds_all = load_dataset("TIGER-Lab/MMLU-Pro")
        if args.split not in ds_all:
            raise ValueError(f"Split {args.split} 不存在，包含这些：{list(ds_all.keys())}")
    except Exception as e:
        print(f"加载数据集失败: {e}")
        print("提示：可能需要先运行: huggingface-cli login")
        sys.exit(1)

    ds = ds_all[args.split]

    # 科目筛选
    if args.subjects:
        subjects = set(args.subjects)
        ds = ds.filter(lambda x: x.get("subject") in subjects)

    # 限制样本数量
    if args.max_examples and args.max_examples > 0:
        ds = ds.select(range(min(args.max_examples, len(ds))))

    # 输出文件
    with open(args.out_questions, "w", encoding="utf-8") as qf, \
         open(args.out_labels, "w", encoding="utf-8") as lf:

        id_counter = 1
        kept = 0
        skipped = 0

        for row in ds:
            try:
                q = (row.get("question") or "").strip()
                options = row.get("options", [])
                ans_idx = row.get("answer_index", None)

                # 基本校验
                if not q or not isinstance(options, list) or len(options) == 0:
                    print(f"警告：题目 {id_counter} 内容缺失或无选项，跳过")
                    skipped += 1
                    id_counter += 1
                    continue

                # 去除 None，并转为字符串
                options = [("" if x is None else str(x)).strip() for x in options]
                num_actual = len(options)

                # 生成标签
                labels_for_text, labels_for_space = build_option_labels(num_actual, args.pad_to)
                if labels_for_text is None:
                    print(f"警告：题目 {id_counter} 选项数量 > 26（实际：{len(options)}），跳过")
                    skipped += 1
                    id_counter += 1
                    continue

                # 对于validation set，使用简化的few-shot格式
                if args.split == "validation":
                    # 构造题干：只展示"实际存在"的选项
                    pieces = [q]
                    for lab, opt in zip(labels_for_text, options):
                        pieces.append(f"{lab}) {opt}")
                    question_text = " ".join(pieces)
                    
                    qline = {
                        "id": id_counter,
                        "question": q,
                        "options": options,
                        "cot_content": row.get("cot_content", ""),
                        "subject": row.get("category", None)
                    }
                else:
                    # 构造题干：只展示"实际存在"的选项
                    pieces = [q]
                    for lab, opt in zip(labels_for_text, options):
                        pieces.append(f"{lab}) {opt}")
                    question_text = " ".join(pieces)

                    # 写出问题（同时提供结构化 choices，方便下游解析）
                    qline = {
                        "id": id_counter,
                        "question": question_text,
                        "choices": [{"label": lab, "text": options[i]} for i, lab in enumerate(labels_for_text)],
                        "num_options": num_actual,
                        "label_space": labels_for_space,  # 下游可据此确定允许的标签集合
                        "subject": row.get("category", None)
                    }
                qf.write(json.dumps(qline, ensure_ascii=False) + "\n")

                # 写出答案
                if ans_idx is not None:
                    # 答案索引必须落在“实际选项”范围内
                    if not (0 <= ans_idx < num_actual):
                        print(f"警告：题目 {id_counter} 的 answer_index 越界（ans_idx={ans_idx}, 选项数={num_actual}），跳过该题")
                        skipped += 1
                        id_counter += 1
                        continue
                    gold_letter = LETTERS[ans_idx]  # 与 labels_for_text 对齐：第 i 个 -> 第 i 个字母
                    lf.write(json.dumps({"id": id_counter, "label": gold_letter}, ensure_ascii=False) + "\n")

                kept += 1
                id_counter += 1

            except Exception as e:
                print(f"处理第 {id_counter} 题时出错: {e}")
                skipped += 1
                id_counter += 1
                continue

    total = id_counter - 1
    print("处理完成：")
    print(f"- 题目文件：{args.out_questions}")
    print(f"- 答案文件：{args.out_labels}")
    print(f"- 总题数（遍历）：{total}")
    print(f"- 成功写出：{kept}")
    print(f"- 跳过：{skipped}")
    if args.pad_to and args.pad_to > 0:
        print(f"- 标签空间已对齐至：{min(args.pad_to, 26)}（题干只展示实际选项）")

if __name__ == "__main__":
    main()
