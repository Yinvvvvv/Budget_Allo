#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llm_allocator.py
- fixedB 模式：每题固定采样/推理 B 次（你原有的逻辑）
- optkg  模式：全局自适应分配（Opt-KG 风格），逐步把下一条轨迹分配给边际收益最大的题

依赖：
  pip install transformers torch

示例：
  # 每题固定 5 次
  python llm_allocator.py --mode fixedB --questions questions.jsonl --out preds.jsonl --B 5

  # 自适应分配：总预算 20k 输出 token（近似），成本权重 1e-4
  python llm_allocator.py --mode optkg --questions questions.jsonl --out preds.jsonl \
      --G 20000 --lambda_cost 1e-4 --delta_effect 0.02
"""

import os, json, re, argparse, random
from collections import Counter
from typing import List, Tuple, Dict, Any

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessor,
    LogitsProcessorList
)

# -------------------------
# Regex & Helpers
# -------------------------

# 抽取 "Answer: X" 或者最后一个独立的 A-D
CHOICE = re.compile(r"\b([A-D])\b")
ANSWER_LINE = re.compile(r"^Answer:\s*([A-D])\s*$", re.IGNORECASE | re.MULTILINE)

def build_user_turn(q: str) -> str:
    """
    让模型先隐式推理，再输出最终选项。最后一行只给 'Answer: X'，方便抽取。
    """
    return (
        "You are a careful reasoner for single-choice questions (options A-D).\n"
        "Rules:\n"
        "1) Think step by step privately.\n"
        "2) Then output ONLY ONE final line exactly as: 'Answer: X' where X ∈ {A,B,C,D}.\n"
        "3) Do not print anything after that line.\n\n"
        f"Question: {q}\nOptions are labeled A-D.\n"
    )

def extract_answer(text: str, default: str = "") -> str:
    """
    优先从 'Answer: X' 提取，否则回退到末尾检索最后一个 A-D。
    """
    m = None
    for _m in ANSWER_LINE.finditer(text):
        m = _m
    if m:
        return m.group(1).upper()
    hits = list(CHOICE.finditer(text[-400:]))  # 只看尾部，噪声更小
    return hits[-1].group(1) if hits else default

def vote_majority(answers: List[str]) -> str:
    """
    多数投票。平票时选出现次数最高的第一个候选。
    """
    if not answers:
        return ""
    cnt = Counter(answers)
    return cnt.most_common(1)[0][0]

def score_from_answers(ans_list: List[str]) -> float:
    """
    多数投票胜率：max_count / n；空时返回 0.5。
    """
    if not ans_list:
        return 0.5
    cnt = Counter(ans_list)
    return cnt.most_common(1)[0][1] / len(ans_list)

# -------------------------
# Logits Sanitizer
# -------------------------

class SanitizeLogitsProcessor(LogitsProcessor):
    """将 logits 中的 NaN/Inf 替换为一个较大的负值，避免 torch.multinomial 出错。"""
    def __init__(self, replace_value: float = -1e4):
        self.replace_value = replace_value

    def __call__(self, input_ids, scores: torch.FloatTensor) -> torch.FloatTensor:
        if not torch.isfinite(scores).all():
            scores = torch.nan_to_num(
                scores,
                nan=self.replace_value,
                posinf=self.replace_value,
                neginf=self.replace_value
            )
        return scores

# -------------------------
# Model Loader
# -------------------------

def load_tokenizer_and_model(model_path: str):
    """
    尝试以多种模式加载 tokenizer 和模型，遇到 OOM 时回退：
    1) float16 + device_map=auto
    2) dtype="auto" + device_map=auto
    3) CPU-only (device_map={"": "cpu"}, low_cpu_mem_usage=True)
    返回 (tokenizer, model)
    """
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    last_exc = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        ).eval()
        print(f"[Load] float16 + device_map=auto: {model_path}")
        return tok, model
    except Exception as e:
        last_exc = e
        print("[Load] float16 failed → trying dtype=auto ...", flush=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype="auto",
            device_map="auto"
        ).eval()
        print(f"[Load] dtype=auto + device_map=auto: {model_path}")
        return tok, model
    except Exception as e:
        last_exc = e
        print("[Load] dtype=auto failed → trying CPU fallback ...", flush=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map={"": "cpu"},
            low_cpu_mem_usage=True
        ).eval()
        print(f"[Load] CPU-only (slow): {model_path}")
        return tok, model
    except Exception as e:
        raise RuntimeError(
            "Failed to load model in multiple fallback modes.\n"
            f"Last error: {e}\nPrevious error: {last_exc}"
        )

# -------------------------
# Generation Helpers
# -------------------------

def make_inputs(tok, model, qtext: str):
    messages = [
        {"role":"system","content":"You are a helpful assistant."},
        {"role":"user","content": build_user_turn(qtext)}
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok([prompt], return_tensors="pt").to(model.device)
    return inputs

def generate_one(
    tok,
    model,
    inputs: Dict[str, torch.Tensor],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    logits_processors: LogitsProcessorList
) -> Tuple[str, int]:
    gen = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.eos_token_id,
        logits_processor=logits_processors
    )
    offset = inputs["input_ids"].shape[-1]
    out_ids = gen[0]
    text = tok.decode(out_ids[offset:], skip_special_tokens=True)
    ans = extract_answer(text, default="")
    token_cost = int(out_ids.shape[-1] - offset)  # 输出 token 近似
    return ans, token_cost

# -------------------------
# Mode: fixedB (你的原逻辑)
# -------------------------

def run_fixedB_mode(args, tok, model):
    logits_processors = LogitsProcessorList([SanitizeLogitsProcessor()])

    # 读取题库
    questions = []
    with open(args.questions, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    results = []
    for item in questions:
        qid, qtext = item["id"], item["question"]

        inputs = make_inputs(tok, model, qtext)
        answers = []

        if args.batch_return:
            # 一次返回 B 条
            gen = model.generate(
                **inputs,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                num_return_sequences=args.B,
                pad_token_id=tok.eos_token_id,
                logits_processor=logits_processors
            )
            offset = inputs["input_ids"].shape[-1]
            for i in range(args.B):
                text = tok.decode(gen[i][offset:], skip_special_tokens=True)
                ans = extract_answer(text, default="")
                answers.append(ans)
        else:
            # 循环生成 B 次
            for _ in range(args.B):
                ans, _tok_cost = generate_one(
                    tok, model, inputs,
                    args.max_new_tokens, args.temperature, args.top_p,
                    logits_processors
                )
                answers.append(ans)

        final = vote_majority(answers)
        results.append({"id": qid, "question": qtext, "answers": answers, "final": final})
        print(f"[fixedB Q{qid}] vote={final} | answers={answers}")

    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("[fixedB] Saved:", args.out)

# -------------------------
# Mode: optkg （全局自适应分配）
# -------------------------

def optkg_allocate_once(tok, model, qtext, temperature, top_p, max_new_tokens, logits_processors):
    """对单个题采样 1 条，返回 (new_answer, token_cost)"""
    inputs = make_inputs(tok, model, qtext)
    ans, token_cost = generate_one(
        tok, model, inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        logits_processors=logits_processors
    )
    return ans, token_cost

def run_optkg_mode(args, tok, model):
    # 读题
    questions = []
    with open(args.questions, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    K = len(questions)

    # 每题状态
    answers: List[List[str]] = [[] for _ in range(K)]
    scores:  List[float]     = [0.5 for _ in range(K)]
    # Beta 后验：有效概率 p_i
    A = [1.0] * K
    B = [1.0] * K
    # 经验提升 EMA
    lift_ema = [0.03] * K

    spent = 0
    logits_processors = LogitsProcessorList([SanitizeLogitsProcessor()])

    # 主循环（直到预算用完或全体边际收益≤0）
    while spent < args.G:
        # 1) 计算每题的期望增益：p * lift - lambda * est_cost
        best_i, best_gain = -1, -1e18
        est_cost = args.max_new_tokens  # 也可替换为该题历史 token_cost 均值
        for i in range(K):
            p_hat = A[i] / (A[i] + B[i])
            gain = p_hat * lift_ema[i] - args.lambda_cost * est_cost
            if gain > best_gain:
                best_gain, best_i = gain, i

        if best_gain <= 0:
            print(f"[optkg STOP] All marginal gains <= 0 (best={best_gain:.6f}).")
            break

        # 2) 给该题再采 1 条
        qtext = questions[best_i]["question"]
        old_score = scores[best_i]
        new_ans, token_cost = optkg_allocate_once(
            tok, model, qtext,
            args.temperature, args.top_p, args.max_new_tokens,
            logits_processors
        )
        spent += token_cost
        answers[best_i].append(new_ans)
        scores[best_i] = score_from_answers(answers[best_i])

        # 3) 更新有效性 y 与 Beta 后验、EMA
        y = 1 if (scores[best_i] - old_score) >= args.delta_effect else 0
        if y == 1:
            A[best_i] += 1.0
        else:
            B[best_i] += 1.0
        lift_obs = max(0.0, scores[best_i] - old_score)
        lift_ema[best_i] = 0.8 * lift_ema[best_i] + 0.2 * lift_obs

        if spent >= args.G:
            break

    # 导出结果
    results = []
    for idx, item in enumerate(questions):
        final = vote_majority(answers[idx])
        results.append({
            "id": item["id"],
            "question": item["question"],
            "answers": answers[idx],
            "final": final
        })
        print(f"[optkg Q{item['id']}] vote={final} | n={len(answers[idx])} | score={scores[idx]:.3f}")

    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("[optkg] Saved:", args.out, "| spent_tokens:", spent)

# -------------------------
# Entrypoint
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=f"/home/{os.getenv('USER')}/models/Qwen2.5-7B-Instruct",
                    help="本地模型目录或 HF 名称")
    ap.add_argument("--questions", default="questions.jsonl", help="输入题库（jsonl，每行含 id, question）")
    ap.add_argument("--out", default="preds.jsonl", help="输出预测文件（jsonl）")

    # generation
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)

    # reproducibility
    ap.add_argument("--seed", type=int, default=42)

    # fixedB mode
    ap.add_argument("--mode", choices=["fixedB", "optkg"], default="fixedB",
                    help="fixedB=每题 B 次；optkg=全局自适应分配")
    ap.add_argument("--B", type=int, default=5, help="fixedB 模式：每题采样/推理次数")
    ap.add_argument("--batch_return", action="store_true",
                    help="fixedB 模式：一次生成 B 条(num_return_sequences=B)，更快")

    # optkg mode
    ap.add_argument("--G", type=int, default=20000,
                    help="optkg 模式：总预算（近似以新增输出 tokens 计）")
    ap.add_argument("--lambda_cost", type=float, default=1e-4,
                    help="optkg 模式：成本权重（越大越节省 token）")
    ap.add_argument("--delta_effect", type=float, default=0.02,
                    help="optkg 模式：判定一次采样是否“有效”的最小分数提升")

    args = ap.parse_args()

    # 可重复性
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    # 加载模型
    tok, model = load_tokenizer_and_model(args.model_path)

    # 分支运行
    if args.mode == "optkg":
        run_optkg_mode(args, tok, model)
    else:
        run_fixedB_mode(args, tok, model)

if __name__ == "__main__":
    main()
