import os, json, re, argparse, random
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList

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

def vote_majority(answers: list[str]) -> str:
    """
    多数投票。平票时选第一个出现次数最高的候选（也可改成自定义 tie-break）。
    """
    if not answers:
        return ""
    cnt = Counter(answers)
    return cnt.most_common(1)[0][0]


class SanitizeLogitsProcessor(LogitsProcessor):
    """LogitsProcessor to replace inf/nan logits with a large negative finite value.

    This helps avoid invalid probabilities (nan/inf) which cause torch.multinomial to fail.
    """
    def __init__(self, replace_value: float = -1e4):
        self.replace_value = replace_value

    def __call__(self, input_ids, scores: torch.FloatTensor) -> torch.FloatTensor:
        # scores shape: (batch_size, vocab_size)
        if not torch.isfinite(scores).all():
            # Replace NaN/inf with a large negative finite value to make softmax stable
            scores = torch.nan_to_num(scores, nan=self.replace_value, posinf=self.replace_value, neginf=self.replace_value)
        return scores

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=f"/home/{os.getenv('USER')}/models/Qwen2.5-7B-Instruct",
                    help="本地模型目录或 HF 名称")
    ap.add_argument("--questions", default="questions.jsonl", help="输入题库（jsonl，含 id, question）")
    ap.add_argument("--out", default="preds.jsonl", help="输出预测文件（jsonl）")
    ap.add_argument("--B", type=int, default=5, help="每题采样/推理次数")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_return", action="store_true",
                    help="一次生成 B 条(num_return_sequences=B)，更快")
    args = ap.parse_args()

    # 可重复性
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    # 加载 tokenizer 与模型（支持 OOM 回退）
    def load_tokenizer_and_model(model_path: str):
        """
        尝试以多种模式加载 tokenizer 和模型，遇到 OOM 时按顺序回退：
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
            print(f"Loaded model with float16 and device_map=auto: {model_path}")
            return tok, model
        except Exception as e:
            last_exc = e
            print("Float16 load failed (trying dtype=auto)...", flush=True)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype="auto",
                device_map="auto"
            ).eval()
            print(f"Loaded model with dtype=auto and device_map=auto: {model_path}")
            return tok, model
        except Exception as e:
            last_exc = e
            print("Model load failed with dtype=auto (trying CPU fallback)...", flush=True)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map={"": "cpu"},
                low_cpu_mem_usage=True
            ).eval()
            print(f"Loaded model on CPU (very slow): {model_path}")
            return tok, model
        except Exception as e:
            raise RuntimeError(
                "Failed to load model in multiple fallback modes.\n"
                f"Last error: {e}\nPrevious error: {last_exc}"
            )

    tok, model = load_tokenizer_and_model(args.model_path)

    # set up logits sanitizer to avoid nan/inf probs during sampling
    logits_processors = LogitsProcessorList()
    logits_processors.append(SanitizeLogitsProcessor())

    # 读取题库（每行 {"id": int/str, "question": "...A) ... B) ... C) ... D) ..."}）
    questions = []
    with open(args.questions, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    results = []
    for item in questions:
        qid, qtext = item["id"], item["question"]

        # 构造 chat prompt
        messages = [
            {"role":"system","content":"You are a helpful assistant."},
            {"role":"user","content": build_user_turn(qtext)}
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok([prompt], return_tensors="pt").to(model.device)

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
                gen = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tok.eos_token_id,
                    logits_processor=logits_processors
                )
                text = tok.decode(gen[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
                ans = extract_answer(text, default="")
                answers.append(ans)

        final = vote_majority(answers)
        results.append({"id": qid, "question": qtext, "answers": answers, "final": final})
        print(f"[Q{qid}] vote={final} | answers={answers}")

    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("Saved:", args.out)

if __name__ == "__main__":
    main()

