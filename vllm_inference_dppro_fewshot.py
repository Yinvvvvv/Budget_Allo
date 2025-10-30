#!/usr/bin/env python3
"""
vLLM MMLU Inference using Remote vLLM Service (+ optional Few-shot CoT)
"""

import os, json, re, argparse, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

# ---------------- deps ----------------
try:
    from openai import OpenAI
except ImportError:
    print("❌ Error: OpenAI Python client is not installed.")
    print("   Please install it with: pip install openai")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("❌ Error: tqdm library is not installed.")
    print("   Please install it with: pip install tqdm")
    sys.exit(1)

# === NEW === few-shot 需要
try:
    from datasets import load_dataset
except ImportError:
    print("❌ Error: datasets library is not installed.")
    print("   Please install it with: pip install datasets")
    sys.exit(1)

try:
    import transformers
except ImportError:
    print("❌ Error: transformers library is not installed.")
    print("   Please install it with: pip install transformers")
    sys.exit(1)

# ---------------- regex ----------------
BOXED_CHOICE = re.compile(
    r"""
    \\boxed\s*\{\s*
        (?:\\text\s*\{\s*([A-J])\s*\}   # \boxed{\text{C}}
        |([A-J]))                        # \boxed{C}
    \s*\}
    """,
    re.VERBOSE | re.IGNORECASE
)

# === NEW === few-shot 全局（用最小改动方案：在 main 里赋值）
USE_FEWSHOT = False
NTRAIN = 5
SUBJECT_KEY = "subject"
INITIAL_PROMPT_TEXT = ""
VAL_BY_SUBJECT = {}
MAX_MODEL_LEN = 4096  # 仅用于粗略长度预算
TOKENIZER = None

# ---------------- prompt builders ----------------
def build_user_turn(q: str) -> str:
    """原有零样本提示"""
    return (
        f"The following are multiple choice questions (with answers).\n\n"
        f"Question: {q}.\n\n"
        "Please reason step-by-step.\n"
        "At the very end, output exactly one line in this format:\n"
        "The final answer is \\boxed{X}\n"
        "Where X is the single option letter from A to J only. "
        "Do not output the option text or any numeric value; only put the letter inside the box. "
        "Do not add punctuation or extra text after the box."
    )

# === NEW === few-shot 辅助
DEFAULT_INITIAL_PROMPT = (
    "You are a careful reasoner for multiple-choice questions in {$}.\n"
    "Reason step-by-step, and at the very end output exactly one line:\n"
    "The final answer is \\boxed{X}\n"
    "where X is the option letter only.\n"
)

def load_initial_prompt(path: str | None) -> str:
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return DEFAULT_INITIAL_PROMPT

def preprocess_mmlu_val(ds_split):
    """移除 N/A 选项"""
    out = []
    for e in ds_split:
        opts = [o for o in e.get("options", []) if o and o != "N/A"]
        if len(opts) >= 2:
            ee = dict(e)
            ee["options"] = opts
            out.append(ee)
    return out

def format_cot_example(example, including_answer=True):
    choices = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    prompt = "Question:\n"
    prompt += example["question"] + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(example.get("options", [])):
        prompt += f"{choices[i]}. {opt}\n"
    if including_answer:
        cot = example.get("cot_content", "").replace(
            "A: Let's think step by step.", "Answer: Let's think step by step."
        )
        prompt += cot + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    return prompt

def token_len(text: str) -> int:
    """CPU 上估 token 数（粗略）。"""
    if TOKENIZER is None:
        return len(text.split())  # 极端兜底
    return len(TOKENIZER(text, add_special_tokens=False)["input_ids"])

def build_fewshot_user_turn(q: str, subject: str, args) -> str:
    """
    构造按学科 few-shot 的 user prompt，并做长度安全回退（减少 k）。
    """
    pool = VAL_BY_SUBJECT.get(subject) or next(iter(VAL_BY_SUBJECT.values()), [])
    init = (INITIAL_PROMPT_TEXT or DEFAULT_INITIAL_PROMPT).replace("{$}", subject or "general")

    if not pool:
        # 没有可用示例，退化到 zero-shot 风格（保持末行规范）
        return (
            init + "\n"
            "Question:\n" + q + "\n"
            "Answer: Let's think step by step."
        )

    k = max(NTRAIN, 0)
    while True:
        prompt = init + "\n"
        for ex in pool[:k]:
            prompt += format_cot_example(ex, including_answer=True)
        # 当前题（无答案）
        curr_stub = {"question": q, "options": [], "cot_content": ""}
        prompt += format_cot_example(curr_stub, including_answer=False)

        if token_len(prompt) < (MAX_MODEL_LEN - args.max_new_tokens) or k == 0:
            return prompt
        k -= 1

# ---------------- extraction & voting ----------------
def extract_answer(text: str, default: str = "") -> str:
    last = None
    for m in BOXED_CHOICE.finditer(text):
        last = m
    if not last:
        return default
    for g in last.groups():
        if g:
            return g.upper()
    return default

def vote_majority(answers: list[str]) -> str:
    if not answers:
        return ""
    cnt = Counter(answers)
    return cnt.most_common(1)[0][0]

# ---------------- single question ----------------
def process_single_question(client, model_to_use, item, args):
    qid, qtext = item["id"], item["question"]
    subj = item.get(SUBJECT_KEY, "")

    # === 修改点极少：按开关选择用哪种 prompt ===
    if USE_FEWSHOT:
        user_content = build_fewshot_user_turn(qtext, subj, args)
    else:
        user_content = build_user_turn(qtext)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_content}
    ]

    answers = []
    try:
        if args.batch_return:
            resp = client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                n=args.B,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_new_tokens,
            )
            for choice in resp.choices:
                text = choice.message.content or ""
                answers.append(extract_answer(text, default=""))
        else:
            for _ in range(args.B):
                resp = client.chat.completions.create(
                    model=model_to_use,
                    messages=messages,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_new_tokens,
                )
                text = resp.choices[0].message.content or ""
                answers.append(extract_answer(text, default=""))

        final = vote_majority(answers)
        return {
            "id": qid,
            "question": qtext,
            "subject": subj,
            "answers": answers,
            "final": final,
            "success": True
        }
    except Exception as e:
        return {
            "id": qid,
            "question": qtext,
            "subject": subj,
            "answers": [],
            "final": "",
            "error": str(e),
            "success": False
        }

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(
        description='vLLM MMLU Inference using Remote vLLM Service',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # server
    ap.add_argument("--host", default="localhost", help="vLLM server host")
    ap.add_argument("--port", type=int, default=8000, help="vLLM server port")
    ap.add_argument("--model_name", "-m", default=None,
                    help="Model name served by vLLM; if None, use the first available")

    # data IO
    ap.add_argument("--questions", default="questions.jsonl",
                    help="jsonl with fields: id, question, and optionally subject/category")
    ap.add_argument("--out", default="preds.jsonl", help="output jsonl")

    # sampling
    ap.add_argument("--B", type=int, default=5, help="samples per question")
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--batch_return", action="store_true",
                    help="use n=B in one request")

    # concurrency
    ap.add_argument("--max_workers", type=int, default=None,
                    help="max concurrent requests (default: auto)")

    # === NEW === few-shot
    ap.add_argument("--use_fewshot", action="store_true", help="enable few-shot CoT by subject")
    ap.add_argument("--ntrain", type=int, default=5, help="k exemplars per subject")
    ap.add_argument("--subject_key", type=str, default="subject",
                    help="field name in questions jsonl for subject/category")
    ap.add_argument("--initial_prompt", type=str, default="cot_prompt_lib/initial_prompt.txt",
                    help="template file; default template if missing")
    ap.add_argument("--validation_set", type=str, default=None,
                    help="validation set jsonl file for few-shot examples; if None, use MMLU-Pro validation split")

    global USE_FEWSHOT, NTRAIN, SUBJECT_KEY, INITIAL_PROMPT_TEXT, VAL_BY_SUBJECT, TOKENIZER

    args = ap.parse_args()
    USE_FEWSHOT = args.use_fewshot
    NTRAIN = args.ntrain
    SUBJECT_KEY = args.subject_key

    print("=" * 80)
    print("vLLM MMLU INFERENCE USING REMOTE SERVICE")
    print("=" * 80)
    print(f" Server: http://{args.host}:{args.port}")
    print(f" Model:  {args.model_name or '(auto)'}")
    print(f" Few-shot: {USE_FEWSHOT} (k={NTRAIN}, subject_key='{SUBJECT_KEY}')")
    print(f" Initial prompt: {args.initial_prompt}")
    print(f" B={args.B}, temp={args.temperature}, top_p={args.top_p}, max_new_tokens={args.max_new_tokens}")
    print(f" Workers: {args.max_workers or 'Auto'}")
    print("=" * 80 + "\n")

    # client
    client = OpenAI(api_key="EMPTY", base_url=f"http://{args.host}:{args.port}/v1")
    try:
        models = client.models.list()
        available_models = [m.id for m in models.data]
        if not available_models:
            print("❌ Error: No models available from the server")
            sys.exit(1)
        if args.model_name and args.model_name not in available_models:
            print(f"⚠️  Specified model '{args.model_name}' not found in {available_models}; still trying it.")
            model_to_use = args.model_name
        else:
            model_to_use = args.model_name or available_models[0]
        print(f"✅ Using model: {model_to_use}\n")
    except Exception as e:
        print(f"❌ Error connecting to vLLM service: {e}")
        sys.exit(1)

    # load questions
    print(f"📖 Loading questions from {args.questions}...")
    questions = []
    with open(args.questions, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    print(f"✅ Loaded {len(questions)} questions\n")

    # === NEW === few-shot 资源准备（仅在开启时）
    if USE_FEWSHOT:
        if args.validation_set:
            print(f"📚 Loading validation set from {args.validation_set} for few-shot exemplars ...")
            val = []
            with open(args.validation_set, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        val.append(json.loads(line))
        else:
            print("📚 Loading MMLU-Pro validation split for few-shot exemplars ...")
            ds = load_dataset("TIGER-Lab/MMLU-Pro")
            val = preprocess_mmlu_val(ds["validation"])
        
        group = defaultdict(list)
        for ex in val:
            subject = ex.get(SUBJECT_KEY, "").strip()
            group[subject].append(ex)
        VAL_BY_SUBJECT = dict(group)
        print(f"   - {len(val)} exemplars across {len(VAL_BY_SUBJECT)} subjects.")

        # tokenizer 用于长度预算（可替换成与服务端相同的 HF 名称）
        try:
            TOKENIZER = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=True)
        except Exception:
            TOKENIZER = transformers.AutoTokenizer.from_pretrained("gpt2", use_fast=True)
        if TOKENIZER.pad_token is None and TOKENIZER.eos_token is not None:
            TOKENIZER.pad_token = TOKENIZER.eos_token
        TOKENIZER.padding_side = "left"

        INITIAL_PROMPT_TEXT = load_initial_prompt(args.initial_prompt)

    # run parallel
    print("🔮 Running inference with concurrent requests ...\n")
    results = []
    start_time = time.time()

    pbar = tqdm(total=len(questions),
                desc="Processing",
                unit="q",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    workers = args.max_workers or min(32, (os.cpu_count() or 8) * 5)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        def process_wrapper(item):
            return process_single_question(client, model_to_use, item, args)
        for res in executor.map(process_wrapper, questions):
            results.append(res)
            if res["success"]:
                pbar.set_postfix({"Q": res["id"], "vote": res["final"]})
            else:
                pbar.set_postfix({"Q": res["id"], "error": "✗"})
            pbar.update(1)
    pbar.close()

    # stats
    elapsed_time = time.time() - start_time
    qps = len(questions)/elapsed_time if elapsed_time > 0 else 0.0
    success_count = sum(1 for r in results if r.get("success", False))

    print("\n" + "="*80)
    print("✅ Inference completed!")
    print(f"   - Total questions: {len(results)}")
    print(f"   - Successful: {success_count}")
    print(f"   - Failed: {len(results) - success_count}")
    print(f"   - Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"   - Throughput: {qps:.2f} questions/sec")
    print("="*80 + "\n")

    # save
    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✅ Results saved to: {args.out}")

if __name__ == "__main__":
    main()
