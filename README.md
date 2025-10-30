# Data Preprocess for MMLU_Pro

python prepare_mmlu_pro.py --split test --out_questions test_questions.jsonl --out_labels test_labels.jsonl --max_examples 200

python prepare_mmlu_pro.py --split validation --out_questions val_questions.jsonl --out_labels val_labels.jsonl

# Load vLLM

bash rank0.sh

# Inference

python vllm_inference_dppro_fewshot.py --use_fewshot --validation_set val_questions.jsonl --questions test_questions.jsonl --out fewshot_preds.jsonl --ntrain 1 --B 100

# Plot and Evaluation

python curve_okg_from_preds.py --preds fewshot_preds.jsonl --out fewshot_preds_okg.jsonl --T 10000 --curve_csv curve_okg.csv --labels test_labels.jsonl

python curve_filtered_from_preds.py --preds fewshot_preds.jsonl --labels test_labels.jsonl --out_csv curve_filtered.csv

**other option**

python eval.py --pred fewshot_preds.jsonl --label test_labels.jsonl






# 🧠 MMLU-Pro Inference Pipeline (vLLM + Few-Shot)

This repository provides a **complete end-to-end workflow** for running [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) inference using a vLLM-based LLM inference setup, with optional *few-shot reasoning*.  
It covers **data preprocessing**, **server launch**, **inference**, and **evaluation**.

---

## 📦 1. Data Preprocessing

Prepare question–label pairs from the MMLU-Pro dataset.

```bash
# Generate 200 test examples
python prepare_mmlu_pro.py \
    --split test \
    --out_questions test_questions.jsonl \
    --out_labels test_labels.jsonl \
    --max_examples 200

# Generate the full validation set
python prepare_mmlu_pro.py \
    --split validation \
    --out_questions val_questions.jsonl \
    --out_labels val_labels.jsonl
```

## ⚙️ 2. Launch vLLM Server

Start the vLLM service (locally or on a remote GPU node):

```bash
bash rank0.sh
```

## 🚀 3. Inference

Perform few-shot inference on the test set using the pretrained vLLM server.

```bash
python vllm_inference_dppro_fewshot.py \
    --use_fewshot \
    --validation_set val_questions.jsonl \
    --questions test_questions.jsonl \
    --out fewshot_preds.jsonl \
    --ntrain 1 \
    --B 100
```

## 📊 4. Plot & Evaluation

Generate OKG and filtered accuracy curves for evaluation.

```bash
# OKG-based dynamic curve
python curve_okg_from_preds.py \
    --preds fewshot_preds.jsonl \
    --out fewshot_preds_okg.jsonl \
    --T 10000 \
    --curve_csv curve_okg.csv \
    --labels test_labels.jsonl

# Filtered (per-question) accuracy curve
python curve_filtered_from_preds.py \
    --preds fewshot_preds.jsonl \
    --labels test_labels.jsonl \
    --out_csv curve_filtered.csv
```

## 🧩 5. Additional Evaluation Option

Compute direct accuracy from predictions and labels:

```bash
python eval.py \
    --pred fewshot_preds.jsonl \
    --label test_labels.jsonl
```

## 📁 Output Summary

| File                   | Description                          |
| ---------------------- | ------------------------------------ |
| `test_questions.jsonl` | Test question set                    |
| `test_labels.jsonl`    | Ground-truth answers                 |
| `val_questions.jsonl`  | Validation few-shot examples         |
| `fewshot_preds.jsonl`  | Model predictions                    |
| `curve_okg.csv`        | OKG-based accuracy curve             |
| `curve_filtered.csv`   | Filtered per-question accuracy curve |


