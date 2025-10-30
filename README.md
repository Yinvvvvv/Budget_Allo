# Data Preprocess for MMLU_Pro

prepare_mmlu_pro.py --split test --out_questions test_questions.jsonl --out_labels test_labels.jsonl 

prepare_mmlu_pro.py --split validation --out_questions val_questions.jsonl --out_labels val_labels.jsonl

# Load vLLM

bash rank0.sh

# Inference

python vllm_inference_dppro_fewshot.py --use_fewshot --validation_set val_questions.jsonl --questions test_questions.jsonl --out fewshot_preds.jsonl --ntrain 1 --B 100

# Plot and Evaluation

python curve_okg_from_preds.py --preds fewshot_preds.jsonl --out fewshot_preds_okg.jsonl --T 10000 --curve_csv curve_okg.csv --labels test_labels.jsonl

python curve_filtered_from_preds.py --preds fewshot_preds.jsonl --labels test_labels.jsonl --out_csv curve_filtered.csv

*other option*

python eval.py --pred fewshot_preds.jsonl --label test_labels.jsonl
