from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, os

model_path = f"/home/{os.getenv('USER')}/models/Qwen2.5-7B-Instruct"

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto"
).eval()

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Say hi in one short sentence."}
]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok([prompt], return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=40)
print(tok.decode(out[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True))