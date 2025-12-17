import torch
import nvtx
import subprocess
import time
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.manual_seed(42)

MODEL = "/home/shadeform/llama-trt-optimization/models/llama-3.1-8b-instruct"
PROMPT = "Write a Python function to reverse a linked list:"

print("Loading model...")
with nvtx.annotate("Model Loading", color="blue"):
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="sdpa"
    )
    model.eval()

with nvtx.annotate("Tokenization", color="green"):
    encoded = tokenizer(PROMPT, return_tensors="pt")
    input_ids = encoded["input_ids"].to("cuda")
    attention_mask = encoded["attention_mask"].to("cuda")

gen_config = dict(
    do_sample=False,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
    use_cache=True,
)

# Warmup
print("Warming up...")
with nvtx.annotate("Warmup", color="gray"):
    with torch.inference_mode():
        for _ in range(3):
            _ = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=50, **gen_config)
    torch.cuda.synchronize()

# Profile TTFT
print("Profiling TTFT...")
with nvtx.annotate("TTFT (First Token)", color="red"):
    with torch.inference_mode():
        _ = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=1, **gen_config)
    torch.cuda.synchronize()

# Profile full generation
print("Profiling full generation (100 tokens)...")
with nvtx.annotate("Full Generation", color="orange"):
    with torch.inference_mode():
        with nvtx.annotate("Prefill", color="yellow"):
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=100,
                **gen_config
            )
    torch.cuda.synchronize()

tokens_out = outputs.shape[1] - input_ids.shape[1]
print(f"Generated {tokens_out} tokens")
print("Profiling complete. Check the .nsys-rep file.")
