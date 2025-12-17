import torch
import nvtx
import os
from tensorrt_llm.runtime import ModelRunner
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

TOKENIZER = "/home/shadeform/llama-trt-optimization/models/llama-3.1-8b-instruct"
PROMPT = "Write a Python function to reverse a linked list:"

print("Loading TRT-LLM INT8 engine...")
with nvtx.annotate("Engine Loading", color="blue"):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    runner = ModelRunner.from_dir("/home/shadeform/llama-trt-optimization/engines/optimized/engine")

with nvtx.annotate("Tokenization", color="green"):
    input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"].to("cuda")
    end_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or end_id

# Warmup
print("Warming up...")
with nvtx.annotate("Warmup", color="gray"):
    for _ in range(3):
        runner.generate(input_ids, max_new_tokens=50, end_id=end_id, pad_id=pad_id)
    torch.cuda.synchronize()

# Profile TTFT
print("Profiling TTFT...")
with nvtx.annotate("TTFT (First Token)", color="red"):
    _ = runner.generate(input_ids, max_new_tokens=1, end_id=end_id, pad_id=pad_id)
    torch.cuda.synchronize()

# Profile full generation
print("Profiling full generation (100 tokens)...")
with nvtx.annotate("Full Generation", color="orange"):
    outputs = runner.generate(input_ids, max_new_tokens=100, end_id=end_id, pad_id=pad_id)
    torch.cuda.synchronize()

if outputs.dim() == 3:
    tokens_out = outputs.shape[2] - input_ids.shape[1]
else:
    tokens_out = outputs.shape[1] - input_ids.shape[1]

print(f"Generated {tokens_out} tokens")
print("Profiling complete.")
