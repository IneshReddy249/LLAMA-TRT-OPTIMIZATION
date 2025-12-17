import torch
import subprocess
import time
import threading
import os
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.manual_seed(42)

MODEL = "/home/shadeform/llama-trt-optimization/models/llama-3.1-8b-instruct"
PROMPT = "Write a Python function to reverse a linked list:"
GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]

gpu_utils = []
monitoring = False

def monitor_gpu():
    while monitoring:
        try:
            result = subprocess.run(
                ['nvidia-smi', '-i', GPU_ID,
                 '--query-gpu=utilization.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, check=True, timeout=1
            )
            gpu_utils.append(float(result.stdout.strip().splitlines()[0]))
            time.sleep(0.05)
        except:
            pass

print("Loading HuggingFace model...")
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

encoded = tokenizer(PROMPT, return_tensors="pt")
input_ids = encoded["input_ids"].to("cuda")
attention_mask = encoded["attention_mask"].to("cuda")
input_len = input_ids.shape[1]

gen_config = dict(
    do_sample=False,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
    use_cache=True,
)

print("Warming up...")
with torch.inference_mode():
    for _ in range(3):
        _ = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=50, **gen_config)
torch.cuda.synchronize()

print("Measuring TTFT (10 runs)...")
ttft_runs = []
for _ in range(10):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.inference_mode():
        start.record()
        _ = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=1, **gen_config)
        end.record()
    torch.cuda.synchronize()
    ttft_runs.append(start.elapsed_time(end))

print("Measuring generation (10 runs x 512 tokens)...")
latencies = []
tokens_list = []
tps_runs = []

gpu_utils.clear()
monitoring = True
monitor_thread = threading.Thread(target=monitor_gpu, daemon=True)
monitor_thread.start()

for _ in range(10):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.inference_mode():
        start.record()
        outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=512, **gen_config)
        end.record()
    torch.cuda.synchronize()
    
    latency = start.elapsed_time(end)
    tokens = outputs.shape[1] - input_len
    tokens = max(tokens, 1)
    
    latencies.append(latency)
    tokens_list.append(tokens)
    tps_runs.append(tokens / (latency / 1000))

monitoring = False
monitor_thread.join(timeout=2)

total_latency = np.median(latencies)
ttft_p50 = np.percentile(ttft_runs, 50)
ttft_p90 = np.percentile(ttft_runs, 90)
ttft_p95 = np.percentile(ttft_runs, 95)
ttft_p99 = np.percentile(ttft_runs, 99)
tps = np.median(tps_runs)
tpot_runs = [lat / max(tok, 1) for lat, tok in zip(latencies, tokens_list)]
tpot = np.median(tpot_runs)
max_gpu_util = max(gpu_utils) if gpu_utils else 0

print(f"\n{'='*70}")
print(f"HUGGINGFACE BASELINE METRICS")
print(f"{'='*70}")
print(f"Total Latency:     {total_latency:.2f} ms")
print(f"TTFT (p50):        {ttft_p50:.2f} ms")
print(f"TTFT (p90):        {ttft_p90:.2f} ms")
print(f"TTFT (p95):        {ttft_p95:.2f} ms")
print(f"TTFT (p99):        {ttft_p99:.2f} ms")
print(f"TPS:               {tps:.2f} tok/s")
print(f"TPOT:              {tpot:.2f} ms")
print(f"GPU Util (max):    {max_gpu_util:.0f}%")
print(f"{'='*70}")
