import torch
import subprocess
import time
import threading
import os
import numpy as np
from tensorrt_llm.runtime import ModelRunner
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

TOKENIZER = "/home/shadeform/llama-trt-optimization/models/llama-3.1-8b-instruct"
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

print("Loading TRT-LLM INT8 engine...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
runner = ModelRunner.from_dir("/home/shadeform/llama-trt-optimization/engines/optimized/engine")

input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"].to("cuda")
input_len = input_ids.shape[1]
end_id = tokenizer.eos_token_id
pad_id = tokenizer.pad_token_id or end_id

print("Warming up...")
for _ in range(3):
    runner.generate(input_ids, max_new_tokens=50, end_id=end_id, pad_id=pad_id)
torch.cuda.synchronize()

print("Measuring TTFT (10 runs)...")
ttft_runs = []
for _ in range(10):
    start = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start.record()
    _ = runner.generate(input_ids, max_new_tokens=1, end_id=end_id, pad_id=pad_id)
    end_event.record()
    torch.cuda.synchronize()
    ttft_runs.append(start.elapsed_time(end_event))

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
    end_event = torch.cuda.Event(enable_timing=True)
    start.record()
    outputs = runner.generate(input_ids, max_new_tokens=512, end_id=end_id, pad_id=pad_id)
    end_event.record()
    torch.cuda.synchronize()
    
    latency = start.elapsed_time(end_event)
    
    if outputs.dim() == 3:
        out_ids = outputs[0, 0].tolist()
    else:
        out_ids = outputs[0].tolist()
    
    gen_part = out_ids[input_len:]
    if end_id in gen_part:
        tokens = gen_part.index(end_id) + 1
    else:
        tokens = len(gen_part)
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
print(f"TRT-LLM INT8 METRICS")
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
