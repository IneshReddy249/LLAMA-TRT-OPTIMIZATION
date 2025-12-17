"""
Triton Inference Server Benchmark for TRT-LLM INT8 Engine
"""
import numpy as np
import requests
from transformers import AutoTokenizer
import time
import statistics
import subprocess
import threading

# Config
TRITON_URL = "localhost:8000"
MODEL_NAME = "tensorrt_llm"
TOKENIZER_PATH = "/home/shadeform/llama-trt-optimization/models/llama-3.1-8b-instruct"
MAX_OUTPUT_TOKENS = 128
WARMUP_RUNS = 2
BENCHMARK_RUNS = 10

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

def run_inference_raw(prompt: str, max_tokens: int = MAX_OUTPUT_TOKENS):
    """Single inference using raw HTTP (avoids gevent issues)"""
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(formatted)
    
    # Build request payload
    payload = {
        "inputs": [
            {"name": "input_ids", "shape": [1, len(input_ids)], "datatype": "INT32", "data": input_ids},
            {"name": "input_lengths", "shape": [1, 1], "datatype": "INT32", "data": [len(input_ids)]},
            {"name": "request_output_len", "shape": [1, 1], "datatype": "INT32", "data": [max_tokens]},
            {"name": "end_id", "shape": [1, 1], "datatype": "INT32", "data": [tokenizer.eos_token_id]},
            {"name": "pad_id", "shape": [1, 1], "datatype": "INT32", "data": [tokenizer.pad_token_id or tokenizer.eos_token_id]},
        ],
        "outputs": [
            {"name": "output_ids"},
            {"name": "sequence_length"}
        ]
    }
    
    start = time.perf_counter()
    resp = requests.post(f"http://{TRITON_URL}/v2/models/{MODEL_NAME}/infer", json=payload)
    latency = time.perf_counter() - start
    
    if resp.status_code != 200:
        raise Exception(f"Inference failed: {resp.text}")
    
    result = resp.json()
    
    # Parse output
    output_ids = result["outputs"][0]["data"]
    seq_length = result["outputs"][1]["data"][0]
    
    generated_tokens = seq_length - len(input_ids)
    
    return {
        "latency_ms": latency * 1000,
        "input_tokens": len(input_ids),
        "output_tokens": generated_tokens,
        "tps": generated_tokens / latency if latency > 0 else 0,
    }

def benchmark_sequential():
    """Sequential request benchmark"""
    print("=" * 60)
    print("TRITON INFERENCE SERVER BENCHMARK")
    print("=" * 60)
    
    prompts = [
        "What is machine learning?",
        "Explain GPU memory bandwidth in simple terms.",
        "Write a Python function to sort a list.",
        "What are transformers in deep learning?",
        "How does quantization improve inference speed?",
    ]
    
    # Warmup
    print(f"\nWarmup ({WARMUP_RUNS} runs)...")
    for i in range(WARMUP_RUNS):
        run_inference_raw(prompts[0])
    
    # Benchmark
    print(f"\nBenchmarking ({BENCHMARK_RUNS} runs)...")
    results = []
    for i in range(BENCHMARK_RUNS):
        prompt = prompts[i % len(prompts)]
        result = run_inference_raw(prompt)
        results.append(result)
        print(f"  Run {i+1}: {result['latency_ms']:.1f}ms, {result['output_tokens']} tokens, {result['tps']:.1f} tok/s")
    
    # Aggregate stats
    latencies = [r["latency_ms"] for r in results]
    tps_values = [r["tps"] for r in results]
    output_tokens = [r["output_tokens"] for r in results]
    
    print("\n" + "=" * 60)
    print("SEQUENTIAL REQUEST RESULTS")
    print("=" * 60)
    print(f"Latency (ms):")
    print(f"  p50: {statistics.median(latencies):.1f}")
    print(f"  p90: {sorted(latencies)[int(len(latencies)*0.9)]:.1f}")
    print(f"  p99: {sorted(latencies)[int(len(latencies)*0.99)]:.1f}")
    print(f"  mean: {statistics.mean(latencies):.1f}")
    print(f"\nThroughput (tok/s):")
    print(f"  p50: {statistics.median(tps_values):.1f}")
    print(f"  mean: {statistics.mean(tps_values):.1f}")
    print(f"\nOutput tokens: {min(output_tokens)}/{int(statistics.median(output_tokens))}/{max(output_tokens)} (min/med/max)")
    
    return statistics.median(tps_values)

def benchmark_concurrent(concurrency_levels=[1, 2, 4, 8]):
    """Concurrent request benchmark using threads"""
    print("\n" + "=" * 60)
    print("CONCURRENT REQUEST BENCHMARK (Inflight Batching)")
    print("=" * 60)
    
    prompt = "Explain neural networks briefly."
    
    for concurrency in concurrency_levels:
        results = []
        lock = threading.Lock()
        
        def worker():
            try:
                r = run_inference_raw(prompt, max_tokens=64)
                with lock:
                    results.append(r)
            except Exception as e:
                print(f"  Error: {e}")
        
        threads = [threading.Thread(target=worker) for _ in range(concurrency)]
        
        start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        total_time = time.perf_counter() - start
        
        if results:
            total_tokens = sum(r["output_tokens"] for r in results)
            avg_latency = statistics.mean(r["latency_ms"] for r in results)
            aggregate_tps = total_tokens / total_time
            
            print(f"\nConcurrency {concurrency}:")
            print(f"  Total time: {total_time*1000:.1f} ms")
            print(f"  Avg latency/request: {avg_latency:.1f} ms")
            print(f"  Aggregate throughput: {aggregate_tps:.1f} tok/s")
            print(f"  Requests completed: {len(results)}/{concurrency}")

if __name__ == "__main__":
    # Check server health
    try:
        resp = requests.get(f"http://{TRITON_URL}/v2/health/ready")
        if resp.status_code != 200:
            print("ERROR: Triton server not ready")
            exit(1)
    except:
        print("ERROR: Cannot connect to Triton server")
        exit(1)
    
    print(f"Server: {TRITON_URL}")
    print(f"Model: {MODEL_NAME}")
    
    # Run benchmarks
    seq_tps = benchmark_sequential()
    benchmark_concurrent()
    
    print("\n" + "=" * 60)
    print("COMPARISON vs DIRECT TRT-LLM")
    print("=" * 60)
    print(f"Direct TRT-LLM (ModelRunner): ~128 tok/s")
    print(f"Triton Server:                ~{seq_tps:.0f} tok/s")
    print(f"Overhead: {((seq_tps/128)-1)*100:+.1f}%")
    print("=" * 60)
