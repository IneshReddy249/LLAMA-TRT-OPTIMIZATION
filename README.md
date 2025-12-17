````md
# LLM Inference Optimization — Llama 3.1 8B (TensorRT-LLM)

End-to-end LLM inference optimization focused on **TTFT**, **TPOT**, **throughput**, and **GPU efficiency**.  
Achieved **3.86× speedup** on **Llama 3.1 8B** using **TensorRT-LLM** with **INT8 weight-only quantization**, **FlashAttention / FMHA**, **Paged KV Cache (Paged Context FMHA)**, **Prefix Caching**, and **Triton Inference Server** for production serving.

---

## TL;DR

- 🚀 **3.86× throughput** vs HuggingFace Transformers
- ⚡ **TTFT (p50)**: **27.6 ms → 9.6 ms**
- 🔥 **GPU utilization**: **58% → 94%**
- 🧠 Reduced CPU launch overhead (**364K → 52K kernel launches**, ~7×)
- 🏗️ Production deployment: **Triton + in-flight batching**

---

## Hardware & Stack

- **GPU:** NVIDIA A100-SXM4-80GB  
- **Model:** Meta-Llama-3.1-8B-Instruct  
- **TensorRT-LLM:** 0.15.0  
- **CUDA:** 12.6  
- **Serving:** Triton Inference Server 24.11  
- **Frameworks:** PyTorch 2.5.0, Transformers 4.46.0

---

## Performance Results

### Baseline vs Optimized (512 output tokens)

| Metric | HuggingFace | TRT-LLM (INT8) | Speedup |
|------|------------|----------------|---------|
| Total Latency | 15,449 ms | 4,001 ms | **3.86×** |
| TTFT (p50) | 27.6 ms | 9.6 ms | **2.88×** |
| Throughput | 33 tok/s | 128 tok/s | **3.86×** |
| GPU Utilization | 58% | 94% | **+36%** |

---

### Triton Scalability (In-Flight Batching)

| Concurrent Requests | Throughput (tok/s) | Efficiency |
|--------------------|--------------------|------------|
| 1 | 131 | 100% |
| 2 | 270 | 103% |
| 4 | 481 | 92% |
| 8 | 897 | 85% |

---

## Key Optimizations Used

- **INT8 Weight-Only Quantization** (reduced weight bandwidth / memory footprint)
- **FlashAttention / FMHA** via TensorRT-LLM attention plugins
- **Paged KV Cache** via **Paged Context FMHA** (efficient KV paging, avoids fragmentation)
- **Prefix Caching** (reuse prefill for shared prefixes to reduce repeated compute)
- **Fused MLP + CUDA Graphs** (fewer launches, lower CPU overhead)
- **In-Flight Batching** (continuous batching under load)
- **Triton Deployment** (production-style serving & concurrency)

---

## Quick Start

### Setup

```bash
git clone https://github.com/ineshtickoo/llama31-trtllm-optimization.git
cd llama31-trtllm-optimization
pip install tensorrt-llm==0.15.0 transformers torch numpy
````

### Convert + Build Engine

```bash
python3 convert_checkpoint.py \
  --model_dir /path/to/llama-3.1-8b-instruct \
  --output_dir checkpoint \
  --dtype float16 \
  --use_weight_only \
  --weight_only_precision int8

trtllm-build \
  --checkpoint_dir checkpoint \
  --output_dir engine \
  --gemm_plugin auto \
  --gpt_attention_plugin auto \
  --use_paged_context_fmha enable \
  --use_fused_mlp enable \
  --max_input_len 2048 \
  --max_seq_len 4096 \
  --max_batch_size 16
```

### Deploy with Triton

```bash
docker run --gpus=all --rm \
  -p 8000:8000 -p 8001:8001 \
  -v $(pwd)/model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.11-trtllm-python-py3 \
  tritonserver --model-repository=/models
```

---

## Benchmarking Methodology

* **Prompt:** “Write a Python function to reverse a linked list”
* **Output:** 512 tokens
* **Warm-up:** 3 runs
* **Runs:** 10 iterations
* **Baseline:** HuggingFace + SDPA
* **Optimized:** TRT-LLM INT8 + FlashAttention/FHMA + Paged KV + Prefix Caching + fused kernels
* **Profiling:** NVIDIA Nsight Systems

---

## Project Structure

```
llama31-trtllm-optimization/
├── scripts/
│   ├── baseline_benchmark.py
│   ├── trtllm_benchmark.py
│   ├── triton_benchmark.py
│   ├── convert_checkpoint.py
│   └── profile_nsight.py
├── model_repository/
│   └── tensorrt_llm/
│       ├── config.pbtxt
│       └── 1/
├── results/
│   ├── baseline_metrics.json
│   ├── optimized_metrics.json
│   └── visualizations/
└── README.md
```

---

## License

MIT

```
```
