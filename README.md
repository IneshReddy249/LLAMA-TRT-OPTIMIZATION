# LLM Inference Optimization: Llama 3.1 8B with TensorRT-LLM

A comprehensive end-to-end LLM inference optimization project demonstrating systematic performance improvements on Llama 3.1 8B using TensorRT-LLM, advanced profiling techniques, and production deployment with NVIDIA Triton Inference Server.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Performance Results](#performance-results)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Optimization Techniques](#optimization-techniques)
- [Profiling & Bottleneck Analysis](#profiling--bottleneck-analysis)
- [Deployment](#deployment)
- [Benchmarking Methodology](#benchmarking-methodology)
- [Future Improvements](#future-improvements)
- [License](#license)

## Overview

This project implements a systematic approach to optimizing large language model inference, focusing on Llama 3.1 8B. The workflow encompasses baseline establishment, iterative optimization with TensorRT-LLM, production deployment via Triton Inference Server, and comprehensive performance profiling using NVIDIA Nsight Systems.

### Project Goals

- Achieve significant latency reduction and throughput improvement
- Maximize GPU utilization for cost-effective inference
- Establish reproducible optimization methodology
- Deploy production-ready inference service
- Provide detailed performance metrics and bottleneck analysis

## Key Features

- **Systematic Optimization Pipeline**: Baseline → Optimize → Deploy → Profile → Iterate
- **Advanced Profiling**: Deep performance analysis using NVIDIA Nsight Systems
- **Production Deployment**: Scalable serving with Triton Inference Server
- **Comprehensive Metrics**: Latency, throughput, GPU utilization, memory efficiency
- **INT8 Quantization**: Weight-only quantization for reduced memory footprint
- **In-Flight Batching**: Dynamic batching with 85-103% scaling efficiency
- **Memory Optimization**: Paged KV cache and optimized memory management
- **Real Performance Data**: All results validated on NVIDIA A100 GPU

## Visual Results

The project includes comprehensive performance visualizations:

1. **Performance Dashboard** - Overall speedup metrics showing 3.86x improvement
2. **Detailed Comparison** - 6-panel analysis comparing HuggingFace baseline vs TRT-LLM INT8
3. **Scalability Analysis** - Triton Server concurrent request handling (up to 897 tok/s with 8 requests)
4. **TTFT Distribution** - Percentile analysis (p50, p90, p95, p99) across both implementations

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Optimization Pipeline                     │
├─────────────────────────────────────────────────────────────────┤
│  1. Baseline                                                     │
│     └─ Hugging Face Llama 3.1 8B (FP16)                        │
│     └─ Initial metrics capture with Nsight Systems              │
│                                                                  │
│  2. TensorRT-LLM Optimization                                   │
│     └─ INT8 Quantization                                        │
│     └─ Paged KV Caching                                         │
│     └─ Flash Attention 2                                        │
│     └─ In-Flight Batching                                       │
│     └─ Chunked Prefill (FMHA)                                   │
│     └─ Prefix Caching                                           │
│                                                                  │
│  3. Production Deployment                                        │
│     └─ NVIDIA Triton Inference Server                           │
│     └─ Dynamic batching & concurrent execution                  │
│     └─ gRPC/HTTP endpoints                                      │
│                                                                  │
│  4. Iterative Profiling                                         │
│     └─ Nsight Systems analysis                                  │
│     └─ Bottleneck identification                                │
│     └─ Performance tuning                                       │
└─────────────────────────────────────────────────────────────────┘
```

## Performance Results

### Throughput & Latency Improvements

| Metric | Baseline (HuggingFace) | Optimized (TRT-LLM INT8) | Improvement |
|--------|------------------------|--------------------------|-------------|
| **Total Latency (512 tokens)** | 15,449 ms | 4,001 ms | **3.86x faster** |
| **Time to First Token (TTFT) - p50** | 27.6 ms | 9.6 ms | **2.88x faster** |
| **Time to First Token (TTFT) - p99** | 31.0 ms | 10.2 ms | **3.04x faster** |
| **Throughput (TPS)** | 33.14 tok/s | 128 tok/s | **3.86x faster** |
| **Time Per Output Token (TPOT)** | 30.17 ms | 7.81 ms | **3.86x faster** |

### Resource Utilization

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| **GPU Utilization** | 58% | 94% | **+36% (1.62x)** |
| **Power Draw** | 51W (12.8%) | ~300W (75%) | **5.9x increase** |
| **Memory Efficiency** | Standard | INT8 Quantized | **50% reduction** |
| **Kernel Launch Overhead** | 364K launches | 52K launches | **7x reduction** |

### Triton Inference Server - Concurrent Request Handling

| Concurrent Requests | Aggregate Throughput (tok/s) | Scaling Efficiency |
|---------------------|-----------------------------|--------------------|
| 1 | 131 tok/s | 100% (baseline) |
| 2 | 270 tok/s | 103% (super-linear!) |
| 4 | 481 tok/s | 92% |
| 8 | 897 tok/s | 85% |

**Key Bottleneck Fixed**: CPU kernel launch overhead reduced from 364K to 52K launches (7x reduction) through CUDA graphs and kernel fusion

*All benchmarks performed on NVIDIA A100-SXM4-80GB GPU with CUDA 12.6 and TensorRT-LLM 0.15.0*

## Tech Stack

### Core Framework
- **TensorRT-LLM** (v0.15.0) - High-performance LLM inference engine
- **NVIDIA Triton Inference Server** (v24.11) - Production model serving
- **PyTorch** (v2.5.0) - Deep learning framework
- **CUDA** (12.6) - GPU acceleration

### Optimization Components
- **INT8 Weight-Only Quantization** - 8-bit quantization for weights
- **Paged Context FMHA** - Memory-efficient fused multi-head attention
- **Fused MLP** - Kernel fusion for feed-forward layers
- **In-Flight Batching** - Continuous batching for improved throughput
- **CUDA Graphs** - Kernel launch overhead reduction
- **GEMM Plugin** - Optimized matrix multiplication

### Profiling & Analysis
- **NVIDIA Nsight Systems** (v2024.6.1) - GPU profiling and tracing
- **nvidia-smi** - Real-time GPU monitoring
- **Custom Python Benchmarking Scripts** - Latency and throughput measurement

### Development Tools
- **Python** (3.10)
- **Transformers** (v4.46.0) - Model loading and tokenization
- **NumPy** (v1.26.4) - Numerical computing
- **Matplotlib** (v3.9.0) - Visualization
- **tritonclient** - Triton server client library

## Prerequisites

### Hardware Requirements
- NVIDIA GPU with Compute Capability ≥ 8.0 (Ampere or newer recommended)
- Minimum 24GB GPU memory (40GB+ recommended for batch optimization)
- 64GB+ system RAM
- 100GB+ free disk space

### Software Requirements
- NVIDIA Driver version ≥ 525.60.13
- CUDA Toolkit 12.6
- Docker and NVIDIA Container Runtime
- Python 3.10+

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ineshtickoo/llama31-trtllm-optimization.git
cd llama31-trtllm-optimization
```

### 2. Set Up Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Install TensorRT-LLM

```bash
# Using Docker (Recommended)
docker pull nvcr.io/nvidia/tensorrt-llm:0.16.0-py3

# Or build from source
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
python3 ./scripts/build_wheel.py --clean --cuda_architectures "80-real;86-real;89-real;90-real"
pip install build/tensorrt_llm-*.whl
```

### 4. Install NVIDIA Triton Server

```bash
docker pull nvcr.io/nvidia/tritonserver:24.12-trtllm-python-py3
```

### 5. Install Profiling Tools

```bash
# NVIDIA Nsight Systems
wget https://developer.download.nvidia.com/devtools/nsight-systems/2024_6/nsight-systems-2024.6.1_2024.6.1.92-1_amd64.deb
sudo dpkg -i nsight-systems-2024.6.1_2024.6.1.92-1_amd64.deb

# NVIDIA Nsight Compute
wget https://developer.download.nvidia.com/devtools/nsight-compute/2024_3/nsight-compute-2024.3.1_2024.3.1.1-1_amd64.deb
sudo dpkg -i nsight-compute-2024.3.1_2024.3.1.1-1_amd64.deb
```

## Project Structure

```
llama31-trtllm-optimization/
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   ├── baseline_config.yaml          # Baseline model configuration
│   ├── optimized_config.yaml         # Optimized TRT-LLM configuration
│   └── triton_config.pbtxt           # Triton server configuration
├── scripts/
│   ├── 01_baseline/
│   │   ├── run_baseline.py           # Baseline inference script
│   │   └── profile_baseline.sh       # Nsight profiling wrapper
│   ├── 02_optimize/
│   │   ├── build_trtllm_engine.py    # TRT-LLM engine builder
│   │   ├── quantize_model.py         # INT8 quantization
│   │   └── test_optimized.py         # Optimized inference testing
│   ├── 03_deploy/
│   │   ├── setup_triton.sh           # Triton setup script
│   │   ├── model_repository/         # Triton model repository
│   │   └── client_test.py            # Inference client
│   ├── 04_benchmark/
│   │   ├── benchmark_suite.py        # Comprehensive benchmarking
│   │   ├── profile_optimized.sh      # Optimized profiling
│   │   └── analyze_nsight.py         # Nsight report analysis
│   └── utils/
│       ├── metrics.py                # Metrics collection utilities
│       ├── visualization.py          # Performance visualization
│       └── dataset.py                # Test dataset preparation
├── models/
│   ├── baseline/                     # Baseline model weights
│   ├── optimized/                    # TRT-LLM engines
│   └── quantized/                    # Quantized checkpoints
├── results/
│   ├── baseline_metrics.json
│   ├── optimized_metrics.json
│   ├── nsight_reports/
│   │   ├── baseline_profile.nsys-rep
│   │   └── optimized_profile.nsys-rep
│   └── visualizations/
│       ├── latency_comparison.png
│       ├── throughput_analysis.png
│       └── gpu_utilization.png
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_analysis.ipynb
│   └── 03_optimization_results.ipynb
└── docs/
    ├── OPTIMIZATION_GUIDE.md
    ├── PROFILING_GUIDE.md
    └── DEPLOYMENT_GUIDE.md
```

## Usage

### 1. Run Baseline Inference

```bash
# Download Llama 3.1 8B model
python scripts/01_baseline/download_model.py --model meta-llama/Meta-Llama-3.1-8B

# Run baseline inference with profiling
bash scripts/01_baseline/profile_baseline.sh
```

### 2. Build Optimized TensorRT-LLM Engine

```bash
# Quantize model to INT8
python scripts/02_optimize/quantize_model.py \
    --model_dir models/baseline/llama-3.1-8b \
    --output_dir models/quantized/llama-3.1-8b-int8 \
    --dtype int8

# Build TensorRT-LLM engine with optimizations
python scripts/02_optimize/build_trtllm_engine.py \
    --model_dir models/quantized/llama-3.1-8b-int8 \
    --output_dir models/optimized/llama-3.1-8b-trtllm \
    --max_batch_size 128 \
    --max_input_len 2048 \
    --max_output_len 2048 \
    --use_paged_kv_cache \
    --use_gpt_attention_plugin \
    --use_gemm_plugin \
    --use_inflight_batching \
    --enable_chunked_context \
    --enable_prefix_caching
```

### 3. Deploy with Triton Inference Server

```bash
# Set up Triton model repository
bash scripts/03_deploy/setup_triton.sh

# Launch Triton Server
docker run --gpus=all --rm --net=host \
    -v $(pwd)/scripts/03_deploy/model_repository:/models \
    nvcr.io/nvidia/tritonserver:24.12-trtllm-python-py3 \
    tritonserver --model-repository=/models \
    --http-port 8000 \
    --grpc-port 8001 \
    --metrics-port 8002

# Test deployment
python scripts/03_deploy/client_test.py --server-url localhost:8001
```

### 4. Run Comprehensive Benchmarks

```bash
# Execute full benchmark suite
python scripts/04_benchmark/benchmark_suite.py \
    --baseline-model models/baseline/llama-3.1-8b \
    --optimized-engine models/optimized/llama-3.1-8b-trtllm \
    --triton-url localhost:8001 \
    --output-dir results/ \
    --num-requests 1000 \
    --concurrent-users 1,10,50,100

# Profile optimized inference
bash scripts/04_benchmark/profile_optimized.sh

# Analyze Nsight reports
python scripts/04_benchmark/analyze_nsight.py \
    --baseline-report results/nsight_reports/baseline_profile.nsys-rep \
    --optimized-report results/nsight_reports/optimized_profile.nsys-rep \
    --output-dir results/visualizations/
```

## Optimization Techniques

### 1. INT8 Weight-Only Quantization

Reduces model weight size by 50% while maintaining full-precision activations for accuracy.

```bash
# Checkpoint conversion with INT8 quantization
python3 convert_checkpoint.py \
    --model_dir /path/to/llama-3.1-8b-instruct \
    --output_dir /path/to/checkpoint \
    --dtype float16 \
    --use_weight_only \
    --weight_only_precision int8
```

**Benefits:**
- 50% reduction in weight memory footprint
- Reduced memory bandwidth requirements
- Maintained activation precision for accuracy
- Minimal accuracy degradation

### 2. Paged Context FMHA (Fused Multi-Head Attention)

Combines paged KV cache management with fused attention kernels.

```bash
# Engine build with paged context FMHA
trtllm-build \
    --checkpoint_dir /path/to/checkpoint \
    --output_dir /path/to/engine \
    --use_paged_context_fmha enable \
    --max_input_len 2048 \
    --max_seq_len 4096
```

**Benefits:**
- Eliminates memory fragmentation in KV cache
- Fuses attention operations into single kernel
- Enables longer sequence lengths
- Better memory efficiency

### 3. Fused MLP & GEMM Plugin

Fuses feed-forward network operations and optimizes matrix multiplications.

```bash
trtllm-build \
    --use_fused_mlp enable \
    --gemm_plugin auto \
    --gpt_attention_plugin auto
```

**Benefits:**
- Reduces kernel launch overhead
- Improves memory access patterns
- Better instruction-level parallelism
- 7x reduction in kernel launches (364K → 52K)

### 4. In-Flight Batching

Continuous batching that processes requests as they arrive without waiting for batch completion.

```
# Triton configuration
parameters {
  key: "batching_type"
  value: { string_value: "inflight_fused_batching" }
}
```

**Benefits:**
- 85-103% scaling efficiency up to 8 concurrent requests
- Reduced average latency under load
- Better GPU utilization (58% → 94%)
- Handles variable-length sequences efficiently

### 5. CUDA Graphs

Captures and replays GPU operations to reduce CPU-GPU synchronization overhead.

**Benefits:**
- 7x reduction in kernel launches
- Reduced Python GIL contention
- Lower CPU overhead
- More consistent latencies

### 6. KV Cache Memory Management

Optimized GPU memory allocation for key-value caches.

```
parameters {
  key: "kv_cache_free_gpu_mem_fraction"
  value: { string_value: "0.40" }
}
```

**Benefits:**
- 40% of free GPU memory allocated to KV cache
- Better batch size scaling
- Reduced OOM errors
- Efficient memory utilization

## Profiling & Bottleneck Analysis

### NVIDIA Nsight Systems

Captures system-wide performance data including GPU kernels, memory transfers, and CPU activity.

```bash
# Profile baseline with Nsight Systems
nsys profile \
    --trace=cuda,nvtx \
    --output=baseline_profile \
    python scripts/baseline_benchmark.py

# Profile optimized version
nsys profile \
    --trace=cuda,nvtx \
    --output=optimized_profile \
    python scripts/trtllm_benchmark.py
```

**Key Metrics Analyzed:**
- Kernel execution time distribution
- CPU kernel launch overhead
- GPU utilization over time
- CUDA API calls and synchronization

### Bottleneck Identification

**Baseline Bottlenecks Found:**
1. **CPU Kernel Launch Overhead** (364K launches)
   - Python generation loop causing excessive kernel launches
   - Poor GPU utilization (58%)
   - Low power draw (51W out of 400W available)

2. **Inefficient Attention Operations**
   - Separate kernels for Q, K, V projections
   - No kernel fusion
   - High memory bandwidth usage

3. **Sequential Token Generation**
   - GIL contention in Python
   - Synchronous execution
   - CPU bottleneck limiting GPU

**Optimizations Applied:**
1. TensorRT-LLM engine → Fused kernels and CUDA graphs
2. INT8 quantization → Reduced memory bandwidth
3. Paged Context FMHA → Optimized attention computation
4. In-flight batching → Better GPU utilization

**Results:**
- Kernel launches: 364K → 52K (7x reduction)
- GPU utilization: 58% → 94% (+36%)
- Power draw: 51W → ~300W (optimal utilization)
- Overall speedup: 3.86x

### Performance Visualizations

The project includes benchmark visualizations showing:
- **HuggingFace vs TRT-LLM INT8 comparison** (6-panel chart)
- **Triton Server Scalability** (concurrent request performance)
- **TTFT Percentile Distribution** (p50, p90, p95, p99)
- **Overall speedup metrics** across all dimensions

## Deployment

### Triton Inference Server Configuration

```protobuf
name: "tensorrt_llm"
backend: "tensorrtllm"
max_batch_size: 4

input [
  {
    name: "input_ids"
    data_type: TYPE_INT32
    dims: [-1]
  },
  {
    name: "input_lengths"
    data_type: TYPE_INT32
    dims: [1]
  },
  {
    name: "request_output_len"
    data_type: TYPE_INT32
    dims: [1]
  },
  {
    name: "end_id"
    data_type: TYPE_INT32
    dims: [1]
    optional: true
  },
  {
    name: "pad_id"
    data_type: TYPE_INT32
    dims: [1]
    optional: true
  }
]

output [
  {
    name: "output_ids"
    data_type: TYPE_INT32
    dims: [-1, -1]
  },
  {
    name: "sequence_length"
    data_type: TYPE_INT32
    dims: [-1]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [0]
  }
]

parameters: {
  key: "gpt_model_type"
  value: { string_value: "inflight_fused_batching" }
}

parameters: {
  key: "gpt_model_path"
  value: { string_value: "/models/tensorrt_llm/1" }
}

parameters: {
  key: "max_beam_width"
  value: { string_value: "1" }
}

parameters: {
  key: "batching_type"
  value: { string_value: "inflight_fused_batching" }
}

parameters: {
  key: "kv_cache_free_gpu_mem_fraction"
  value: { string_value: "0.40" }
}
```

### Launching Triton Server

```bash
# Start Triton Inference Server
docker run --gpus=all --rm --net=host \
    -v /path/to/model_repository:/models \
    nvcr.io/nvidia/tritonserver:24.11-trtllm-python-py3 \
    tritonserver --model-repository=/models
```

## Benchmarking Methodology

### Test Configuration

- **Hardware**: NVIDIA A100-SXM4-80GB (PCIe)
- **Model**: Meta-Llama-3.1-8B-Instruct
- **Test Prompt**: "Write a Python function to reverse a linked list:"
- **Output Length**: 512 tokens
- **Warmup Runs**: 3 iterations
- **Benchmark Runs**: 10 iterations per test
- **Concurrent Request Tests**: 1, 2, 4, 8 concurrent requests

### Metrics Collected

1. **Latency Metrics**
   - Total Latency (ms) - End-to-end generation time
   - Time to First Token (TTFT) - Prefill latency
   - Time per Output Token (TPOT) - Decode latency
   - Percentiles: p50, p90, p95, p99

2. **Throughput Metrics**
   - Tokens per second (TPS)
   - Aggregate throughput (concurrent requests)

3. **Resource Utilization**
   - GPU utilization (%) via nvidia-smi
   - Power draw (W)
   - Memory utilization

### Baseline Implementation

**HuggingFace Transformers with SDPA:**
```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="cuda",
    attn_implementation="sdpa"  # Scaled Dot Product Attention
)
```

### Optimized Implementation

**TensorRT-LLM INT8:**
```bash
# Checkpoint conversion
python3 convert_checkpoint.py \
    --model_dir /path/to/model \
    --output_dir /path/to/checkpoint \
    --dtype float16 \
    --use_weight_only \
    --weight_only_precision int8

# Engine build
trtllm-build \
    --checkpoint_dir /path/to/checkpoint \
    --output_dir /path/to/engine \
    --gemm_plugin auto \
    --gpt_attention_plugin auto \
    --use_paged_context_fmha enable \
    --use_fused_mlp enable \
    --max_input_len 2048 \
    --max_seq_len 4096 \
    --max_batch_size 16
```

### Reproducibility

All experiments use:
- Fixed random seeds (torch.manual_seed(42))
- TOKENIZERS_PARALLELISM="false" to avoid threading issues
- GPU in default performance state
- Consistent test prompts across runs
- Multiple warmup iterations before measurement

```python
# GPU monitoring thread
def monitor_gpu():
    while monitoring:
        result = subprocess.run(
            ['nvidia-smi', '-i', GPU_ID,
             '--query-gpu=utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        gpu_utils.append(float(result.stdout.strip()))
        time.sleep(0.05)
```

## Future Improvements

### Planned Enhancements

1. **Multi-GPU Support**
   - Tensor parallelism across multiple GPUs
   - Pipeline parallelism for large batches
   - Expert parallelism for MoE models

2. **Advanced Quantization**
   - FP8 quantization support
   - Mixed precision strategies
   - Per-layer quantization analysis

3. **Speculative Decoding**
   - Draft model integration
   - Verification mechanism
   - Adaptive speculation strategies

4. **Memory Optimization**
   - Cross-layer KV sharing
   - Compression techniques
   - Offloading strategies

5. **Monitoring & Observability**
   - Prometheus metrics integration
   - Grafana dashboards
   - Distributed tracing

6. **Automated Optimization**
   - AutoTuner for hyperparameters
   - Profile-guided optimization
   - A/B testing framework

### Research Directions

- Kernel fusion opportunities
- Custom CUDA kernels for specific operations
- Alternative attention mechanisms
- Hardware-aware architecture search

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure:
- Code follows PEP 8 style guidelines
- All tests pass
- Documentation is updated
- Commit messages are clear and descriptive

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{llama31_trtllm_optimization,
  author = {Your Name},
  title = {LLM Inference Optimization: Llama 3.1 8B with TensorRT-LLM},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/llama31-trtllm-optimization}
}
```

## Acknowledgments

- NVIDIA for TensorRT-LLM and Triton Inference Server
- Meta AI for Llama 3.1 model
- Hugging Face for model hosting and transformers library
- The open-source community for various optimization techniques

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**Inesh Tickoo**  
MS Computer Science, Florida Atlantic University (Dec 2025)  
Email: itickoo2023@fau.edu  
LinkedIn: [linkedin.com/in/inesh-tickoo](https://linkedin.com/in/inesh-tickoo)  
GitHub: [@ineshtickoo](https://github.com/ineshtickoo)

---

**Project Status**: Completed  
**Last Updated**: December 2024  
**Hardware**: NVIDIA A100-SXM4-80GB via Shadeform Cloud  
**Framework**: TensorRT-LLM 0.15.0
