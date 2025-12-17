# LLM Inference Optimization Project

## Llama 3.1 8B Instruct - TensorRT-LLM Optimization

### Hardware Configuration
- **GPU**: NVIDIA A100-SXM4-80GB
- **CUDA**: 12.6
- **Framework**: TensorRT-LLM 0.15.0

### Baseline Performance (HuggingFace + SDPA)

| Metric | Value |
|--------|-------|
| Total Latency | 15204.42 ms |
| TTFT (p50) | 28.50 ms |
| TTFT (p99) | 33.20 ms |
| Throughput (TPS) | 33.67 tok/s |
| TPOT | 29.70 ms |
| GPU Utilization | 58% |

### Key Findings

- **CPU Bottleneck**: GPU utilization only 58%, indicating Python generation loop overhead
- **Power Efficiency**: Only 51W power draw out of 400W available (12.75% utilization)
- **Optimization Opportunity**: Significant headroom for TensorRT-LLM improvements

### Optimization Strategy

1. **Stage 1**: Convert to TensorRT-LLM FP16 (Target: 3x speedup)
2. **Stage 2**: Apply INT8 quantization (Additional 1.4x)
3. **Stage 3**: Enable FlashAttention + Paged KV Cache (Additional 1.4x)
4. **Stage 4**: Optimize prefix caching for repeated prompts

### Expected Results

- **Target Throughput**: 118 tok/s (3.5x improvement)
- **Target TPOT**: 8.5 ms (3.5x reduction)
- **Target GPU Utilization**: 95%

### Generated Visualizations

- `performance_dashboard.png` - Complete performance overview
- `optimization_roadmap.png` - Step-by-step optimization strategy
- `system_architecture.png` - Technical architecture diagram
