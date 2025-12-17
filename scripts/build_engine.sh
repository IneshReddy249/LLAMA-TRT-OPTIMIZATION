#!/bin/bash
set -e
export PATH=$PATH:~/.local/bin

trtllm-build \
    --checkpoint_dir /home/shadeform/llama-trt-optimization/engines/optimized/checkpoint \
    --output_dir /home/shadeform/llama-trt-optimization/engines/optimized/engine \
    --gemm_plugin auto \
    --gpt_attention_plugin auto \
    --use_paged_context_fmha enable \
    --use_fused_mlp enable \
    --max_input_len 2048 \
    --max_seq_len 4096 \
    --max_batch_size 16
