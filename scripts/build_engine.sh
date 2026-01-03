#!/bin/bash
CHECKPOINT_DIR="/workspace/models/llama-3.1-8b-fp8"
OUTPUT_DIR="/workspace/engines/llama-8b-fp8"

mkdir -p $OUTPUT_DIR

trtllm-build \
    --checkpoint_dir $CHECKPOINT_DIR \
    --output_dir $OUTPUT_DIR \
    --gemm_plugin float16 \
    --gpt_attention_plugin float16 \
    --max_batch_size 16 \
    --max_input_len 2048 \
    --max_seq_len 4096 \
    --paged_kv_cache enable \
    --use_paged_context_fmha enable \
    --multiple_profiles enable \
    --use_fp8_context_fmha enable
