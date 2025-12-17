#!/bin/bash
set -e

python3 convert_checkpoint.py \
    --model_dir /home/shadeform/llama-trt-optimization/models/llama-3.1-8b-instruct \
    --output_dir /home/shadeform/llama-trt-optimization/engines/optimized/checkpoint \
    --dtype float16 \
    --use_weight_only \
    --weight_only_precision int8
