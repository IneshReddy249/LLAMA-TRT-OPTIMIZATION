python3 /opt/TensorRT-LLM-examples/quantization/quantize.py \
  --model_dir /workspace/models/llama-3.1-8b-instruct \
  --output_dir /workspace/models/llama-3.1-8b-fp8 \
  --dtype float16 \
  --qformat fp8 \
  --calib_dataset /workspace/calib_data.json \
  --calib_size 512 \
  --kv_cache_dtype fp8
