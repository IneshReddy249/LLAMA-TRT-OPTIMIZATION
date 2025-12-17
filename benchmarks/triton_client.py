import numpy as np
import tritonclient.http as httpclient
from transformers import AutoTokenizer
import time

# Setup
TRITON_URL = "localhost:8000"
MODEL_NAME = "tensorrt_llm"
TOKENIZER_PATH = "/home/shadeform/llama-trt-optimization/models/llama-3.1-8b-instruct"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

# Create client
client = httpclient.InferenceServerClient(url=TRITON_URL)

# Test prompt
prompt = "Explain the concept of GPU memory bandwidth in three sentences."
messages = [{"role": "user", "content": prompt}]
formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer.encode(formatted, return_tensors="np").astype(np.int32)

print(f"Prompt: {prompt}")
print(f"Input tokens: {input_ids.shape[1]}")
print("-" * 50)

# Prepare inputs - note batch dimension
batch_size = 1
inputs = [
    httpclient.InferInput("input_ids", [batch_size, input_ids.shape[1]], "INT32"),
    httpclient.InferInput("input_lengths", [batch_size, 1], "INT32"),
    httpclient.InferInput("request_output_len", [batch_size, 1], "INT32"),
    httpclient.InferInput("end_id", [batch_size, 1], "INT32"),
    httpclient.InferInput("pad_id", [batch_size, 1], "INT32"),
]

inputs[0].set_data_from_numpy(input_ids)
inputs[1].set_data_from_numpy(np.array([[input_ids.shape[1]]], dtype=np.int32))
inputs[2].set_data_from_numpy(np.array([[256]], dtype=np.int32))
inputs[3].set_data_from_numpy(np.array([[tokenizer.eos_token_id]], dtype=np.int32))
inputs[4].set_data_from_numpy(np.array([[tokenizer.pad_token_id or tokenizer.eos_token_id]], dtype=np.int32))

# Request both outputs
outputs = [
    httpclient.InferRequestedOutput("output_ids"),
    httpclient.InferRequestedOutput("sequence_length"),
]

# Run inference
print("Sending request to Triton...")
start = time.perf_counter()
result = client.infer(MODEL_NAME, inputs, outputs=outputs)
latency = time.perf_counter() - start

# Get outputs
output_ids = result.as_numpy("output_ids")
seq_length = result.as_numpy("sequence_length")

print(f"Output shape: {output_ids.shape}")
print(f"Sequence length: {seq_length}")

# Extract generated tokens (skip input)
# output_ids shape is typically [batch, beam, seq_len]
if len(output_ids.shape) == 3:
    generated_ids = output_ids[0, 0, input_ids.shape[1]:]
else:
    generated_ids = output_ids[0, input_ids.shape[1]:]

generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

print(f"\n=== RESPONSE ===")
print(generated_text)
print(f"\n=== METRICS ===")
print(f"Latency: {latency*1000:.1f} ms")
print(f"Output tokens: {len(generated_ids)}")
print(f"Throughput: {len(generated_ids)/latency:.1f} tok/s")
