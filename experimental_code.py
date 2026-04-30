import time
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

# Small model for safe testing on Colab T4 or smaller GPUs
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Simulating multiple users sending requests at once
prompts = [
    "The future of artificial intelligence is",
    "To bake a chocolate cake, you need",
    "Once upon a time in a distant galaxy",
    "The most important invention of the 20th century was",
    "In order to learn Python effectively, one should",
    "The quick brown fox jumps over the",
    "Quantum computing will change the world because",
    "The best way to manage memory in C is"
]

MAX_TOKENS = 100
BATCH_SIZE = len(prompts)

print("=" * 50)
print(" EXPERIMENT 1: HUGGINGFACE (STANDARD INFERENCE)")
print("=" * 50)

# -----------------------------
# HUGGINGFACE BASELINE TEST
# -----------------------------

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Add padding token if missing (needed for batch processing)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model using safer auto device placement
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Convert prompts into model input tensors
inputs = tokenizer(
    prompts,
    return_tensors="pt",
    padding=True
).to("cuda")

# Count original input tokens
input_token_count = inputs["input_ids"].numel()

# Reset GPU memory tracker
torch.cuda.reset_peak_memory_stats()

# Start timer
start_time = time.time()

# Run generation using standard HuggingFace inference
with torch.no_grad():
    hf_outputs = hf_model.generate(
        **inputs,
        max_new_tokens=MAX_TOKENS,
        use_cache=True,  # Standard contiguous KV-cache
        pad_token_id=tokenizer.pad_token_id
    )

# IMPORTANT: wait for GPU to finish before stopping timer
torch.cuda.synchronize()
hf_time = time.time() - start_time

# Track approximate peak GPU memory usage
hf_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)

# Count generated tokens only
hf_generated_tokens = hf_outputs.numel() - input_token_count

# Throughput = tokens generated per second
hf_throughput = hf_generated_tokens / hf_time

print(f"HuggingFace Latency: {hf_time:.2f} seconds")
print(f"HuggingFace Throughput: {hf_throughput:.2f} tokens/sec")
print(f"HuggingFace Peak VRAM: {hf_memory:.2f} GB\n")

# -----------------------------
# CLEAN GPU MEMORY BEFORE vLLM
# -----------------------------

print("Cleaning up VRAM...")

del hf_model
del inputs
del hf_outputs

gc.collect()
torch.cuda.empty_cache()

# Give CUDA a moment to fully release memory
time.sleep(2)

print("=" * 50)
print(" EXPERIMENT 2: vLLM (PAGEDATTENTION)")
print("=" * 50)

# -----------------------------
# vLLM PAGEDATTENTION TEST
# -----------------------------

# Restrict memory usage so Colab T4 doesn't crash
vllm_model = LLM(
    model=MODEL_ID,
    gpu_memory_utilization=0.4
)

sampling_params = SamplingParams(
    max_tokens=MAX_TOKENS
)

# Reset memory tracker again
torch.cuda.reset_peak_memory_stats()

# Start timer
start_time = time.time()

# Run vLLM generation
vllm_outputs = vllm_model.generate(
    prompts,
    sampling_params
)

# IMPORTANT: wait for GPU to finish
torch.cuda.synchronize()
vllm_time = time.time() - start_time

# Approximate execution memory footprint
vllm_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)

# Count total generated tokens
vllm_generated_tokens = sum(
    len(output.outputs[0].token_ids)
    for output in vllm_outputs
)

# Throughput calculation
vllm_throughput = vllm_generated_tokens / vllm_time

print(f"vLLM Latency: {vllm_time:.2f} seconds")
print(f"vLLM Throughput: {vllm_throughput:.2f} tokens/sec")
print(f"vLLM Approx. Peak VRAM: {vllm_memory:.2f} GB\n")

print("=" * 50)
print(" RESULTS SUMMARY")
print("=" * 50)

print(f"Throughput Increase: {vllm_throughput / hf_throughput:.2f}x faster with vLLM")
print(f"Latency Improvement: {hf_time / vllm_time:.2f}x faster response with vLLM")
