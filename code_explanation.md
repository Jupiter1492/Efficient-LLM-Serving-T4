# Code Explanation — Efficient LLM Serving with vLLM and PagedAttention

## Overview

This project benchmarks standard HuggingFace inference against vLLM, a high-performance LLM serving engine that uses PagedAttention for efficient memory management.

The goal is to measure differences in:

* Latency (execution time)
* Throughput (tokens per second)
* GPU memory behavior

---

## Key Concept: KV Cache

During text generation, transformers store previously computed attention values in a structure called the KV (Key-Value) cache.

* Each generated token adds more data to the cache
* Memory usage grows with sequence length
* This becomes the primary bottleneck in LLM serving

Traditional systems allocate this memory contiguously, which leads to fragmentation and wasted GPU memory.

PagedAttention solves this by storing KV cache in non-contiguous blocks, similar to virtual memory in operating systems. ([vLLM][1])

---

## Imports

```python
import time
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
```

* `time` → measures latency
* `torch` → handles GPU execution and memory tracking
* `gc` → manual memory cleanup
* `transformers` → baseline inference
* `vllm` → optimized inference using PagedAttention

---

## Model Selection

```python
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

* Small model chosen for compatibility with limited GPU (T4)
* Ensures reproducible results without memory overflow

---

## Prompt Setup

```python
prompts = [...]
```

* Simulates multiple concurrent users
* Enables batching behavior

---

## HuggingFace Baseline

### Steps:

1. Load tokenizer and model
2. Add padding token if missing
3. Convert prompts to tensors
4. Run generation using:

```python
hf_model.generate(...)
```

### Important Detail:

```python
use_cache=True
```

* Uses standard KV cache (contiguous memory)

---

## Performance Measurement

### Latency

```python
start_time = time.time()
...
torch.cuda.synchronize()
hf_time = time.time() - start_time
```

* `torch.cuda.synchronize()` ensures accurate timing
* Without it, GPU operations may still be running

---

### Throughput

```python
hf_generated_tokens = hf_outputs.numel() - input_token_count
hf_throughput = hf_generated_tokens / hf_time
```

* Measures tokens generated per second

---

### Memory Usage

```python
torch.cuda.max_memory_allocated()
```

* Tracks peak GPU memory usage during execution

---

## Memory Cleanup

```python
del hf_model
gc.collect()
torch.cuda.empty_cache()
```

* Prevents interference between experiments
* Ensures fair comparison

---

## vLLM + PagedAttention

```python
vllm_model = LLM(model=MODEL_ID, gpu_memory_utilization=0.4)
```

### Key Differences:

* Uses block-based KV cache
* Avoids contiguous allocation
* Enables better memory utilization

PagedAttention divides memory into fixed-size blocks, allowing non-contiguous storage and reducing fragmentation. ([Mintlify][2])

---

## vLLM Execution

```python
vllm_outputs = vllm_model.generate(prompts, sampling_params)
```

* Handles batching internally
* Supports dynamic scheduling

---

## Results Calculation

```python
vllm_generated_tokens = sum(...)
vllm_throughput = vllm_generated_tokens / vllm_time
```

---

## Key Findings

* vLLM improves throughput compared to HuggingFace
* Reduces latency under batch workloads
* Uses memory more efficiently

Research shows vLLM can achieve 2–4× higher throughput due to better KV cache management. ([AI-HPC][3])

---

## Important Notes

* vLLM memory metrics may appear low due to custom allocation
* Results vary depending on hardware and batch size
* This experiment validates trends, not absolute performance

---

## Summary

This implementation demonstrates that:

* Memory, not compute, is the main bottleneck in LLM serving
* PagedAttention solves fragmentation using OS-inspired techniques
* vLLM enables higher throughput without modifying the model
