# Efficient LLM Serving Using vLLM and PagedAttention

## CAP6614 Final Project

University of Central Florida

## Author

Nicholas Lisle
Nicholas Lisle conducted all aspects of this project, including implementation, experimentation, analysis, and report writing.

---

## Project Overview

This project evaluates the efficiency improvements of vLLM and PagedAttention compared to standard HuggingFace inference for Large Language Model (LLM) serving.

The goal is to measure real-world performance differences in:

* Throughput (tokens per second)
* Latency (response time)
* Approximate GPU memory usage

without modifying the model itself.

---

## Experimental Setup

### Model

TinyLlama/TinyLlama-1.1B-Chat-v1.0

### Hardware

Google Colab environment using an NVIDIA Tesla T4 GPU

### Frameworks

* PyTorch
* HuggingFace Transformers
* vLLM (PagedAttention)

---

## Benchmark Results

| Method      |  Latency |        Throughput |
| ----------- | -------: | ----------------: |
| HuggingFace | 4.91 sec | 162.86 tokens/sec |
| vLLM        | 1.88 sec | 425.28 tokens/sec |

Results are based on a single-run benchmark on a Tesla T4 GPU and may vary depending on hardware and batch size.

### Key Result

vLLM achieved approximately **2.6× higher throughput** and significantly lower latency compared to standard HuggingFace inference.

---

## Method Summary

Traditional HuggingFace inference uses a contiguous KV cache, which leads to memory fragmentation and inefficient GPU utilization.

vLLM introduces **PagedAttention**, which:

* Breaks KV cache into fixed-size memory blocks
* Uses a mapping system similar to virtual memory
* Significantly reduces memory fragmentation
* Enables larger batch sizes and higher throughput

---

## How to Run

### Install dependencies

pip install -r requirements.txt

---

### Run benchmark

python experimental_code.py

---

## Reproducing Results

Use Google Colab with GPU enabled:

Runtime → Change Runtime Type → Select T4 GPU

Then run the script to compare HuggingFace and vLLM performance.

---

## Files in This Repository

* experimental_code.py → main benchmark script
* code_explanation.md → explanation of code logic
* requirements.txt → dependencies
* final_report.pdf → written report
* presentation_slides.pdf → presentation slides
* colab_output.pdf → execution output proof
* results-benchmark_output.txt → raw results
