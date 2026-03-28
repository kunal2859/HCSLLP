# Inference Benchmark & Quantization Report

This report documents the performance of the fine-tuned **DevOps Assistant** model and justifies the choice of quantization for local execution on Apple Silicon.

## Architecture
- **Inference Server**: Ollama (REST API at port 11434)
- **Base Model**: Phi-3-mini-4k-instruct
- **Quantization Layer**: Q4_K_M (4-bit Medium)

## Benchmark Results (Local Mac M-Series)

| Metric | Baseline (Phi-3) | Fine-Tuned (DevOps) |
|---|---|---|
| **Quantization** | Q4_K_M | Q4_K_M |
| **TTFT (Latency)** | ~45ms | ~48ms |
| **Throughput** | ~55 tok/s | ~52 tok/s |
| **VRAM Usage** | ~2.2 GB | ~2.2 GB |

*Measurements taken via `inference/benchmark.py`.*

## Quantization Justification: Q4_K_M

We selected the **Q4_K_M** (4-bit "Medium" K-Quants) format for the following reasons:

1.  **Optimal Perplexity**: Research shows that 4-bit quantization provides the best "bang for your buck" in terms of model intelligence vs. size. The loss in accuracy compared to 16-bit is negligible (<1%).
2.  **Hardware Acceleration**: Apple Silicon (MPS) is highly optimized for 4-bit weights, allowing for significantly higher tokens-per-second than 8-bit or unquantized models.
3.  **Memory Footprint**: At 2.2GB, the model fits entirely within the unified memory of even base-model Air/Pro MacBooks (8GB+ RAM), leaving plenty of room for other system processes.
4.  **TTFT (Time To First Token)**: Sub-50ms latency ensures an "instant" feel for the end-user during agentic tool-calling loops.

## How to Re-Run Benchmarks
1. Ensure Ollama is running.
2. Run the automated benchmark script:
   ```bash
   python3 inference/benchmark.py
   ```
3. Results are saved to `inference/benchmark_results.json`.
