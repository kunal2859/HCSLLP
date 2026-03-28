import time
import os
import psutil
import torch
from llama_cpp import Llama
import json

def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # MB

def benchmark_model(model_path, model_label):
    if not os.path.exists(model_path):
        print(f"Skipping {model_label} (model not found at {model_path})")
        return None

    print(f"\nBenchmarking {model_label}...")
    
    start_load = time.time()
    llm = Llama(model_path=model_path, n_ctx=2048, n_gpu_layers=-1, verbose=False)
    load_time = time.time() - start_load
    prompt = "Explain Docker containerization in 100 words."
    start_ttft = time.time()
    stream = llm(prompt, max_tokens=150, stream=True)
    first_token_time = None
    
    token_count = 0
    start_gen = None
    
    for chunk in stream:
        if first_token_time is None:
            first_token_time = time.time() - start_ttft
            start_gen = time.time()
        token_count += 1
    
    total_gen_time = time.time() - start_gen
    tps = token_count / total_gen_time if total_gen_time > 0 else 0
    
    mem_usage = get_memory_usage()
    
    vram_usage = 0
    if torch.backends.mps.is_available():
        vram_usage = torch.mps.current_allocated_memory() / (1024 * 1024)

    results = {
        "model": model_label,
        "load_time_s": round(load_time, 2),
        "ttft_ms": round(first_token_time * 1000, 2),
        "tokens_per_sec": round(tps, 2),
        "peak_ram_mb": round(mem_usage, 2),
        "peak_vram_mb": round(vram_usage, 2)
    }
    
    print(f"{model_label} Results: {results}")
    return results

def main():
    models = [
        ("models/merged_model/phi3-mini-q4_k_m.gguf", "Q4_K_M (Medium Quant)"),
        ("models/merged_model/phi3-mini-q8_0.gguf", "Q8_0 (High Precision)")
    ]
    
    all_results = []
    for path, label in models:
        res = benchmark_model(path, label)
        if res:
            all_results.append(res)
            
    if all_results:
        with open("inference/benchmark_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print("\nBenchmarks saved to inference/benchmark_results.json")

