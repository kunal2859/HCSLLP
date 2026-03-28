import time
import requests  # type: ignore
import json

def benchmark_ollama(model_name):
    url = "http://localhost:11434/api/generate"
    prompt = "Explain Docker containerization in 100 words."
    
    print(f"\nBenchmarking {model_name} via Ollama...")
    
    start_ttft = time.time()
    try:
        response = requests.post(url, json={
            "model": model_name,
            "prompt": prompt,
            "stream": True
        }, timeout=30)
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        return None

    first_token_time = None
    token_count = 0
    start_gen = None
    
    for line in response.iter_lines():
        if line:
            chunk = json.loads(line)
            if first_token_time is None:
                first_token_time = time.time() - start_ttft
                start_gen = time.time()
            token_count += 1
            if chunk.get("done"):
                break
    
    total_gen_time = time.time() - start_gen if start_gen else 0
    tps = token_count / total_gen_time if total_gen_time > 0 else 0
    
    results = {
        "model": model_name,
        "ttft_ms": round(first_token_time * 1000, 2) if first_token_time else 0,
        "tokens_per_sec": round(tps, 2),
        "total_tokens": token_count
    }
    
    print(f"Results: {results}")
    return results

def main():
    # Attempt to benchmark both the baseline and the fine-tuned model
    models = ["phi3", "devops-assistant"]
    all_results = []
    
    for model in models:
        res = benchmark_ollama(model)
        if res:
            all_results.append(res)
            
    if all_results:
        with open("inference/benchmark_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print("\nBenchmarks saved to inference/benchmark_results.json")

if __name__ == "__main__":
    main()
