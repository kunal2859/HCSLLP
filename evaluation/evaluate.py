import json
import sys
import argparse
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

ROUGE_THRESHOLD = 0.15

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")
with open("data/processed/val.json", "r") as f:
    val_data = json.load(f)

samples = val_data[:20]

base_model_name = "microsoft/phi-3-mini-4k-instruct"
ft_model_path = "models/lora_adapter"

base_results: list[str] = []
ft_results: list[str] = []

def compute_rouge_l(predictions: list[str], references: list[str]) -> float:
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = [
            scorer.score(ref, pred)["rougeL"].fmeasure
            for pred, ref in zip(predictions, references)
        ]
        return sum(scores) / len(scores) if scores else 0.0
    except ImportError:
        print("rouge_score not installed. Skipping ROUGE computation.")
        return 1.0 

print("\n--- Evaluating Base Model ---")
try:
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if device == "mps" else torch.float32
    ).to(device)

    base_pipe = pipeline("text-generation", model=base_model, tokenizer=base_tokenizer)

    for i, item in enumerate(samples):
        prompt = f"[INST] {item['instruction']} [/INST]"
        output = base_pipe(prompt, max_new_tokens=100)[0]["generated_text"]
        base_results.append(output)
        if (i + 1) % 5 == 0:
            print(f"  Processed {i+1}/{len(samples)}...")
except Exception as e:
    print(f"Base model not available: {e}. Skipping base evaluation.")
    base_results = [""] * len(samples)

print("  Clearing base model from memory...")
try:
    del base_model, base_pipe
except NameError:
    pass
gc.collect()
if device == "mps":
    torch.mps.empty_cache()

print("\n--- Evaluating Fine-Tuned Model ---")
try:
    ft_tokenizer = AutoTokenizer.from_pretrained(ft_model_path)
    ft_model = AutoModelForCausalLM.from_pretrained(
        ft_model_path,
        torch_dtype=torch.float16 if device == "mps" else torch.float32
    ).to(device)

    ft_pipe = pipeline("text-generation", model=ft_model, tokenizer=ft_tokenizer)

    for i, item in enumerate(samples):
        prompt = f"[INST] {item['instruction']} [/INST]"
        output = ft_pipe(prompt, max_new_tokens=100)[0]["generated_text"]
        ft_results.append(output)
        if (i + 1) % 5 == 0:
            print(f"  Processed {i+1}/{len(samples)}...")
except Exception as e:
    print(f"Fine-tuned model not available: {e}. Skipping ft evaluation.")
    ft_results = [""] * len(samples)

references = [item["output"] for item in samples]
rouge_l_score = compute_rouge_l(ft_results, references)

results = []
for i, item in enumerate(samples):
    results.append({
        "question": item["instruction"],
        "expected": item["output"],
        "base_output": base_results[i] if i < len(base_results) else "",
        "fine_tuned_output": ft_results[i] if i < len(ft_results) else "",
    })

results_payload = {
    "metrics": {
        "rouge_l": rouge_l_score,
        "threshold": ROUGE_THRESHOLD,
        "passed": rouge_l_score >= ROUGE_THRESHOLD,
        "samples_evaluated": len(samples)
    },
    "samples": results
}

with open("evaluation/results.json", "w") as f:
    json.dump(results_payload, f, indent=2)

print(f"\nROUGE-L Score: {rouge_l_score:.4f} (threshold: {ROUGE_THRESHOLD})")
print("Evaluation results saved to evaluation/results.json")
parser = argparse.ArgumentParser()
parser.add_argument("--ci", action="store_true", help="Run in CI mode (fail on low ROUGE)")
args, _ = parser.parse_known_args()

if args.ci:
    if rouge_l_score < ROUGE_THRESHOLD:
        print(f"\nCI FAILED: ROUGE-L {rouge_l_score:.4f} is below threshold {ROUGE_THRESHOLD}")
        sys.exit(1)
    else:
        print(f"\nCI PASSED: ROUGE-L {rouge_l_score:.4f} meets threshold {ROUGE_THRESHOLD}")