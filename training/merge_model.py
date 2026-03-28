import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_and_save():
    base_model_name = "microsoft/phi-3-mini-4k-instruct"
    adapter_path = "models/lora_adapter"
    save_path = "models/merged_model"

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    print("Loading adapter weights...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("Merging weights (this might take a moment)...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to {save_path}...")
    merged_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print("Merge complete!")
    print("\nNote: To convert to GGUF, we would typically use llama.cpp's convert.py:")
    print(f"python3 llama.cpp/convert.py {save_path} --outfile {save_path}/model.gguf")

if __name__ == "__main__":
    merge_and_save()
