from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "microsoft/phi-3-mini-4k-instruct"
lora_path = "models/lora_adapter"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype="auto"
)

model = PeftModel.from_pretrained(model, lora_path)
model = model.merge_and_unload()

print("Model loaded successfully!")