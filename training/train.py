import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

def load_data(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

train_data = load_data("data/processed/train.json")
val_data = load_data("data/processed/val.json")

def format_data(data):
    formatted = []
    for item in data:
        text = f"<s>[INST] {item['instruction']} [/INST] {item['output']}</s>"
        formatted.append({"text": text})
    return formatted

train_dataset = Dataset.from_list(format_data(train_data))
val_dataset = Dataset.from_list(format_data(val_data))

model_name = "microsoft/phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16 if device == "mps" else torch.float32
)
model.to(device)

def tokenize(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    
    tokens["labels"] = tokens["input_ids"].copy()
    
    return tokens

train_dataset = train_dataset.map(tokenize)
val_dataset = val_dataset.map(tokenize)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir="models/lora_adapter",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=50,
    gradient_accumulation_steps=4,
    save_total_limit=2,
    learning_rate=2e-4,
    fp16=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

model.save_pretrained("models/lora_adapter")
tokenizer.save_pretrained("models/lora_adapter")

print("Training completed and model saved!")