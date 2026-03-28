import json
import random
from sklearn.model_selection import train_test_split

with open("data/raw/devops_qa.json", "r") as f:
    data = json.load(f)

print(f"Total raw samples: {len(data)}")

seen = set()
cleaned_data = []
for item in data:
    if item['instruction'] not in seen:
        seen.add(item['instruction'])
        item['instruction'] = item['instruction'].strip()
        item['output'] = item['output'].strip()
        cleaned_data.append(item)

print(f"Cleaned samples (duplicates removed): {len(cleaned_data)}")

random.shuffle(cleaned_data)

train_data, val_data = train_test_split(
    cleaned_data,
    test_size=0.2,
    random_state=42
)

print(f"Train size: {len(train_data)}")
print(f"Validation size: {len(val_data)}")

with open("data/processed/train.json", "w") as f:
    json.dump(train_data, f, indent=2)

with open("data/processed/val.json", "w") as f:
    json.dump(val_data, f, indent=2)

print("Preprocessing completed!")