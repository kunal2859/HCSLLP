import json

with open("data/raw/devops_qa.json") as f:
    data = json.load(f)

augmented = []
for item in data:
    augmented.append(item)
    augmented.append({
        "instruction": "Explain " + item["instruction"].lower(),
        "input": "",
        "output": item["output"]
    })

data = (data + augmented)[:200]

with open("data/raw/devops_qa.json", "w") as f:
    json.dump(data, f, indent=2)