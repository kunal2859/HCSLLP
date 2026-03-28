from model_loader import model, tokenizer

def chat(query):
    prompt = f"<s>[INST] {query} [/INST]"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "[/INST]" in response:
        response = response.split("[/INST]")[-1].strip()
    
    return response

while True:
    q = input("You: ")
    print("Bot:", chat(q))