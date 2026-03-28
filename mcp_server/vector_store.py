import json
import os
import hashlib
from sentence_transformers import SentenceTransformer
import chromadb

DATA_PATH = "data/raw/devops_qa.json"
CHROMA_PATH = "./chroma_db"
HASH_FILE = os.path.join(CHROMA_PATH, "data_hash.txt")

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection("devops_docs")

def get_data_hash():
    if not os.path.exists(DATA_PATH):
        return None
    with open(DATA_PATH, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def populate_db(force=False):
    global collection
    
    if not os.path.exists(DATA_PATH):
        print(f"Warning: {DATA_PATH} not found.")
        return

    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    documents = list(set([
        f"Q: {item['instruction']} A: {item['output']}".strip()
        for item in data
    ]))

    if force:
        print("Clearing old collection for re-indexing...")
        try:
            client.delete_collection("devops_docs")
        except Exception:
            pass
        collection = client.create_collection("devops_docs")

    if collection.count() == 0:
        print(f"Indexing {len(documents)} unique documents...")
        embeddings = model.encode(documents).tolist()
        collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=[str(i) for i in range(len(documents))]
        )
        
        current_hash = get_data_hash()
        with open(HASH_FILE, "w") as f:
            f.write(current_hash)
        print("Data indexed successfully!")

def auto_sync():
    current_hash = get_data_hash()
    if not current_hash:
        return

    last_hash = None
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE, "r") as f:
            last_hash = f.read().strip()

    if current_hash != last_hash:
        print("Change detected in devops_qa.json! Starting auto-sync...")
        populate_db(force=True)
    else:
        populate_db(force=False)

def search_documents(query, top_k=3):
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    return results["documents"][0]

auto_sync()