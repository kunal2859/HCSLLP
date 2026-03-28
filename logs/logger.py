import json
import os
from datetime import datetime

LOG_FILE = "logs/requests.json"

def log_request(query: str, tools: list, latency_ms: int, input_tokens: int, output_tokens: int):
    if not os.path.exists("logs"):
        os.makedirs("logs")
        
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "tools_invoked": tools,
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens
    }
    
    try:
        logs = []
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r") as f:
                try:
                    logs = json.load(f)
                except json.JSONDecodeError:
                    logs = []
        
        logs.append(log_entry)
        
        with open(LOG_FILE, "w") as f:
            json.dump(logs, f, indent=2)
            
    except Exception as e:
        print(f"Logging failed: {e}")