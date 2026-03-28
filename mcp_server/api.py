from fastapi import FastAPI
from pydantic import BaseModel
from mcp_server.vector_store import search_documents
import time
import subprocess

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/tools")
def get_tools():
    return {
        "tools": [
            {"name": "devops_search", "description": "Semantic search over DevOps docs."},
            {"name": "run_python", "description": "Execute Python code for math or logic."},
            {"name": "system_status", "description": "Check if systems are healthy."}
        ]
    }

class ToolRequest(BaseModel):
    tool: str
    input: str

def devops_search(query):
    try:
        results = search_documents(query, top_k=3)
        return " ".join(results)
    except Exception as e:
        return f"Search error: {str(e)}"

def run_python(code):
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True,
            text=True,
            timeout=3
        )
        if result.stderr:
            return f"Error: {result.stderr.strip()}"
        return result.stdout.strip() or "Success (no output)"
    except subprocess.TimeoutExpired:
        return "Error: Execution timeout"
    except Exception as e:
        return f"Error: {str(e)}"

@app.post("/tool")
def call_tool(req: ToolRequest):
    start = time.time()
    if req.tool in ["devops_search", "search_documents"]:
        result = devops_search(req.input)
    elif req.tool == "run_python":
        result = run_python(req.input)
    elif req.tool == "system_status":
        result = "Healthy"
    else:
        result = f"Invalid tool: {req.tool}"

    latency = int((time.time() - start) * 1000)

    return {
        "tool": req.tool,
        "input": req.input,
        "output": result,
        "latency_ms": latency
    }