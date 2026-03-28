import time
import subprocess
import traceback
from datetime import datetime
from mcp.server.fastmcp import FastMCP
from mcp_server.vector_store import search_documents

mcp = FastMCP("DevOps Assistant")

@mcp.tool(
    name="search_documents",
    description="Semantic search over DevOps dataset to find relevant answers.",
    schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The question or topic to search for"},
            "top_k": {"type": "integer", "description": "Number of results to return", "default": 3}
        },
        "required": ["query"]
    }
)
def search_documents_tool(query: str, top_k: int = 3):
    """Perform semantic search over DevOps knowledge base."""
    try:
        start = time.time()
        results = search_documents(query, top_k=top_k)
        latency = int((time.time() - start) * 1000)

        return {
            "status": "success",
            "results": results,
            "latency_ms": latency
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@mcp.tool(
    name="run_python",
    description="Execute Python code snippets safely in a subprocess.",
    schema={
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "The Python code to execute"}
        },
        "required": ["code"]
    }
)
def run_python_tool(code: str):
    """Execute Python safely using subprocess."""
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True,
            text=True,
            timeout=3
        )

        return {
            "status": "success",
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "exit_code": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "error": "Execution timeout (3s limit reached)"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@mcp.tool(
    name="explain_devops",
    description="Explain a DevOps term using the knowledge base.",
    schema={
        "type": "object",
        "properties": {
            "term": {"type": "string", "description": "The DevOps term to explain"}
        },
        "required": ["term"]
    }
)
def explain_devops_tool(term: str):
    """Provide explanation using semantic retrieval."""
    try:
        results = search_documents(term, top_k=2)
        return {
            "status": "success",
            "term": term,
            "explanation": results
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@mcp.tool(
    name="get_date_context",
    description="Get the current date and time for temporal context.",
    schema={"type": "object", "properties": {}}
)
def get_date_context_tool():
    """Returns the current date and time."""
    return {
        "status": "success",
        "datetime": str(datetime.now()),
        "iso_format": datetime.now().isoformat()
    }

@mcp.tool(
    name="system_status",
    description="Check the health of the MCP server.",
    schema={"type": "object", "properties": {}}
)
def system_status_tool():
    """Returns system health info."""
    return {
        "status": "success",
        "message": "DevOps MCP Server is healthy and running 🚀",
        "timestamp": time.time()
    }
