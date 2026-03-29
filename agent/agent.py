import requests # type: ignore
import re
import time
from typing import List, Dict
from transformers import AutoTokenizer
from logs.logger import log_request

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "devops-assistant"

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3-mini-4k-instruct")

class ReActAgent:
    def __init__(self, model_name: str = MODEL, max_steps: int = 5):
        self.model_name = model_name
        self.max_steps = max_steps
        self.memory: List[Dict[str, str]] = []
        self.api_url, self.tools = self._discover_tools()

    def _discover_tools(self):
        for host in ["http://127.0.0.1:8000", "http://localhost:8000"]:
            try:
                resp = requests.get(f"{host}/tools", timeout=3)
                if resp.status_code == 200:
                    print(f"Discovered tools from {host}")
                    return host, resp.json()["tools"]
            except Exception:
                pass
        print("Tool discovery failed. Using local fallback.")
        return "http://127.0.0.1:8000", [
            {"name": "devops_search", "description": "Search DevOps documentation."},
            {"name": "run_python", "description": "Execute Python for math/logic."},
            {"name": "system_status", "description": "Check if systems are healthy."}
        ]

    def _build_plan(self, query: str) -> List[Dict]:
        plan = []
        q = query.lower()

        if any(w in q for w in ["healthy", "health", "status", "up", "running", "uptime"]):
            plan.append({"tool": "system_status", "input": "check"})

        devops_keywords = [
            "docker", "kubernetes", "k8s", "ci/cd", "terraform", "ansible", 
            "jenkins", "nodes", "container", "pipeline", "microservice",
            "cpu", "memory", "limit", "resource", "pod", "deployment", "config",
            "gitops", "git-ops"
        ]
        if any(w in q for w in devops_keywords):
            search_q = re.sub(r"(is the system healthy\?|if so,?|tell me|what is|how many|calculate|5 times|50%|half of)", "", query, flags=re.IGNORECASE).strip()
            search_q = re.sub(r"^[^\w\s]|[^\w\s]$|^\s*(s|find|the)\s+", "", search_q, flags=re.IGNORECASE).strip()
            plan.append({"tool": "devops_search", "input": search_q or query})
        math_patterns = [
            r"(\d+)\s*([\+\-\*\/])\s*(\d+)",
            r"(\d+)\s*(times|x|\*)\s*(.*)",
            r"multiply\s*(.*)\s*by\s*(\d+)",
            r"calculate\s*(\d+)%\s*of\s*(.*)",
            r"(\d+)%\s*of\s*(.*)",
            r"half\s*of\s*(.*)"
        ]
        
        for pat in math_patterns:
            match = re.search(pat, q)
            if match:
                if "%" in pat or "half" in pat:
                    val = "1000" if "cpu" in q else "100"
                    perc = match.group(1) if "%" in pat else "50"
                    plan.append({"tool": "run_python", "input": f"print({val} * {int(perc)/100})"})
                else:
                    v1, op, v2 = match.groups()
                    op = "*" if op.lower() in ["times", "x"] else op
                    plan.append({"tool": "run_python", "input": f"print({v1} {op} {v2})"})
                break

        return plan

    def _execute_tool(self, tool: str, input_str: str) -> str:
        print(f"Action: {tool} | Input: {input_str[:60]}...")
        try:
            resp = requests.post(f"{self.api_url}/tool", json={
                "tool": tool.lower().strip(),
                "input": input_str.strip()
            }, timeout=15)
            return resp.json().get("output", "Empty output")
        except Exception as e:
            return f"Tool error: {e}"

    def _synthesize(self, query: str, observations: List[Dict]) -> str:
        obs_text = "\n".join([f"- {o['tool']}: {o['result']}" for o in observations])
        full_context = "\n".join([f"User: {t['user']}\nAssistant: {t['assistant']}" for t in self.memory[-3:]])

        prompt = f"""Context from tools:
{obs_text}

{f"Previous Conversation:{chr(10)}{full_context}" if full_context else ""}

User Question: {query}
Answer all parts of the question concisely using the provided context:"""

        self.last_input_tokens = len(tokenizer.encode(prompt))
        
        try:
            resp = requests.post(OLLAMA_URL, json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0}
            }, timeout=45)
            answer = resp.json().get("response", "").strip()
            # Cleanly truncate any hallucinated metadata or EOS tokens
            answer = answer.split("</s>")[0].split("<|end|>")[0].split("Instruction")[0].strip()
            self.last_output_tokens = len(tokenizer.encode(answer))
            
            return answer
        except Exception as e:
            return f"Error in synthesis: {str(e)}"

    def solve(self, query: str) -> str:
        start_time = time.time()
        print(f"\nQuery: {query}")

        plan = self._build_plan(query)
        print(f"Plan: {[p['tool'] for p in plan]}")

        observations = []
        tools_invoked = []
        for step in plan:
            result = self._execute_tool(step["tool"], step["input"])
            observations.append({"tool": step["tool"], "result": result})
            tools_invoked.append(step["tool"])
            print(f"Observation: {result[:80]}...")

        answer = self._synthesize(query, observations)
        
        latency_ms = int((time.time() - start_time) * 1000)
        log_request(
            query=query,
            tools=tools_invoked,
            latency_ms=latency_ms,
            input_tokens=getattr(self, "last_input_tokens", 0),
            output_tokens=getattr(self, "last_output_tokens", 0)
        )
        
        self.memory.append({"user": query, "assistant": answer})
        return answer