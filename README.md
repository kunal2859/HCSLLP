# DevOps AI Assistant 

[![CI/CD Pipeline](https://github.com/kunalaggarwal/HCSLLP/actions/workflows/ci.yml/badge.svg)](https://github.com/kunalaggarwal/HCSLLP/actions/workflows/ci.yml)

A local, agentic AI assistant powered by **Phi-3-mini** and **ChromaDB**, optimized for Apple Silicon (MPS). It uses a ReAct pattern to answer DevOps questions by searching a specialized knowledge base or executing Python code.

## Features
- **MPS Acceleration**: Fine-tuned on Apple Silicon for 50x faster training.
- **Zero-Touch Sync**: Automatically re-indexes its knowledge base when you update the data.
- **Secure Sandbox**: Executes Python snippets in isolated subprocesses with timeouts.
- **Tool-Enabled**: Can choose between semantic search, math execution, or general answering.

---

## System Requirements
- **Hardware**: Mac with M1/M2/M3 chip (16GB RAM recommended).
- **OS**: macOS (for MPS acceleration).
- **Software**: 
  - Python 3.10+
  - [Ollama](https://ollama.com/) running with the `phi3` model.

---

## Required GitHub Secrets
Configure these under **Settings → Secrets → Actions** in your repository:

| Secret | Description |
|---|---|
| `GHCR_TOKEN` | GitHub personal access token with `write:packages` scope |
| `DEPLOY_HOST` | IP or hostname of your deployment server |
| `DEPLOY_USER` | SSH username on the deployment server |
| `DEPLOY_KEY` | SSH private key (contents of `~/.ssh/id_rsa`) |

---

## Setup Instructions

1. **Clone & Virtual Env**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Start the API Server**:
   ```bash
   uvicorn mcp_server.api:app --reload
   ```

3. **Start the Assistant**:
   ```bash
   python3 -m app.main
   ```

4. **Update Knowledge**:
   - Simply add new Q&A pairs to `data/raw/devops_qa.json`.
   - The bot will **automatically re-index** the next time it starts!

---

## Known Limitations
- **Ollama Dependency**: The agent requires a local Ollama instance running at `localhost:11434`.
- **Reasoning Leak**: Small models like Phi-3 can occasionally leak internal thoughts (e.g., `ACTION: ...`) into the final output. The `agent/agent.py` contains a cleanup function to mitigate this.
- **Single-Turn Memory**: The short context window means the bot only remembers the last ~3 interactions clearly.

---

## Future Improvements
If I had more time, I would prioritize:
### **Large-Scale Fine-Tuning**
Migration from **Phi-3-mini** to a larger model like **Llama-3-70B** or **Mistral-Large**, fine-tuned on a significantly larger and more diverse DevOps dataset (Terraform, Cloud-native, Security logs). This would improve complex reasoning and reduce the need for strict output filtering.