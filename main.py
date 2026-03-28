import subprocess
import time

def start_services():
    print("Starting DevOps AI Assistant Services...")
    
    print("Starting Tool API (Port 8000)...")
    tool_api = subprocess.Popen(
        ["python3", "-m", "uvicorn", "mcp_server.api:app", "--port", "8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    time.sleep(1) 
    
    print("Waiting for Tool API to be ready (load embeddings)...")
    import requests # type: ignore
    for i in range(20): 
        try:
            resp = requests.get("http://127.0.0.1:8000/health", timeout=1)
            if resp.status_code == 200:
                print("ool API is ready!")
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        print("Tool API timed out during startup.")

    try:
        from agent.agent import ReActAgent
        agent = ReActAgent()
        print("DevOps AI Assistant Ready (type 'exit' to quit)")
        
        while True:
            user_input = input("\nYou: ")
            if user_input.lower().strip() in ["exit", "quit", "q"]:
                break
            
            answer = agent.solve(user_input)
            print(f"\nBot: {answer}")
            print("-" * 40)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        tool_api.terminate()

if __name__ == "__main__":
    start_services()
