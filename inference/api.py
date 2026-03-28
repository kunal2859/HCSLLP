from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from llama_cpp import Llama
import json
import os

app = FastAPI()

MODEL_PATH = os.getenv("MODEL_PATH", "models/merged_model/phi3-mini-q4_k_m.gguf")
llm = None

def get_llm():
    global llm
    if llm is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(status_code=500, detail=f"Model not found at {MODEL_PATH}")
        llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_gpu_layers=-1) # -1 for full GPU
    return llm

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = False

@app.get("/health")
def health():
    return {"status": "ready", "model": MODEL_PATH}

@app.post("/generate")
def generate(request: GenerateRequest):
    model = get_llm()
    
    if request.stream:
        def stream_generator():
            stream = model(
                request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=True
            )
            for chunk in stream:
                yield f"data: {json.dumps(chunk)}\n\n"
        
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    
    else:
        output = model(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        return output