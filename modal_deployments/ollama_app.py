# inspired by https://github.com/irfansharif/ollama-modal/blob/master/ollama-modal.py

import modal
import os
import subprocess
import time
import json
from typing import Optional

from modal import build, enter, method
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL = os.environ.get("MODEL", "llama3:instruct")

def pull(model: str = MODEL):
    subprocess.run(["systemctl", "daemon-reload"])
    subprocess.run(["systemctl", "enable", "ollama"])
    subprocess.run(["systemctl", "start", "ollama"])
    time.sleep(2)  # wait for service to start
    subprocess.run(["ollama", "pull", model], stdout=subprocess.PIPE)

image = (
    modal.Image.debian_slim()
    .apt_install("curl", "systemctl")
    .run_commands(
        "curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz",
        "tar -C /usr -xzf ollama-linux-amd64.tgz",
        "useradd -r -s /bin/false -U -m -d /usr/share/ollama ollama",
        "usermod -a -G ollama $(whoami)",
    )
    .add_local_file("ollama.service", "/etc/systemd/system/ollama.service", copy=True)
    .pip_install("ollama", "fastapi", "pydantic")
    .run_function(pull)
)

app = modal.App(name="ollama", image=image)

# Request/Response models for OpenAI compatibility
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 150
    stream: Optional[bool] = False

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list
    usage: dict

with image.imports():
    import ollama

@app.cls(gpu="a10g", scaledown_window=300)
class Ollama:
    @enter()
    def load(self):
        subprocess.run(["systemctl", "start", "ollama"])
        # Wait for ollama to be ready
        time.sleep(3)

    @method()
    def infer(self, text: str, verbose: bool = False):
        """Original inference method for Modal remote calls"""
        stream = ollama.chat(
            model=MODEL, messages=[{"role": "user", "content": text}], stream=True
        )
        for chunk in stream:
            yield chunk["message"]["content"]
            if verbose:
                print(chunk["message"]["content"], end="", flush=True)
        return

    @method()
    def chat_completion(self, messages: list[dict], temperature: float = 0.7, max_tokens: int = 150):
        """OpenAI-compatible chat completion method"""
        try:
            response = ollama.chat(
                model=MODEL, 
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            )
            return response
        except Exception as e:
            raise Exception(f"Ollama inference failed: {str(e)}")

# Create FastAPI app for web endpoints
web_app = FastAPI()

@web_app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint"""
    try:
        ollama_instance = Ollama()
        
        # Convert Pydantic messages to dict format
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Call the ollama model
        response = ollama_instance.chat_completion.remote(
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        # Format response in OpenAI format
        import time
        
        formatted_response = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response["message"]["content"]
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,  # Ollama doesn't provide token counts
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
        
        return formatted_response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@web_app.get("/v1/models")
async def list_models():
    """OpenAI-compatible models endpoint"""
    return {
        "object": "list",
        "data": [{
            "id": MODEL,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "modal"
        }]
    }

@web_app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": MODEL}

# Deploy the FastAPI app
@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    return web_app

# Original local entrypoint for testing
@app.local_entrypoint()
def main(text: str = "Why is the sky blue?", lookup: bool = False):
    if lookup:
        ollama = modal.Cls.lookup("ollama", "Ollama")
    else:
        ollama = Ollama()
    for chunk in ollama.infer.remote(text, verbose=False):
        print(chunk, end="", flush=False) 