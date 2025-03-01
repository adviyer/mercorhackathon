# Dockerfile for Code Generator Service
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=3000

# Create app directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir fastapi uvicorn httpx pydantic

# Create the code_generator_service.py file
RUN echo '
import os
import json
import logging
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel
import uvicorn
import httpx
import time
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("code-generator")

# FastAPI app
app = FastAPI(title="Code Generator Service")

# Configuration
DEEPSEEK_API_URL = os.environ.get("DEEPSEEK_API_URL", "https://p01--deepseek-ollama--8bsw8fx29k5g.code.run:11434/v1/chat/completions")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
TIMEOUT_SECONDS = int(os.environ.get("TIMEOUT_SECONDS", "120"))

# Simple test endpoint
@app.get("/")
async def root():
    return {"message": "Code Generator service is running"}

# Health check endpoint
@app.get("/status")
async def status():
    return {"status": "running", "timestamp": time.time()}

# Model definitions
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    model: Optional[str] = "deepseek-r-1"
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 4096

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """Simple proxy to DeepSeek API for testing."""
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            headers = {"Content-Type": "application/json"}
            if DEEPSEEK_API_KEY:
                headers["Authorization"] = f"Bearer {DEEPSEEK_API_KEY}"
                
            response = await client.post(
                DEEPSEEK_API_URL,
                headers=headers,
                json=request.dict()
            )
            
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Error calling DeepSeek API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"API call failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("code_generator_service:app", host="0.0.0.0", port=int(os.environ.get("PORT", 3000)))
' > /app/code_generator_service.py

# Create a simple Node.js entrypoint
RUN echo '
// This file exists only for compatibility with Node.js deployment systems
// It will launch the Python FastAPI application
const { spawn } = require("child_process");
console.log("Starting Python FastAPI application...");

const pythonProcess = spawn("python3", ["-m", "uvicorn", "code_generator_service:app", "--host", "0.0.0.0", "--port", process.env.PORT || 3000]);

pythonProcess.stdout.on("data", (data) => {
  console.log(`${data}`);
});

pythonProcess.stderr.on("data", (data) => {
  console.error(`${data}`);
});

pythonProcess.on("close", (code) => {
  console.log(`Python process exited with code ${code}`);
  process.exit(code);
});
' > /app/server.js

# Expose port
EXPOSE 3000

# Set environment variables
ENV DEEPSEEK_API_URL="https://p01--deepseek-ollama--8bsw8fx29k5g.code.run:11434/v1/chat/completions"
ENV MAX_RETRIES=3
ENV TIMEOUT_SECONDS=120

# Create a package.json file for Node.js compatibility
RUN echo '{"name":"code-generator-service","version":"1.0.0","scripts":{"start":"node server.js"}}' > /app/package.json

# Start command - this allows both Node.js and Python
CMD ["node", "server.js"]