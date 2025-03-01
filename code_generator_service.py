# code_generator_service.py
import os
import json
import logging
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks
from pydantic import BaseModel
import uvicorn
import asyncio
import httpx
import time
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("code-generator")

# FastAPI app
app = FastAPI(title="Code Generator Service")

# Configuration
DEEPSEEK_API_URL = os.environ.get("DEEPSEEK_API_URL", "http://localhost:8080/v1/chat/completions")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
TIMEOUT_SECONDS = int(os.environ.get("TIMEOUT_SECONDS", "120"))

# In-memory transaction log for debugging
transaction_log = []

# Models
class Message(BaseModel):
    role: str
    content: str

class GenerationRequest(BaseModel):
    current_code: str
    performance_data: Dict[str, Any]
    bottlenecks: List[Dict[str, Any]]
    optimization_strategy: Dict[str, Any]
    optimization_history: Optional[List[Dict[str, Any]]] = None
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 4096

class ChatCompletionRequest(BaseModel):
    model: str = "deepseek-r-1"
    messages: List[Message]
    temperature: float = 0.2
    max_tokens: int = 4096
    stream: bool = False

class GenerationResponse(BaseModel):
    optimized_code: str
    reasoning: str
    estimated_improvements: Dict[str, Any]

# Client for calling DeepSeek API
@asynccontextmanager
async def get_client():
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
        yield client

async def call_deepseek_api(client: httpx.AsyncClient, request: ChatCompletionRequest) -> Dict[str, Any]:
    """Call the DeepSeek API with retries."""
    headers = {"Content-Type": "application/json"}
    if DEEPSEEK_API_KEY:
        headers["Authorization"] = f"Bearer {DEEPSEEK_API_KEY}"
    
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.post(
                DEEPSEEK_API_URL,
                headers=headers,
                json=request.dict()
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"API call failed (attempt {attempt+1}/{MAX_RETRIES}): {str(e)}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error(f"All {MAX_RETRIES} attempts failed")
                raise HTTPException(status_code=500, detail=f"Failed to call DeepSeek API: {str(e)}")

def create_optimization_prompt(request: GenerationRequest) -> str:
    """Create a well-structured prompt for code optimization."""
    
    # Extract key information
    code = request.current_code
    perf_data = json.dumps(request.performance_data, indent=2)
    bottlenecks = json.dumps(request.bottlenecks, indent=2)
    strategy = json.dumps(request.optimization_strategy, indent=2)
    
    # Format optimization history if available
    history_text = ""
    if request.optimization_history and len(request.optimization_history) > 0:
        # Include only the latest 3 iterations to avoid prompt getting too long
        recent_history = request.optimization_history[-3:]
        history_text = "## Recent Optimization History\n"
        for item in recent_history:
            history_text += f"- Iteration {item.get('iteration', 'unknown')}: {item.get('code_diff_summary', 'unknown')}\n"
            if 'bottlenecks' in item and len(item['bottlenecks']) > 0:
                history_text += f"  - Addressed: {item['bottlenecks'][0].get('name', 'unknown bottleneck')}\n"
    
    prompt = f"""# GPU Fluid Simulation Optimization Task

You are an expert GPU programmer specializing in fluid simulation optimization. Your task is to improve the performance of the provided GPU code based on the identified bottlenecks and optimization strategy.

## Current Performance Metrics
```json
{perf_data}
```

## Identified Bottlenecks
```json
{bottlenecks}
```

## Optimization Strategy to Implement
```json
{strategy}
```

{history_text}

## Current Code
```
{code}
```

## Your Task:

1. Analyze the code and identified bottlenecks
2. Implement the specified optimization strategy
3. Focus on making targeted changes to address the primary bottleneck
4. Add clear comments explaining your optimization reasoning
5. Return the COMPLETE optimized code, do not omit any sections

Your response should follow this format:
1. A detailed explanation of your optimization approach
2. The complete optimized code
3. Expected performance improvements
4. Any trade-offs made

IMPORTANT: Make sure to provide the full code implementation.
"""
    return prompt

@app.post("/optimize", response_model=GenerationResponse)
async def optimize_code(request: GenerationRequest, background_tasks: BackgroundTasks):
    """Endpoint to generate optimized code based on performance data and bottlenecks."""
    logger.info("Received optimization request")
    
    # Create the optimization prompt
    prompt = create_optimization_prompt(request)
    
    # Format the request for DeepSeek API
    chat_request = ChatCompletionRequest(
        model="deepseek-r-1",
        messages=[
            Message(role="system", content="You are an expert GPU programmer specializing in performance optimization for fluid simulations. Provide detailed, accurate, and efficient code optimizations."),
            Message(role="user", content=prompt)
        ],
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )
    
    # Call DeepSeek API
    start_time = time.time()
    async with get_client() as client:
        result = await call_deepseek_api(client, chat_request)
    elapsed_time = time.time() - start_time
    
    # Extract the generated text
    generation = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    
    # Log the transaction
    transaction = {
        "timestamp": time.time(),
        "request": request.dict(),
        "response_time_seconds": elapsed_time,
        "generation_length": len(generation)
    }
    background_tasks.add_task(lambda: transaction_log.append(transaction))
    
    # Extract code and reasoning
    try:
        # Try to extract code blocks from the response
        code_section = extract_code_section(generation)
        reasoning_section = extract_reasoning_section(generation)
        improvements_section = extract_improvements_section(generation)
    except Exception as e:
        logger.error(f"Error extracting sections: {str(e)}")
        # If extraction fails, use some heuristics to separate code and text
        code_section = generation
        reasoning_section = "See optimized code"
        improvements_section = {"general": "Performance improvement expected"}
    
    return GenerationResponse(
        optimized_code=code_section,
        reasoning=reasoning_section,
        estimated_improvements=improvements_section
    )

def extract_code_section(text: str) -> str:
    """Extract code blocks from generated text."""
    import re
    
    # First try to extract code blocks
    code_blocks = re.findall(r'```(?:\w*\n)?(.*?)```', text, re.DOTALL)
    
    if code_blocks:
        # Find the largest code block (likely the complete implementation)
        return max(code_blocks, key=len)
    
    # If no code blocks found, try to extract indented sections
    indented_blocks = re.findall(r'(?:^|\n)( {4,}.*(?:\n {4,}.*)*)', text)
    if indented_blocks:
        return max(indented_blocks, key=len)
    
    # If all else fails, return the original text with a warning
    logger.warning("Could not extract code blocks from generation")
    return text

def extract_reasoning_section(text: str) -> str:
    """Extract the reasoning section from generated text."""
    # Look for sections that might contain reasoning
    indicators = [
        "approach", "reasoning", "explanation", "rationale", 
        "optimization", "changes", "improvements"
    ]
    
    lines = text.split('\n')
    reasoning_lines = []
    in_reasoning_section = False
    
    for line in lines:
        # Check if this line indicates the start of a reasoning section
        lower_line = line.lower()
        if any(ind in lower_line for ind in indicators) and not line.strip().startswith('```'):
            in_reasoning_section = True
            reasoning_lines.append(line)
        elif in_reasoning_section:
            # End the reasoning section if we hit a code block or another section
            if line.strip().startswith('```') or (line.strip().startswith('#') and len(line.strip()) > 2):
                in_reasoning_section = False
            else:
                reasoning_lines.append(line)
    
    if reasoning_lines:
        return '\n'.join(reasoning_lines)
    else:
        return "Reasoning not explicitly provided in the generation."

def extract_improvements_section(text: str) -> Dict[str, Any]:
    """Extract expected improvements information."""
    improvements = {}
    
    # Look for performance improvement claims
    import re
    fps_improvements = re.findall(r'(\d+(?:\.\d+)?)(?:%|\s*percent)?\s*(?:increase|improvement|higher|better)\s*(?:in|to)?\s*fps', text.lower())
    if fps_improvements:
        try:
            improvements["fps_increase_percentage"] = float(fps_improvements[0])
        except ValueError:
            improvements["fps_increase_percentage"] = "mentioned but not quantified"
    
    # Look for other performance indicators
    if "memory" in text.lower() or "bandwidth" in text.lower():
        improvements["memory_optimization"] = True
    
    if "occupancy" in text.lower():
        improvements["improved_occupancy"] = True
    
    if "parallel" in text.lower() or "concurren" in text.lower():
        improvements["improved_parallelism"] = True
    
    if not improvements:
        improvements["general"] = "Unspecified performance improvements expected"
    
    return improvements

# For debugging
@app.get("/status")
async def status():
    """Get service status."""
    return {
        "status": "running",
        "transactions": len(transaction_log),
        "last_transaction": transaction_log[-1] if transaction_log else None
    }

# Direct DeepSeek API proxy endpoint for testing
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Proxy endpoint to test direct access to DeepSeek API."""
    body = await request.json()
    
    async with get_client() as client:
        headers = {"Content-Type": "application/json"}
        if DEEPSEEK_API_KEY:
            headers["Authorization"] = f"Bearer {DEEPSEEK_API_KEY}"
            
        response = await client.post(
            DEEPSEEK_API_URL,
            headers=headers,
            json=body
        )
        
        return response.json()

if __name__ == "__main__":
    uvicorn.run("code_generator_service:app", host="0.0.0.0", port=8000, reload=True)