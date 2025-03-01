# evaluator_service.py
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
logger = logging.getLogger("evaluator")

# FastAPI app
app = FastAPI(title="Performance Evaluator Service")

# Configuration
DEEPSEEK_API_URL = os.environ.get("DEEPSEEK_API_URL", "http://localhost:8080/v1/chat/completions")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
TIMEOUT_SECONDS = int(os.environ.get("TIMEOUT_SECONDS", "120"))

# In-memory history of evaluations
evaluation_history = []

# Models
class Message(BaseModel):
    role: str
    content: str

class EvaluationRequest(BaseModel):
    current_code: str
    performance_data: Dict[str, Any]
    previous_bottlenecks: Optional[List[Dict[str, Any]]] = None
    optimization_history: Optional[List[Dict[str, Any]]] = None
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 4096

class ChatCompletionRequest(BaseModel):
    model: str = "deepseek-r-1"
    messages: List[Message]
    temperature: float = 0.2
    max_tokens: int = 4096
    stream: bool = False

class EvaluationResponse(BaseModel):
    bottlenecks: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    performance_analysis: Dict[str, Any]
    raw_evaluation: Optional[str] = None

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

def create_evaluation_prompt(request: EvaluationRequest) -> str:
    """Create a well-structured prompt for performance evaluation."""
    
    # Extract key information
    code = request.current_code
    perf_data = json.dumps(request.performance_data, indent=2)
    
    # Format previous bottlenecks if available
    prev_bottlenecks_text = ""
    if request.previous_bottlenecks and len(request.previous_bottlenecks) > 0:
        prev_bottlenecks_text = "## Previously Identified Bottlenecks\n```json\n"
        prev_bottlenecks_text += json.dumps(request.previous_bottlenecks, indent=2)
        prev_bottlenecks_text += "\n```\n\n"
    
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
    
    prompt = f"""# GPU Fluid Simulation Performance Evaluation

You are an expert GPU performance analyst specializing in fluid simulations. Your task is to analyze the provided GPU code and performance metrics to identify bottlenecks and suggest optimization strategies.

## Current Performance Metrics
```json
{perf_data}
```

{prev_bottlenecks_text}
{history_text}

## Current Code to Analyze
```
{code}
```

## Your Task:

1. Analyze the code and performance metrics thoroughly
2. Identify the top 3 performance bottlenecks based on the metrics
3. For each bottleneck, suggest specific optimization strategies
4. Rank the bottlenecks by potential impact
5. Consider any previously identified bottlenecks and recent optimization history

Provide your analysis in the following JSON format:
```json
{{
    "bottlenecks": [
        {{
            "name": "descriptive bottleneck name",
            "description": "detailed description of the issue",
            "impact": "high/medium/low",
            "evidence": "evidence from performance data or code that indicates this bottleneck"
        }}
        // other bottlenecks...
    ],
    "recommendations": [
        {{
            "target": "name of the bottleneck this addresses",
            "strategy": "detailed description of the optimization strategy",
            "expected_impact": "estimated performance improvement",
            "implementation_difficulty": "high/medium/low",
            "code_suggestion": "brief example of how to implement this optimization"
        }}
        // other recommendations...
    ],
    "performance_analysis": {{
        "primary_limitation": "compute/memory/synchronization/etc.",
        "utilization_assessment": "assessment of current resource utilization",
        "optimization_potential": "high/medium/low"
    }}
}}
```

IMPORTANT: Focus on concrete, actionable bottlenecks with clear evidence in the performance data or code.
"""
    return prompt

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_performance(request: EvaluationRequest, background_tasks: BackgroundTasks):
    """Endpoint to evaluate performance and identify bottlenecks."""
    logger.info("Received evaluation request")
    
    # Create the evaluation prompt
    prompt = create_evaluation_prompt(request)
    
    # Format the request for DeepSeek API
    chat_request = ChatCompletionRequest(
        model="deepseek-r-1",
        messages=[
            Message(role="system", content="You are an expert GPU performance analyst specializing in fluid simulations on GPU. Provide detailed, accurate, and actionable performance analysis."),
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
    evaluation_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    
    # Log the evaluation
    evaluation_entry = {
        "timestamp": time.time(),
        "request_summary": {
            "code_length": len(request.current_code),
            "performance_metrics": list(request.performance_data.keys())
        },
        "response_time_seconds": elapsed_time,
        "evaluation_length": len(evaluation_text)
    }
    background_tasks.add_task(lambda: evaluation_history.append(evaluation_entry))
    
    # Extract JSON data from the response
    try:
        # Try to extract JSON from the response
        json_data = extract_json_from_text(evaluation_text)
        
        # Make sure required fields are present
        bottlenecks = json_data.get("bottlenecks", [])
        recommendations = json_data.get("recommendations", [])
        performance_analysis = json_data.get("performance_analysis", {})
        
        if not bottlenecks:
            bottlenecks = extract_bottlenecks_from_text(evaluation_text)
        
        if not recommendations:
            recommendations = extract_recommendations_from_text(evaluation_text)
            
        if not performance_analysis:
            performance_analysis = extract_performance_analysis_from_text(evaluation_text)
            
    except Exception as e:
        logger.error(f"Error extracting structured data: {str(e)}")
        # Fallback to text extraction if JSON parsing fails
        bottlenecks = extract_bottlenecks_from_text(evaluation_text)
        recommendations = extract_recommendations_from_text(evaluation_text)
        performance_analysis = extract_performance_analysis_from_text(evaluation_text)
    
    return EvaluationResponse(
        bottlenecks=bottlenecks,
        recommendations=recommendations,
        performance_analysis=performance_analysis,
        raw_evaluation=evaluation_text
    )

def extract_json_from_text(text: str) -> Dict[str, Any]:
    """Extract JSON data from the generated text."""
    import re
    import json
    
    # Find JSON blocks in the text
    json_blocks = re.findall(r'```json(.*?)```', text, re.DOTALL)
    
    if not json_blocks:
        # Try without the json specifier
        json_blocks = re.findall(r'```(.*?)```', text, re.DOTALL)
    
    if json_blocks:
        # Try each block until we find valid JSON
        for block in json_blocks:
            try:
                return json.loads(block.strip())
            except json.JSONDecodeError:
                continue
    
    # If no valid JSON blocks found, look for just a dictionary pattern
    dict_match = re.search(r'{.*}', text, re.DOTALL)
    if dict_match:
        try:
            return json.loads(dict_match.group(0))
        except json.JSONDecodeError:
            pass
    
    # If all parsing fails, return an empty dict
    raise ValueError("No valid JSON found in response")

def extract_bottlenecks_from_text(text: str) -> List[Dict[str, Any]]:
    """Extract bottlenecks from raw text when JSON parsing fails."""
    bottlenecks = []
    
    # Common bottleneck indicators
    indicators = [
        "bottleneck", "issue", "problem", "limitation", 
        "constraint", "slow", "inefficient"
    ]
    
    lines = text.split('\n')
    current_bottleneck = {}
    
    for i, line in enumerate(lines):
        line_lower = line.lower()
        
        # Check if this line indicates a bottleneck
        if any(ind in line_lower for ind in indicators):
            # Save previous bottleneck if it exists
            if current_bottleneck and "name" in current_bottleneck:
                bottlenecks.append(current_bottleneck)
                current_bottleneck = {}
            
            # Extract the bottleneck name
            current_bottleneck["name"] = line.strip()
            current_bottleneck["description"] = ""
            
            # Try to determine impact
            if "critical" in line_lower or "severe" in line_lower or "major" in line_lower or "high" in line_lower:
                current_bottleneck["impact"] = "high"
            elif "minor" in line_lower or "small" in line_lower or "low" in line_lower:
                current_bottleneck["impact"] = "low"
            else:
                current_bottleneck["impact"] = "medium"
            
            # Look ahead to gather description
            for j in range(1, 5):
                if i + j < len(lines) and not any(ind in lines[i + j].lower() for ind in indicators):
                    current_bottleneck["description"] += " " + lines[i + j].strip()
                else:
                    break
    
    # Add the last bottleneck if it exists
    if current_bottleneck and "name" in current_bottleneck:
        bottlenecks.append(current_bottleneck)
    
    # If no bottlenecks were found, create a generic one
    if not bottlenecks:
        bottlenecks.append({
            "name": "General Performance Issues",
            "description": "The evaluation couldn't identify specific bottlenecks. Consider general optimization techniques.",
            "impact": "medium",
            "evidence": "Based on overall performance metrics."
        })
    
    # Add missing fields
    for b in bottlenecks:
        if "evidence" not in b:
            b["evidence"] = "Identified from code and performance patterns."
    
    return bottlenecks

def extract_recommendations_from_text(text: str) -> List[Dict[str, Any]]:
    """Extract recommendations from raw text when JSON parsing fails."""
    recommendations = []
    
    # Common recommendation indicators
    indicators = [
        "recommend", "suggest", "optimize", "improve", 
        "enhancement", "strategy", "solution"
    ]
    
    lines = text.split('\n')
    current_rec = {}
    
    for i, line in enumerate(lines):
        line_lower = line.lower()
        
        # Check if this line indicates a recommendation
        if any(ind in line_lower for ind in indicators):
            # Save previous recommendation if it exists
            if current_rec and "strategy" in current_rec:
                recommendations.append(current_rec)
                current_rec = {}
            
            # Extract the recommendation strategy
            current_rec["strategy"] = line.strip()
            
            # Try to identify target bottleneck
            current_rec["target"] = "general optimization"
            
            # Try to determine expected impact
            if "significant" in line_lower or "substantial" in line_lower or "major" in line_lower or "high" in line_lower:
                current_rec["expected_impact"] = "high improvement expected"
            elif "minor" in line_lower or "small" in line_lower or "low" in line_lower:
                current_rec["expected_impact"] = "minor improvement expected"
            else:
                current_rec["expected_impact"] = "moderate improvement expected"
            
            # Try to determine implementation difficulty
            if "complex" in line_lower or "difficult" in line_lower or "challenging" in line_lower:
                current_rec["implementation_difficulty"] = "high"
            elif "simple" in line_lower or "easy" in line_lower or "straightforward" in line_lower:
                current_rec["implementation_difficulty"] = "low"
            else:
                current_rec["implementation_difficulty"] = "medium"
    
    # Add the last recommendation if it exists
    if current_rec and "strategy" in current_rec:
        recommendations.append(current_rec)
    
    # If no recommendations were found, create a generic one
    if not recommendations:
        recommendations.append({
            "target": "general optimization",
            "strategy": "Review memory access patterns and thread divergence",
            "expected_impact": "moderate improvement expected",
            "implementation_difficulty": "medium",
            "code_suggestion": "Look for coalesced memory access patterns and reduce branch divergence"
        })
    
    # Add missing fields
    for r in recommendations:
        if "code_suggestion" not in r:
            r["code_suggestion"] = "Implementation details not specified."
    
    return recommendations

def extract_performance_analysis_from_text(text: str) -> Dict[str, Any]:
    """Extract overall performance analysis from raw text."""
    analysis = {
        "primary_limitation": "unknown",
        "utilization_assessment": "unknown",
        "optimization_potential": "medium"
    }
    
    text_lower = text.lower()
    
    # Try to identify primary limitation
    if "memory bandwidth" in text_lower or "memory bound" in text_lower:
        analysis["primary_limitation"] = "memory bandwidth"
    elif "compute" in text_lower and ("bound" in text_lower or "limited" in text_lower):
        analysis["primary_limitation"] = "compute"
    elif "divergen" in text_lower:
        analysis["primary_limitation"] = "thread divergence"
    elif "synchronization" in text_lower or "atomic" in text_lower:
        analysis["primary_limitation"] = "synchronization"
    
    # Try to assess utilization
    if "under-utilized" in text_lower or "low utilization" in text_lower:
        analysis["utilization_assessment"] = "resources under-utilized"
    elif "well utilized" in text_lower or "good utilization" in text_lower:
        analysis["utilization_assessment"] = "resources well utilized"
    elif "over-utilized" in text_lower or "maxed out" in text_lower:
        analysis["utilization_assessment"] = "resources over-utilized"
    
    # Try to determine optimization potential
    if "significant" in text_lower and "potential" in text_lower:
        analysis["optimization_potential"] = "high"
    elif "limited" in text_lower and "potential" in text_lower:
        analysis["optimization_potential"] = "low"
    
    return analysis

# For debugging
@app.get("/status")
async def status():
    """Get service status."""
    return {
        "status": "running",
        "evaluations": len(evaluation_history),
        "last_evaluation": evaluation_history[-1] if evaluation_history else None
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
    uvicorn.run("evaluator_service:app", host="0.0.0.0", port=8000, reload=True)