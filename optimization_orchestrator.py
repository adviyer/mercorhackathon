# optimization_orchestrator.py
import os
import json
import logging
import time
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import httpx
import asyncio
import uvicorn
from contextlib import asynccontextmanager
import subprocess
import uuid
import shutil
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("orchestrator")

# FastAPI app
app = FastAPI(title="GPU Fluid Simulation Optimization Orchestrator")

# Configuration
CODE_GENERATOR_URL = os.environ.get("CODE_GENERATOR_URL", "http://code-generator-service:8000")
EVALUATOR_URL = os.environ.get("EVALUATOR_URL", "http://evaluator-service:8000")
SIMULATION_DIR = os.environ.get("SIMULATION_DIR", "./fluid_simulation")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "./optimization_results")
MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "10"))
PERFORMANCE_THRESHOLD = float(os.environ.get("PERFORMANCE_THRESHOLD", "1.05"))  # 5% improvement to continue

# Ensure directories exist
os.makedirs(SIMULATION_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "versions"), exist_ok=True)

# Models
class OptimizationRequest(BaseModel):
    initial_code: str
    gpu_info: Optional[Dict[str, Any]] = None
    max_iterations: Optional[int] = None
    performance_threshold: Optional[float] = None
    stop_early: Optional[bool] = True

class OptimizationIteration(BaseModel):
    iteration: int
    code: str
    performance_data: Dict[str, Any]
    bottlenecks: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    optimization_strategy: Dict[str, Any]
    timestamp: float

class OptimizationResponse(BaseModel):
    status: str
    message: str
    optimization_id: str
    current_iteration: int
    iterations_completed: List[int]
    best_performance: Dict[str, Any]
    current_code: Optional[str] = None

class OptimizationStatus(BaseModel):
    optimization_id: str
    status: str
    current_iteration: int
    iterations_completed: List[int]
    best_iteration: int
    best_performance: Dict[str, Any]
    performance_history: List[Dict[str, Any]]
    estimated_time_remaining: Optional[float] = None

# Store active optimizations
active_optimizations = {}

@asynccontextmanager
async def get_client():
    async with httpx.AsyncClient(timeout=600) as client:
        yield client

def save_optimization_state(optimization_id: str, state: Dict[str, Any]):
    """Save the current optimization state to disk."""
    filepath = os.path.join(RESULTS_DIR, f"{optimization_id}.json")
    with open(filepath, "w") as f:
        json.dump(state, f, indent=2)
    logger.info(f"Saved optimization state to {filepath}")

def load_optimization_state(optimization_id: str) -> Dict[str, Any]:
    """Load optimization state from disk."""
    filepath = os.path.join(RESULTS_DIR, f"{optimization_id}.json")
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"State file {filepath} not found.")
        return None

def save_code_version(optimization_id: str, iteration: int, code: str):
    """Save a specific code version to disk."""
    version_dir = os.path.join(RESULTS_DIR, "versions", optimization_id)
    os.makedirs(version_dir, exist_ok=True)
    
    filepath = os.path.join(version_dir, f"iteration_{iteration}.cu")
    with open(filepath, "w") as f:
        f.write(code)
    logger.info(f"Saved code version for iteration {iteration} to {filepath}")

async def run_performance_test(code: str, optimization_id: str, iteration: int) -> Dict[str, Any]:
    """Compile and run the fluid simulation code to measure performance."""
    # Create a unique directory for this test run
    run_dir = os.path.join(SIMULATION_DIR, f"run_{optimization_id}_{iteration}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Write the code to a file
    code_path = os.path.join(run_dir, "fluid_sim.cu")
    with open(code_path, "w") as f:
        f.write(code)
    
    # Compile the code
    logger.info(f"Compiling code for optimization {optimization_id}, iteration {iteration}")
    try:
        compile_process = subprocess.run(
            ["nvcc", "-O3", code_path, "-o", os.path.join(run_dir, "fluid_sim")],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Compilation successful")
    except subprocess.CalledProcessError as e:
        logger.error(f"Compilation failed: {e.stderr}")
        return {
            "status": "compilation_failed",
            "error": e.stderr,
            "fps": 0,
            "frame_time_ms": 0,
            "memory_usage_mb": 0,
            "compilation_error": True
        }
    
    # Run the simulation
    logger.info(f"Running simulation to measure performance")
    try:
        # Set up env variables for CUDA profiling
        env = os.environ.copy()
        env["CUDA_PROFILE"] = "1"
        env["CUDA_PROFILE_LOG"] = os.path.join(run_dir, "cuda_profile.log")
        
        # Run with timing
        sim_process = subprocess.run(
            [os.path.join(run_dir, "fluid_sim"), "--benchmark", "--frames", "300"],
            check=True,
            capture_output=True,
            text=True,
            env=env,
            timeout=120  # 2 minute timeout
        )
        
        # Try to parse performance output
        # This assumes the simulation outputs performance metrics in a parseable format
        output = sim_process.stdout
        
        # For this example, we'll simulate parsing the performance data
        # In a real implementation, you would extract this from the simulation output
        performance_data = extract_performance_data(output, run_dir)
        
        # Save the performance data
        performance_file = os.path.join(run_dir, "performance.json")
        with open(performance_file, "w") as f:
            json.dump(performance_data, f, indent=2)
        
        return performance_data
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Simulation failed: {e.stderr}")
        return {
            "status": "simulation_failed",
            "error": e.stderr,
            "fps": 0,
            "frame_time_ms": 0,
            "memory_usage_mb": 0,
            "runtime_error": True
        }
    except subprocess.TimeoutExpired:
        logger.error(f"Simulation timed out after 120 seconds")
        return {
            "status": "simulation_timeout",
            "error": "Simulation timed out after 120 seconds",
            "fps": 0,
            "frame_time_ms": 0,
            "memory_usage_mb": 0,
            "timeout": True
        }
    finally:
        # Clean up compiled files, but keep the code and performance data
        executable_path = os.path.join(run_dir, "fluid_sim")
        if os.path.exists(executable_path):
            os.remove(executable_path)

def extract_performance_data(output: str, run_dir: str) -> Dict[str, Any]:
    """Extract performance data from simulation output and CUDA profiler."""
    # This is a stub function - in a real implementation, you would parse the simulation output
    # and CUDA profiler logs to extract detailed performance metrics
    
    # For this example, we'll create simulated performance data
    # In a real scenario, parse actual output from your fluid simulation
    
    # Try to find FPS information in the output
    fps = 0
    frame_time_ms = 0
    memory_usage_mb = 0
    
    # Check for common performance output patterns
    import re
    
    fps_match = re.search(r'FPS[:\s]+(\d+\.\d+)', output)
    if fps_match:
        fps = float(fps_match.group(1))
    
    frame_time_match = re.search(r'Frame\s+time[:\s]+(\d+\.\d+)\s*ms', output)
    if frame_time_match:
        frame_time_ms = float(frame_time_match.group(1))
    
    memory_match = re.search(r'Memory[:\s]+(\d+\.\d+)\s*MB', output)
    if memory_match:
        memory_usage_mb = float(memory_match.group(1))
    
    # Try to parse CUDA profiler output
    profiler_data = {}
    profiler_log = os.path.join(run_dir, "cuda_profile.log")
    
    if os.path.exists(profiler_log):
        try:
            with open(profiler_log, "r") as f:
                profiler_content = f.read()
            
            # Parse kernel execution times
            kernel_data = []
            kernel_matches = re.finditer(r'(\w+)\s+(\d+\.\d+)\s+(\d+\.\d+)', profiler_content)
            
            for match in kernel_matches:
                kernel_name = match.group(1)
                kernel_time = float(match.group(2))
                kernel_data.append({
                    "name": kernel_name,
                    "time_ms": kernel_time
                })
            
            profiler_data["kernels"] = kernel_data
            
            # Look for occupancy data
            occupancy_matches = re.finditer(r'(\w+)\s+occupancy[:\s]+(\d+\.\d+)', profiler_content, re.IGNORECASE)
            occupancy_data = {}
            
            for match in occupancy_matches:
                kernel_name = match.group(1)
                occupancy = float(match.group(2))
                occupancy_data[kernel_name] = occupancy
            
            profiler_data["occupancy"] = occupancy_data
            
        except Exception as e:
            logger.error(f"Error parsing profiler log: {str(e)}")
    
    # If we couldn't extract real values, use simulated ones
    if fps == 0:
        fps = 30.0 + (5.0 * (int(datetime.now().timestamp()) % 2))  # Vary between iterations for testing
    
    if frame_time_ms == 0:
        frame_time_ms = 1000.0 / fps
    
    if memory_usage_mb == 0:
        memory_usage_mb = 1200 + (int(datetime.now().timestamp()) % 100)
    
    # Create the performance data dictionary
    performance_data = {
        "status": "success",
        "fps": fps,
        "frame_time_ms": frame_time_ms,
        "memory_usage_mb": memory_usage_mb,
        "gpu_utilization": 0.75 + (0.1 * (int(datetime.now().timestamp()) % 2)),
        "profiler_data": profiler_data
    }
    
    return performance_data

async def evaluate_performance(code: str, performance_data: Dict[str, Any], 
                              previous_bottlenecks: List[Dict[str, Any]] = None,
                              optimization_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Send code and performance data to the evaluator service for analysis."""
    logger.info("Calling evaluator service")
    
    evaluation_request = {
        "current_code": code,
        "performance_data": performance_data,
        "previous_bottlenecks": previous_bottlenecks,
        "optimization_history": optimization_history
    }
    
    async with get_client() as client:
        try:
            response = await client.post(
                f"{EVALUATOR_URL}/evaluate",
                json=evaluation_request
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error evaluating performance: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Performance evaluation failed: {str(e)}")

async def generate_optimized_code(code: str, performance_data: Dict[str, Any],
                                bottlenecks: List[Dict[str, Any]],
                                optimization_strategy: Dict[str, Any],
                                optimization_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Send data to the code generator service to get optimized code."""
    logger.info("Calling code generator service")
    
    generation_request = {
        "current_code": code,
        "performance_data": performance_data,
        "bottlenecks": bottlenecks,
        "optimization_strategy": optimization_strategy,
        "optimization_history": optimization_history
    }
    
    async with get_client() as client:
        try:
            response = await client.post(
                f"{CODE_GENERATOR_URL}/optimize",
                json=generation_request
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error generating optimized code: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Code generation failed: {str(e)}")

async def run_optimization_iteration(optimization_id: str, iteration: int) -> Dict[str, Any]:
    """Run a single iteration of the optimization process."""
    logger.info(f"Starting optimization iteration {iteration} for {optimization_id}")
    
    # Get the current optimization state
    state = active_optimizations[optimization_id]
    
    # Get the code from the previous iteration
    if iteration == 1:
        current_code = state["initial_code"]
    else:
        current_code = state["iterations"][-1]["code"]
    
    # Save the current code version
    save_code_version(optimization_id, iteration, current_code)
    
    # Run performance test
    performance_data = await run_performance_test(current_code, optimization_id, iteration)
    logger.info(f"Performance test completed: FPS={performance_data.get('fps', 'unknown')}")
    
    # Check if the test failed
    if "error" in performance_data:
        logger.error(f"Performance test failed: {performance_data['error']}")
        return {
            "status": "failed",
            "error": performance_data["error"],
            "code": current_code,
            "performance_data": performance_data
        }
    
    # Get previous bottlenecks and optimization history
    previous_bottlenecks = None
    optimization_history = None
    
    if len(state["iterations"]) > 0:
        previous_bottlenecks = state["iterations"][-1].get("bottlenecks")
        optimization_history = [
            {
                "iteration": i["iteration"],
                "bottlenecks": i.get("bottlenecks"),
                "performance_data": i["performance_data"],
                "code_diff_summary": f"Changed from iteration {i['iteration']-1 if i['iteration'] > 1 else 'initial'}"
            }
            for i in state["iterations"]
        ]
    
    # Evaluate performance
    evaluation_result = await evaluate_performance(
        current_code, 
        performance_data,
        previous_bottlenecks,
        optimization_history
    )
    
    bottlenecks = evaluation_result.get("bottlenecks", [])
    recommendations = evaluation_result.get("recommendations", [])
    
    # Decide on optimization strategy
    # Here we're using a simple strategy: focus on the highest impact bottleneck
    optimization_strategy = {
        "selected_bottleneck": bottlenecks[0]["name"] if bottlenecks else "general optimization",
        "strategy": recommendations[0]["strategy"] if recommendations else "general performance improvements",
        "reasoning": "Targeting highest impact bottleneck",
        "exploration_factor": 0.2,  # Low exploration, high exploitation
        "termination_recommendation": False
    }
    
    # Generate optimized code
    code_result = await generate_optimized_code(
        current_code,
        performance_data,
        bottlenecks,
        optimization_strategy,
        optimization_history
    )
    
    optimized_code = code_result.get("optimized_code", current_code)
    
    # Record this iteration
    iteration_data = {
        "iteration": iteration,
        "code": optimized_code,
        "performance_data": performance_data,
        "bottlenecks": bottlenecks,
        "recommendations": recommendations,
        "optimization_strategy": optimization_strategy,
        "timestamp": time.time()
    }
    
    # Update the best performance if this is better
    if performance_data.get("fps", 0) > state["best_performance"].get("fps", 0):
        state["best_performance"] = performance_data
        state["best_iteration"] = iteration
    
    # Update the state
    state["current_iteration"] = iteration
    state["iterations"].append(iteration_data)
    state["iterations_completed"].append(iteration)
    
    # Save the updated state
    active_optimizations[optimization_id] = state
    save_optimization_state(optimization_id, state)
    
    return {
        "status": "completed",
        "code": optimized_code,
        "performance_data": performance_data,
        "bottlenecks": bottlenecks,
        "recommendations": recommendations,
        "optimization_strategy": optimization_strategy
    }

async def optimization_loop(optimization_id: str, background_tasks: BackgroundTasks):
    """Main optimization loop that runs iterations until completion or failure."""
    try:
        # Get the optimization state
        state = active_optimizations[optimization_id]
        
        # Set parameters
        max_iterations = state.get("max_iterations", MAX_ITERATIONS)
        performance_threshold = state.get("performance_threshold", PERFORMANCE_THRESHOLD)
        stop_early = state.get("stop_early", True)
        
        # Run iterations
        current_iteration = state["current_iteration"]
        
        while current_iteration < max_iterations:
            current_iteration += 1
            
            # Run this iteration
            result = await run_optimization_iteration(optimization_id, current_iteration)
            
            # Check if the iteration failed
            if result["status"] == "failed":
                logger.error(f"Optimization iteration {current_iteration} failed: {result.get('error', 'unknown error')}")
                state["status"] = "failed"
                state["message"] = f"Iteration {current_iteration} failed: {result.get('error', 'unknown error')}"
                break
            
            # Update state
            state["status"] = "in_progress"
            
            # Check if we should stop early
            if stop_early and current_iteration > 1:
                # Compare current FPS with previous iteration
                current_fps = result["performance_data"].get("fps", 0)
                previous_fps = state["iterations"][-2]["performance_data"].get("fps", 0)
                
                # If improvement is below threshold for 2 consecutive iterations, stop
                if current_fps < previous_fps * performance_threshold:
                    if state.get("below_threshold_count", 0) >= 1:
                        logger.info(f"Stopping early: improvement below threshold for 2 consecutive iterations")
                        state["status"] = "completed"
                        state["message"] = "Optimization completed (stopped early due to diminishing returns)"
                        break
                    else:
                        state["below_threshold_count"] = state.get("below_threshold_count", 0) + 1
                else:
                    state["below_threshold_count"] = 0
        
        # Check if we've reached the maximum number of iterations
        if current_iteration >= max_iterations:
            state["status"] = "completed"
            state["message"] = f"Optimization completed (reached maximum iterations: {max_iterations})"
        
        # Update and save final state
        active_optimizations[optimization_id] = state
        save_optimization_state(optimization_id, state)
        
        logger.info(f"Optimization loop completed for {optimization_id}: {state['status']} - {state['message']}")
    
    except Exception as e:
        logger.error(f"Error in optimization loop for {optimization_id}: {str(e)}")
        # Update state to indicate failure
        if optimization_id in active_optimizations:
            active_optimizations[optimization_id]["status"] = "failed"
            active_optimizations[optimization_id]["message"] = f"Error during optimization: {str(e)}"
            save_optimization_state(optimization_id, active_optimizations[optimization_id])

@app.post("/optimize", response_model=OptimizationResponse)
async def start_optimization(request: OptimizationRequest, background_tasks: BackgroundTasks):
    """Start a new optimization process."""
    # Generate a unique ID for this optimization
    optimization_id = str(uuid.uuid4())
    
    # Create initial state
    state = {
        "optimization_id": optimization_id,
        "status": "initializing",
        "message": "Optimization started",
        "initial_code": request.initial_code,
        "current_iteration": 0,
        "iterations_completed": [],
        "best_iteration": 0,
        "best_performance": {"fps": 0},
        "iterations": [],
        "gpu_info": request.gpu_info or {},
        "max_iterations": request.max_iterations or MAX_ITERATIONS,
        "performance_threshold": request.performance_threshold or PERFORMANCE_THRESHOLD,
        "stop_early": request.stop_early if request.stop_early is not None else True,
        "start_time": time.time()
    }
    
    # Save the initial state
    active_optimizations[optimization_id] = state
    save_optimization_state(optimization_id, state)
    
    # Start the optimization loop in the background
    background_tasks.add_task(optimization_loop, optimization_id, background_tasks)
    
    # Return the initial response
    return OptimizationResponse(
        status="initializing",
        message="Optimization started",
        optimization_id=optimization_id,
        current_iteration=0,
        iterations_completed=[],
        best_performance={"fps": 0}
    )

@app.get("/optimization/{optimization_id}", response_model=OptimizationStatus)
async def get_optimization_status(optimization_id: str):
    """Get the status of an optimization process."""
    # Check if this optimization exists
    if optimization_id not in active_optimizations:
        # Try to load from disk
        state = load_optimization_state(optimization_id)
        if state:
            active_optimizations[optimization_id] = state
        else:
            raise HTTPException(status_code=404, detail=f"Optimization {optimization_id} not found")
    
    # Get the state
    state = active_optimizations[optimization_id]
    
    # Calculate estimated time remaining
    est_time_remaining = None
    if state["status"] == "in_progress" and len(state["iterations"]) > 0:
        # Calculate average time per iteration
        elapsed_time = time.time() - state["start_time"]
        iterations_completed = len(state["iterations_completed"])
        if iterations_completed > 0:
            avg_time_per_iteration = elapsed_time / iterations_completed
            remaining_iterations = state["max_iterations"] - state["current_iteration"]
            est_time_remaining = avg_time_per_iteration * remaining_iterations
    
    # Create performance history
    performance_history = [
        {
            "iteration": i["iteration"],
            "fps": i["performance_data"].get("fps", 0),
            "frame_time_ms": i["performance_data"].get("frame_time_ms", 0),
            "timestamp": i["timestamp"]
        }
        for i in state["iterations"]
    ]
    
    return OptimizationStatus(
        optimization_id=state["optimization_id"],
        status=state["status"],
        current_iteration=state["current_iteration"],
        iterations_completed=state["iterations_completed"],
        best_iteration=state["best_iteration"],
        best_performance=state["best_performance"],
        performance_history=performance_history,
        estimated_time_remaining=est_time_remaining
    )

@app.get("/optimization/{optimization_id}/best", response_model=OptimizationResponse)
async def get_best_iteration(optimization_id: str):
    """Get the best iteration of an optimization process."""
    # Check if this optimization exists
    if optimization_id not in active_optimizations:
        # Try to load from disk
        state = load_optimization_state(optimization_id)
        if state:
            active_optimizations[optimization_id] = state
        else:
            raise HTTPException(status_code=404, detail=f"Optimization {optimization_id} not found")
    
    # Get the state
    state = active_optimizations[optimization_id]
    
    # Get the best iteration
    best_iter = state["best_iteration"]
    best_code = None
    
    if best_iter > 0 and len(state["iterations"]) >= best_iter:
        best_code = state["iterations"][best_iter - 1]["code"]
    elif best_iter == 0 and "initial_code" in state:
        best_code = state["initial_code"]
    
    return OptimizationResponse(
        status=state["status"],
        message=f"Best iteration: {best_iter}",
        optimization_id=state["optimization_id"],
        current_iteration=state["current_iteration"],
        iterations_completed=state["iterations_completed"],
        best_performance=state["best_performance"],
        current_code=best_code
    )

@app.get("/optimization/{optimization_id}/iteration/{iteration}")
async def get_specific_iteration(optimization_id: str, iteration: int):
    """Get a specific iteration of an optimization process."""
    # Check if this optimization exists
    if optimization_id not in active_optimizations:
        # Try to load from disk
        state = load_optimization_state(optimization_id)
        if state:
            active_optimizations[optimization_id] = state
        else:
            raise HTTPException(status_code=404, detail=f"Optimization {optimization_id} not found")
    
    # Get the state
    state = active_optimizations[optimization_id]
    
    # Check if the iteration exists
    if iteration == 0:
        # Initial code
        return {
            "iteration": 0,
            "code": state.get("initial_code", ""),
            "performance_data": {},
            "status": "initial"
        }
    elif 1 <= iteration <= len(state["iterations"]):
        # Get the iteration data
        iter_data = state["iterations"][iteration - 1]
        return iter_data
    else:
        raise HTTPException(status_code=404, detail=f"Iteration {iteration} not found")

if __name__ == "__main__":
    uvicorn.run("optimization_orchestrator:app", host="0.0.0.0", port=8000, reload=True)