"""
3D Fluid Simulation for H100 GPUs on Northflank
Based on WebGPU-Ocean, adapted for headless rendering to video

This script orchestrates:
1. Running the physics simulation
2. Rendering the simulation to a video file
3. Proper setup for Northflank environment with H100 GPUs
"""

import os
import sys
import time
import subprocess
import argparse

def setup_environment():
    """Set up the environment for GPU simulation on Northflank."""
    print("Setting up environment for GPU acceleration...")
    
    # Install dependencies
    subprocess.check_call([
        "pip", "install", "taichi", "numpy", "opencv-python", "moderngl", 
        "moderngl-window", "pillow", "PyOpenGL", "PyOpenGL-accelerate"
    ])
    
    # Check GPU availability
    try:
        import taichi as ti
        ti.init(arch=ti.gpu)
        gpu_info = ti.lang.impl.get_runtime().prog.get_kernel_profiler().get_device_name()
        print(f"Taichi initialized with GPU: {gpu_info}")
    except Exception as e:
        print(f"Warning: Could not initialize Taichi with GPU: {e}")
        print("Falling back to CPU for physics simulation.")
    
    # Try to detect NVIDIA GPUs with nvidia-smi
    try:
        nvidia_smi = subprocess.check_output(["nvidia-smi"]).decode()
        print("NVIDIA GPU detected:")
        print(nvidia_smi)
    except:
        print("Warning: nvidia-smi not available or no NVIDIA GPU found.")

def run_simulation(grid_size=(128, 128, 128), particles=1000, duration=5.0, 
                  save_interval=3.0, output_file="fluid_simulation.mp4"):
    """Run the full fluid simulation and rendering pipeline."""
    from fluid_physics import FluidSimulation
    from fluid_renderer import FluidRenderer
    import moderngl_window as mglw
    
    print(f"Starting 3D fluid simulation with:")
    print(f"  - Grid size: {grid_size}")
    print(f"  - Particles: {particles}")
    print(f"  - Duration: {duration} seconds")
    print(f"  - Output: {output_file}")
    
    # Create a custom config for the renderer
    config = {
        "class": FluidRenderer,
        "window": {
            "size": (1920, 1080),
            "title": "3D Fluid Simulation",
            "vsync": True,
            "resizable": False,
            "fullscreen": False,
            "cursor": True,
        },
        "headless": True,  # Important for Northflank environment
        "gl_version": (4, 3),
        "simulation_params": {
            "grid_size": grid_size,
            "particles_count": particles,
            "simulation_time": duration,
            "save_interval": save_interval,
            "output_video_path": output_file
        }
    }
    
    # Run the moderngl window application
    mglw.run_window_config(FluidRenderer, **config)
    
    print(f"Simulation completed. Output video saved to: {output_file}")
    print(f"Particle position data saved to: simulation_data/")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='3D Fluid Simulation')
    parser.add_argument('--grid', type=int, default=128, 
                        help='Grid size (same for x, y, z)')
    parser.add_argument('--particles', type=int, default=1000, 
                        help='Number of particles to track')
    parser.add_argument('--duration', type=float, default=5.0, 
                        help='Simulation duration in seconds')
    parser.add_argument('--save-interval', type=float, default=3.0, 
                        help='Interval to save particle positions')
    parser.add_argument('--output', type=str, default="fluid_simulation.mp4", 
                        help='Output video file path')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Set up environment
    setup_environment()
    
    # Run simulation with parsed arguments
    grid_size = (args.grid, args.grid, args.grid)
    run_simulation(
        grid_size=grid_size, 
        particles=args.particles,
        duration=args.duration,
        save_interval=args.save_interval,
        output_file=args.output
    )