"""
3D Fluid Simulation for H100 GPUs on Northflank
Based on WebGPU-Ocean, adapted for headless rendering to video

This script orchestrates:
1. Running the physics simulation
2. Rendering frames to a directory
3. Proper setup for Northflank environment with H100 GPUs
"""

import os
import sys
import time
import subprocess
import argparse
import traceback

def setup_environment():
    """Set up the environment for GPU simulation on Northflank."""
    print("Setting up environment for GPU acceleration...")
    
    # Check if Taichi is already installed
    try:
        import taichi as ti
        print(f"Taichi version: {ti.__version__}")
    except ImportError:
        print("Installing Taichi...")
        subprocess.check_call([
            "pip", "install", "--no-cache-dir", "taichi"
        ])
    
    # Check OpenCV
    try:
        import cv2
        print(f"OpenCV version: {cv2.__version__}")
    except ImportError:
        print("Installing OpenCV...")
        subprocess.check_call([
            "pip", "install", "--no-cache-dir", "opencv-python-headless"
        ])
        
    # Check NumPy
    try:
        import numpy as np
        print(f"NumPy version: {np.__version__}")
    except ImportError:
        print("Installing NumPy...")
        subprocess.check_call([
            "pip", "install", "--no-cache-dir", "numpy"
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
                  save_interval=3.0, output_dir="frames"):
    """Run the fluid simulation and render frames to a directory."""
    import traceback
    from fluid_physics import FluidSimulation
    
    print(f"Starting 3D fluid simulation with:")
    print(f"  - Grid size: {grid_size}")
    print(f"  - Particles: {particles}")
    print(f"  - Duration: {duration} seconds")
    print(f"  - Output directory: {output_dir}")
    
    # Use the frame renderer (no display or OpenGL required)
    try:
        from frames_renderer import run_frames_renderer
        output_path = run_frames_renderer(
            grid_size=grid_size,
            particles_count=particles,
            simulation_time=duration,
            save_interval=save_interval,
            output_dir=output_dir
        )
        
        # Verify that files were created
        import glob
        frames = glob.glob(os.path.join(output_dir, "*.png"))
        print(f"Created {len(frames)} frame images")
        
        particle_files = glob.glob(os.path.join("simulation_data", "*.pkl"))
        print(f"Created {len(particle_files)} particle data files")
        
    except Exception as e:
        print(f"Frame renderer failed: {e}")
        print(f"Exception details:\n{traceback.format_exc()}")
        raise
    
    print(f"Simulation completed. Output frames saved to: {output_path}")
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
    parser.add_argument('--output-dir', type=str, default="frames", 
                        help='Output directory for frames')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Set up environment
    setup_environment()
    
    # Run simulation with parsed arguments
    grid_size = (args.grid, args.grid, args.grid)
    try:
        run_simulation(
            grid_size=grid_size, 
            particles=args.particles,
            duration=args.duration,
            save_interval=args.save_interval,
            output_dir=args.output_dir
        )
    except Exception as e:
        import traceback
        print(f"Fatal error: {e}")
        print(traceback.format_exc())
        print("Attempting to run with smallest possible grid size...")
        
        try:
            # Try one more time with minimal settings
            run_simulation(
                grid_size=(64, 64, 64),
                particles=500,
                duration=3.0,
                save_interval=3.0,
                output_dir=args.output_dir
            )
        except Exception as e2:
            print(f"Second attempt also failed: {e2}")
            print("Please check the logs for details.")
    
    # Keep the container running after simulation is complete
    print("\n")
    print("=" * 80)
    print("Simulation completed. Container will remain running.")
    print("You can now copy the files using:")
    print(f"  - Frame images: /app/{args.output_dir}/")
    print("  - Particle data: /app/simulation_data/")
    print("=" * 80)
    print("\n")
    
    # Keep the process running indefinitely
    import time
    try:
        while True:
            time.sleep(3600)  # Sleep for an hour at a time
            print("Container still running... You can copy the files.")
    except KeyboardInterrupt:
        print("Container terminating.")