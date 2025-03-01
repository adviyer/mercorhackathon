"""
Simple frame-by-frame renderer for the fluid simulation.
Does not require any display or OpenGL, only saves individual frames to disk.
"""

import numpy as np
import cv2
import os
import pickle
from fluid_physics import FluidSimulation

class FramesRenderer:
    """Renders simulation data as individual frames, with no display requirements."""
    
    def __init__(self, 
                 output_dir="frames",
                 grid_size=(128, 128, 128),
                 particles_count=1000,
                 simulation_time=5.0,
                 save_interval=3.0,
                 frame_size=(1920, 1080)):
        
        self.output_dir = output_dir
        self.frame_size = frame_size
        
        # Initialize the physics simulation
        self.simulation = FluidSimulation(
            grid_size=grid_size,
            particles_count=particles_count,
            simulation_time=simulation_time,
            save_interval=save_interval
        )
        
        # Make sure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Frame renderer initialized. Output directory: {self.output_dir}")
    
    def run_simulation_and_render(self):
        """Run the physics simulation and render frames to the output directory."""
        print("Running physics simulation...")
        
        # Run the simulation
        simulation_results = self.simulation.run_simulation()
        density_frames = simulation_results['density_frames']
        particle_frames = simulation_results['particle_frames']
        grid_size = simulation_results['grid_size']
        frame_count = len(density_frames)
        
        print(f"Simulation complete. Rendering {frame_count} frames...")
        
        # Render each frame
        for frame_idx in range(frame_count):
            # Get data for current frame
            density_data = density_frames[frame_idx]
            particles = particle_frames[frame_idx]
            
            # Create frame image
            frame = self.render_frame(density_data, particles, grid_size, frame_idx)
            
            # Save frame
            frame_path = os.path.join(self.output_dir, f"frame_{frame_idx:04d}.png")
            cv2.imwrite(frame_path, frame)
            
            # Print progress
            if frame_idx % 10 == 0 or frame_idx == frame_count - 1:
                print(f"Rendered frame {frame_idx+1}/{frame_count}")
        
        print(f"All frames saved to {self.output_dir}/ directory")
        return self.output_dir
    
    def render_frame(self, density_data, particles, grid_size, frame_idx):
        """Create a 2D visualization of the 3D fluid simulation."""
        # Create blank frame
        frame = np.zeros((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8)
        
        # Fill with background color
        frame[:] = (40, 40, 40)  # Dark gray background
        
        # Create a top-down view (slice through the middle of the z-axis)
        z_slice = grid_size[2] // 2
        slice_data = density_data[:, :, z_slice]
        
        # Create a front view (slice through the middle of the y-axis)
        y_slice = grid_size[1] // 2
        front_slice = density_data[:, y_slice, :]
        
        # Scale and position the visualizations
        margin = 40
        view_size = min((self.frame_size[0] - 3*margin) // 2, 
                        (self.frame_size[1] - 2*margin))
        
        # Draw the top-down view
        top_view_x = margin
        top_view_y = (self.frame_size[1] - view_size) // 2
        
        # Scale and colorize the density data
        top_view = self._colorize_density(slice_data, view_size)
        frame[top_view_y:top_view_y+view_size, top_view_x:top_view_x+view_size] = top_view
        
        # Draw the front view
        front_view_x = 2*margin + view_size
        front_view_y = (self.frame_size[1] - view_size) // 2
        
        # Scale and colorize the front view
        front_view = self._colorize_density(front_slice, view_size)
        frame[front_view_y:front_view_y+view_size, front_view_x:front_view_x+view_size] = front_view
        
        # Draw particles
        self._draw_particles(frame, particles, grid_size, 
                            top_view_x, top_view_y, view_size,
                            front_view_x, front_view_y)
        
        # Add frame text and border
        cv2.putText(frame, f"Frame: {frame_idx}", (margin, margin), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        cv2.putText(frame, "Top View", (top_view_x, top_view_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cv2.putText(frame, "Front View", (front_view_x, front_view_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Draw borders around the views
        cv2.rectangle(frame, 
                     (top_view_x-1, top_view_y-1), 
                     (top_view_x+view_size+1, top_view_y+view_size+1), 
                     (100, 100, 100), 1)
        
        cv2.rectangle(frame, 
                     (front_view_x-1, front_view_y-1), 
                     (front_view_x+view_size+1, front_view_y+view_size+1), 
                     (100, 100, 100), 1)
        
        return frame
    
    def _colorize_density(self, density_slice, view_size):
        """Convert density data to a colorized image."""
        # Normalize the density values
        density_norm = np.clip(density_slice, 0, 1)
        
        # Resize to view size
        density_resized = cv2.resize(density_norm, (view_size, view_size))
        
        # Create a colormap (blue gradient)
        cmap = np.zeros((view_size, view_size, 3), dtype=np.uint8)
        
        # Set the blue channel based on density
        cmap[:, :, 0] = (density_resized * 255).astype(np.uint8)  # Blue channel
        
        # Add some color variation
        cmap[:, :, 1] = (density_resized * 128).astype(np.uint8)  # Green channel
        
        return cmap
    
    def _draw_particles(self, frame, particles, grid_size, 
                       top_x, top_y, view_size,
                       front_x, front_y):
        """Draw particles on both views."""
        # Draw each particle
        for p in particles:
            # Normalize particle position to [0,1]
            px, py, pz = p[0], p[1], p[2]
            
            # Calculate positions in top view (x, z coordinates)
            top_px = int(top_x + px * view_size)
            top_py = int(top_y + pz * view_size)
            
            # Calculate positions in front view (x, y coordinates)
            front_px = int(front_x + px * view_size)
            front_py = int(front_y + (1-py) * view_size)  # Invert y for display
            
            # Draw particles
            cv2.circle(frame, (top_px, top_py), 2, (255, 180, 0), -1)  # Yellow dot
            cv2.circle(frame, (front_px, front_py), 2, (255, 180, 0), -1)  # Yellow dot

def run_frames_renderer(grid_size=(128, 128, 128), particles_count=1000, 
                       simulation_time=5.0, save_interval=3.0,
                       output_dir="frames"):
    """Run the frame-by-frame renderer."""
    renderer = FramesRenderer(
        output_dir=output_dir,
        grid_size=grid_size,
        particles_count=particles_count,
        simulation_time=simulation_time,
        save_interval=save_interval
    )
    
    return renderer.run_simulation_and_render()

if __name__ == "__main__":
    # Direct test run
    run_frames_renderer()