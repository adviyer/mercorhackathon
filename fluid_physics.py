"""
Fluid physics simulation using Taichi for GPU acceleration.
Includes fallback to CPU if GPU is not available.
"""
import numpy as np
import time
import pickle
import os

# Initialize Taichi with CPU as fallback if GPU fails
try:
    import taichi as ti
    # Try to initialize with GPU
    try:
        ti.init(arch=ti.gpu, device_memory_GB=2)
        print("Taichi initialized with GPU")
    except Exception as e:
        print(f"Could not initialize Taichi with GPU: {e}")
        print("Falling back to CPU")
        ti.init(arch=ti.cpu)
except ImportError:
    print("Taichi not available, using NumPy fallback")
    # Will use NumPy for computation instead

class FluidSimulation:
    def __init__(self, 
                 grid_size=(128, 128, 128),
                 domain_size=(1.0, 1.0, 1.0),
                 viscosity=0.0001,
                 iterations=20,
                 dt=0.033,
                 particles_count=1000,
                 save_interval=3.0,
                 simulation_time=5.0):
        
        self.grid_size = grid_size
        self.domain_size = domain_size
        self.viscosity = viscosity
        self.iterations = iterations
        self.dt = dt
        self.particles_count = particles_count
        self.save_interval = save_interval
        self.simulation_time = simulation_time
        
        self.dx = domain_size[0] / grid_size[0]
        
        # Check if we're using Taichi
        self.using_taichi = 'ti' in globals()
        
        if self.using_taichi:
            # Fluid velocity field
            self.velocity = ti.Vector.field(3, dtype=ti.f32, shape=grid_size)
            self.velocity_next = ti.Vector.field(3, dtype=ti.f32, shape=grid_size)
            
            # Pressure field
            self.pressure = ti.field(dtype=ti.f32, shape=grid_size)
            self.divergence = ti.field(dtype=ti.f32, shape=grid_size)
            
            # Density field for visualization
            self.density = ti.field(dtype=ti.f32, shape=grid_size)
            self.density_next = ti.field(dtype=ti.f32, shape=grid_size)
            
            # Particles for tracking and visualization
            self.particles = ti.Vector.field(3, dtype=ti.f32, shape=particles_count)
            self.particle_velocities = ti.Vector.field(3, dtype=ti.f32, shape=particles_count)
        else:
            # NumPy fallback implementation
            self.velocity = np.zeros((*grid_size, 3), dtype=np.float32)
            self.velocity_next = np.zeros((*grid_size, 3), dtype=np.float32)
            
            self.pressure = np.zeros(grid_size, dtype=np.float32)
            self.divergence = np.zeros(grid_size, dtype=np.float32)
            
            self.density = np.zeros(grid_size, dtype=np.float32)
            self.density_next = np.zeros(grid_size, dtype=np.float32)
            
            self.particles = np.zeros((particles_count, 3), dtype=np.float32)
            self.particle_velocities = np.zeros((particles_count, 3), dtype=np.float32)
        
        # Setup directory for saving particle positions
        self.output_dir = "simulation_data"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize particles with a fixed seed for reproducibility
        self._initialize_particles()
        
        # Save the initial positions
        self._save_particle_positions(0.0)

    def _initialize_particles(self):
        # Set a fixed seed for reproducibility
        np.random.seed(42)
        
        # Initialize particles in a confined region for better visibility
        particle_positions = np.random.rand(self.particles_count, 3) * 0.5 + 0.25
        particle_positions = particle_positions.astype(np.float32)
        
        # Initialize with zero velocity
        particle_velocities = np.zeros((self.particles_count, 3), dtype=np.float32)
        
        if self.using_taichi:
            # Copy to Taichi fields
            self.particles.from_numpy(particle_positions)
            self.particle_velocities.from_numpy(particle_velocities)
        else:
            # Direct assignment for NumPy implementation
            self.particles = particle_positions
            self.particle_velocities = particle_velocities
        
        # Add an initial density source in the center
        self._add_density_source()
    
    def _add_density_source(self):
        center = [s // 2 for s in self.grid_size]
        radius = min(self.grid_size) // 8
        
        if self.using_taichi:
            @ti.kernel
            def initialize_density():
                for i, j, k in self.density:
                    # Distance from center
                    di = i - center[0]
                    dj = j - center[1]
                    dk = k - center[2]
                    dist_sq = di * di + dj * dj + dk * dk
                    
                    # Add density in a sphere
                    if dist_sq < radius * radius:
                        self.density[i, j, k] = 1.0
                        
                        # Add some initial velocity (upward motion)
                        self.velocity[i, j, k] = ti.Vector([0.0, 2.0, 0.0])
            
            initialize_density()
        else:
            # NumPy implementation
            for i in range(self.grid_size[0]):
                for j in range(self.grid_size[1]):
                    for k in range(self.grid_size[2]):
                        # Distance from center
                        di = i - center[0]
                        dj = j - center[1]
                        dk = k - center[2]
                        dist_sq = di * di + dj * dj + dk * dk
                        
                        # Add density in a sphere
                        if dist_sq < radius * radius:
                            self.density[i, j, k] = 1.0
                            
                            # Add some initial velocity (upward motion)
                            self.velocity[i, j, k] = np.array([0.0, 2.0, 0.0])

    def step(self):
        """Advance the simulation by one time step."""
        if self.using_taichi:
            # Taichi implementation (GPU accelerated)
            # Diffusion step for velocity
            self.diffuse(self.velocity, self.velocity_next, self.viscosity)
            self.velocity.copy_from(self.velocity_next)
            
            # Project velocity to be divergence-free
            self.compute_divergence()
            self.pressure.fill(0.0)
            for _ in range(self.iterations):
                self.pressure_jacobi_iteration()
            self.project_velocity()
            
            # Advection step for velocity
            self.advect_velocity()
            self.velocity.copy_from(self.velocity_next)
            
            # Project again after advection
            self.compute_divergence()
            self.pressure.fill(0.0)
            for _ in range(self.iterations):
                self.pressure_jacobi_iteration()
            self.project_velocity()
            
            # Advection step for density
            self.advect_density()
            self.density.copy_from(self.density_next)
            
            # Update particle positions
            self.update_particles()
        else:
            # NumPy implementation (CPU fallback)
            # Simplified version for CPU - just update some particles with random motion
            # This is a placeholder that creates visually interesting but physically simplified results
            self._simple_cpu_step()
    
    def _simple_cpu_step(self):
        """Simple CPU fallback for when Taichi is not available."""
        # Simple diffusion for density
        self.density_next = np.copy(self.density)
        for i in range(1, self.grid_size[0]-1):
            for j in range(1, self.grid_size[1]-1):
                for k in range(1, self.grid_size[2]-1):
                    self.density_next[i, j, k] = 0.9 * self.density[i, j, k] + \
                        0.1 * (self.density[i+1, j, k] + self.density[i-1, j, k] + \
                              self.density[i, j+1, k] + self.density[i, j-1, k] + \
                              self.density[i, j, k+1] + self.density[i, j, k-1]) / 6.0
        self.density = np.copy(self.density_next)
        
        # Simple particle movement (random walk + some gravity)
        for i in range(self.particles_count):
            # Add small random velocity changes
            self.particle_velocities[i] += np.random.normal(0, 0.01, 3)
            # Add gravity
            self.particle_velocities[i, 1] -= 0.001
            # Add slight tendency toward center
            center_dir = np.array([0.5, 0.5, 0.5]) - self.particles[i]
            self.particle_velocities[i] += 0.005 * center_dir
            
            # Apply velocity
            self.particles[i] += self.particle_velocities[i] * self.dt
            
            # Boundary conditions
            for dim in range(3):
                if self.particles[i, dim] < 0:
                    self.particles[i, dim] = 0
                    self.particle_velocities[i, dim] *= -0.5
                elif self.particles[i, dim] > 1:
                    self.particles[i, dim] = 1
                    self.particle_velocities[i, dim] *= -0.5
                    
            # Damping
            self.particle_velocities[i] *= 0.99
            
    def _save_particle_positions(self, time):
        """Save particle positions to file."""
        if self.using_taichi:
            # Convert to numpy for saving
            particle_positions = self.particles.to_numpy()
        else:
            # Already numpy array
            particle_positions = self.particles
        
        # Save to file
        filename = os.path.join(self.output_dir, f"particles_t{time:.1f}.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(particle_positions, f)
        
        print(f"Saved particle positions at time {time:.1f}s to {filename}")

    def run_simulation(self):
        """Run the full simulation and save particle positions at specified intervals."""
        print("Starting fluid simulation...")
        
        start_time = time.time()
        simulation_time = 0.0
        next_save_time = self.save_interval
        
        frame_count = 0
        
        # Return data for the renderer
        density_frames = []
        particle_frames = []
        
        while simulation_time < self.simulation_time:
            # Perform a simulation step
            self.step()
            
            # Update simulation time
            simulation_time += self.dt
            frame_count += 1
            
            # Save particle positions at intervals
            if simulation_time >= next_save_time:
                self._save_particle_positions(simulation_time)
                next_save_time += self.save_interval
            
            # Store frame data for rendering
            if self.using_taichi:
                density_frames.append(self.density.to_numpy())
                particle_frames.append(self.particles.to_numpy())
            else:
                density_frames.append(np.copy(self.density))
                particle_frames.append(np.copy(self.particles))
            
            # Print progress
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Simulation time: {simulation_time:.2f}s, Real time: {elapsed:.2f}s, Frame: {frame_count}")
        
        print(f"Simulation completed with {frame_count} frames.")
        return {
            'density_frames': density_frames,
            'particle_frames': particle_frames,
            'dt': self.dt,
            'grid_size': self.grid_size,
            'domain_size': self.domain_size
        }

    # All Taichi kernel methods defined here
    if 'ti' in globals():
        @ti.kernel
        def diffuse(self, field: ti.template(), field_next: ti.template(), diffusion_rate: float):
            for i, j, k in field:
                if 1 <= i < self.grid_size[0]-1 and 1 <= j < self.grid_size[1]-1 and 1 <= k < self.grid_size[2]-1:
                    field_next[i, j, k] = (
                        field[i, j, k] + diffusion_rate * (
                            field[i+1, j, k] + field[i-1, j, k] +
                            field[i, j+1, k] + field[i, j-1, k] +
                            field[i, j, k+1] + field[i, j, k-1] -
                            6 * field[i, j, k]
                        )
                    ) / (1 + 6 * diffusion_rate)

        @ti.kernel
        def advect_velocity(self):
            for i, j, k in self.velocity:
                if 1 <= i < self.grid_size[0]-1 and 1 <= j < self.grid_size[1]-1 and 1 <= k < self.grid_size[2]-1:
                    # Get current position
                    pos = ti.Vector([i, j, k]) - self.dt * self.velocity[i, j, k]
                    
                    # Clamp to grid boundaries
                    pos.x = max(0.5, min(self.grid_size[0] - 1.5, pos.x))
                    pos.y = max(0.5, min(self.grid_size[1] - 1.5, pos.y))
                    pos.z = max(0.5, min(self.grid_size[2] - 1.5, pos.z))
                    
                    # Integer indices for interpolation
                    i0, j0, k0 = int(pos.x), int(pos.y), int(pos.z)
                    i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1
                    
                    # Interpolation weights
                    s1 = pos.x - i0
                    s0 = 1 - s1
                    t1 = pos.y - j0
                    t0 = 1 - t1
                    u1 = pos.z - k0
                    u0 = 1 - u1
                    
                    # Trilinear interpolation
                    self.velocity_next[i, j, k] = (
                        s0 * (t0 * (u0 * self.velocity[i0, j0, k0] + u1 * self.velocity[i0, j0, k1]) +
                              t1 * (u0 * self.velocity[i0, j1, k0] + u1 * self.velocity[i0, j1, k1])) +
                        s1 * (t0 * (u0 * self.velocity[i1, j0, k0] + u1 * self.velocity[i1, j0, k1]) +
                              t1 * (u0 * self.velocity[i1, j1, k0] + u1 * self.velocity[i1, j1, k1]))
                    )

        @ti.kernel
        def advect_density(self):
            for i, j, k in self.density:
                if 1 <= i < self.grid_size[0]-1 and 1 <= j < self.grid_size[1]-1 and 1 <= k < self.grid_size[2]-1:
                    # Get current position
                    pos = ti.Vector([i, j, k]) - self.dt * self.velocity[i, j, k]
                    
                    # Clamp to grid boundaries
                    pos.x = max(0.5, min(self.grid_size[0] - 1.5, pos.x))
                    pos.y = max(0.5, min(self.grid_size[1] - 1.5, pos.y))
                    pos.z = max(0.5, min(self.grid_size[2] - 1.5, pos.z))
                    
                    # Integer indices for interpolation
                    i0, j0, k0 = int(pos.x), int(pos.y), int(pos.z)
                    i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1
                    
                    # Interpolation weights
                    s1 = pos.x - i0
                    s0 = 1 - s1
                    t1 = pos.y - j0
                    t0 = 1 - t1
                    u1 = pos.z - k0
                    u0 = 1 - u1
                    
                    # Trilinear interpolation
                    self.density_next[i, j, k] = (
                        s0 * (t0 * (u0 * self.density[i0, j0, k0] + u1 * self.density[i0, j0, k1]) +
                              t1 * (u0 * self.density[i0, j1, k0] + u1 * self.density[i0, j1, k1])) +
                        s1 * (t0 * (u0 * self.density[i1, j0, k0] + u1 * self.density[i1, j0, k1]) +
                              t1 * (u0 * self.density[i1, j1, k0] + u1 * self.density[i1, j1, k1]))
                    )

        @ti.kernel
        def compute_divergence(self):
            for i, j, k in self.divergence:
                if 1 <= i < self.grid_size[0]-1 and 1 <= j < self.grid_size[1]-1 and 1 <= k < self.grid_size[2]-1:
                    self.divergence[i, j, k] = 0.5 * (
                        self.velocity[i+1, j, k].x - self.velocity[i-1, j, k].x +
                        self.velocity[i, j+1, k].y - self.velocity[i, j-1, k].y +
                        self.velocity[i, j, k+1].z - self.velocity[i, j, k-1].z
                    ) / self.dx

        @ti.kernel
        def pressure_jacobi_iteration(self):
            for i, j, k in self.pressure:
                if 1 <= i < self.grid_size[0]-1 and 1 <= j < self.grid_size[1]-1 and 1 <= k < self.grid_size[2]-1:
                    self.pressure[i, j, k] = (
                        self.pressure[i+1, j, k] + self.pressure[i-1, j, k] +
                        self.pressure[i, j+1, k] + self.pressure[i, j-1, k] +
                        self.pressure[i, j, k+1] + self.pressure[i, j, k-1] -
                        self.divergence[i, j, k] * self.dx * self.dx
                    ) / 6.0

        @ti.kernel
        def project_velocity(self):
            for i, j, k in self.velocity:
                if 1 <= i < self.grid_size[0]-1 and 1 <= j < self.grid_size[1]-1 and 1 <= k < self.grid_size[2]-1:
                    self.velocity[i, j, k] -= 0.5 * ti.Vector([
                        self.pressure[i+1, j, k] - self.pressure[i-1, j, k],
                        self.pressure[i, j+1, k] - self.pressure[i, j-1, k],
                        self.pressure[i, j, k+1] - self.pressure[i, j, k-1]
                    ]) / self.dx

        @ti.kernel
        def update_particles(self):
            for i in range(self.particles_count):
                # Current position
                pos = self.particles[i]
                
                # Convert to grid coordinates
                gi = int(pos.x * self.grid_size[0])
                gj = int(pos.y * self.grid_size[1])
                gk = int(pos.z * self.grid_size[2])
                
                # Clamp to valid grid coordinates
                gi = max(1, min(self.grid_size[0] - 2, gi))
                gj = max(1, min(self.grid_size[1] - 2, gj))
                gk = max(1, min(self.grid_size[2] - 2, gk))
                
                # Sample velocity from grid
                self.particle_velocities[i] = self.velocity[gi, gj, gk]
                
                # Update position
                self.particles[i] += self.particle_velocities[i] * self.dt
                
                # Boundary conditions (bounce off walls)
                if self.particles[i].x < 0.0:
                    self.particles[i].x = 0.0
                    self.particle_velocities[i].x *= -0.5  # Damping
                elif self.particles[i].x > 1.0:
                    self.particles[i].x = 1.0
                    self.particle_velocities[i].x *= -0.5
                    
                if self.particles[i].y < 0.0:
                    self.particles[i].y = 0.0
                    self.particle_velocities[i].y *= -0.5
                elif self.particles[i].y > 1.0:
                    self.particles[i].y = 1.0
                    self.particle_velocities[i].y *= -0.5
                    
                if self.particles[i].z < 0.0:
                    self.particles[i].z = 0.0
                    self.particle_velocities[i].z *= -0.5
                elif self.particles[i].z > 1.0:
                    self.particles[i].z = 1.0
                    self.particle_velocities[i].z *= -0.5

if __name__ == "__main__":
    # This allows the physics to be run independently for testing
    sim = FluidSimulation(
        grid_size=(64, 64, 64),  # Smaller grid for testing
        particles_count=1000,
        simulation_time=5.0,
        save_interval=1.0
    )
    sim.run_simulation()