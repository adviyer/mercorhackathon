import numpy as np
import moderngl
import moderngl_window as mglw
from moderngl_window.meta import WindowConfig
from moderngl_window.scene import Material
from moderngl_window.resources.meta import texture
import time
from pathlib import Path
import os
import cv2

from fluid_physics import FluidSimulation

class FluidRenderer(WindowConfig):
    gl_version = (4, 3)
    title = "3D Fluid Simulation"
    resource_dir = Path(__file__).parent
    aspect_ratio = 16 / 9
    window_size = (1920, 1080)
    samples = 4
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Configure OpenGL
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        
        # Set up video writer
        self.video_writer = None
        self.output_video_path = "fluid_simulation.mp4"
        self.setup_video_writer()
        
        # Initialize the physics simulation
        self.simulation = FluidSimulation(
            grid_size=(128, 128, 128),
            particles_count=1000,
            simulation_time=5.0,
            save_interval=3.0
        )
        
        # Run physics simulation
        print("Running physics simulation...")
        simulation_results = self.simulation.run_simulation()
        self.density_frames = simulation_results['density_frames']
        self.particle_frames = simulation_results['particle_frames']
        self.grid_size = simulation_results['grid_size']
        self.dt = simulation_results['dt']
        self.frame_count = len(self.density_frames)
        
        print(f"Simulation complete. Rendering {self.frame_count} frames...")
        
        # Keep track of frames
        self.current_frame = 0
        
        # Create shaders and programs
        self.create_programs()
        
        # Create volumetric data texture
        self.density_texture = self.ctx.texture3d(
            self.grid_size, 1, dtype='f4'
        )
        
        # Set up the particle rendering
        self.setup_particles()
        
        # Create grid and axes for reference
        self.create_grid()
        
        print("Renderer initialized, starting rendering...")
    
    def setup_video_writer(self):
        """Set up the video writer for saving frames."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.output_video_path,
            fourcc,
            30,  # FPS
            self.window_size
        )
        
        # Create directory for individual frames if needed
        os.makedirs("frames", exist_ok=True)
    
    def create_programs(self):
        """Create the shader programs."""
        # Shader for volumetric rendering
        self.volume_program = self.ctx.program(
            vertex_shader='''
                #version 430
                
                in vec3 in_position;
                in vec3 in_normal;
                in vec2 in_texcoord_0;
                
                uniform mat4 m_proj;
                uniform mat4 m_view;
                uniform mat4 m_model;
                
                out vec3 position;
                out vec3 normal;
                out vec2 uv;
                
                void main() {
                    vec4 p = m_model * vec4(in_position, 1.0);
                    gl_Position = m_proj * m_view * p;
                    
                    position = p.xyz;
                    normal = mat3(m_model) * in_normal;
                    uv = in_texcoord_0;
                }
            ''',
            fragment_shader='''
                #version 430
                
                in vec3 position;
                in vec3 normal;
                in vec2 uv;
                
                uniform sampler3D volume_texture;
                uniform vec3 grid_size;
                uniform vec3 eye_position;
                
                out vec4 fragColor;
                
                void main() {
                    // Ray direction (from camera to fragment)
                    vec3 ray_dir = normalize(position - eye_position);
                    
                    // Starting point is the position on the cube
                    vec3 ray_pos = position;
                    
                    // Transform to normalized volume coordinates [0,1]
                    vec3 inv_size = 1.0 / grid_size;
                    ray_pos = (ray_pos + 0.5) * inv_size;
                    
                    // Ray marching parameters
                    const int MAX_STEPS = 128;
                    const float STEP_SIZE = 0.01;
                    
                    // Accumulated color and opacity
                    vec4 color = vec4(0.0);
                    
                    // Ray march through the volume
                    for (int i = 0; i < MAX_STEPS; i++) {
                        // Current position in volume (normalized)
                        vec3 pos = ray_pos + ray_dir * STEP_SIZE * float(i);
                        
                        // Check if we're outside the volume
                        if (pos.x < 0.0 || pos.x > 1.0 || 
                            pos.y < 0.0 || pos.y > 1.0 || 
                            pos.z < 0.0 || pos.z > 1.0) {
                            break;
                        }
                        
                        // Sample the volume
                        float density = texture(volume_texture, pos).r;
                        
                        // Skip empty space
                        if (density < 0.01) continue;
                        
                        // Color based on density and position
                        vec3 col = mix(vec3(0.1, 0.2, 0.8), vec3(0.8, 0.9, 1.0), density);
                        
                        // Accumulate color and opacity
                        float alpha = density * 0.1; // Adjust for better visibility
                        color.rgb += (1.0 - color.a) * col * alpha;
                        color.a += (1.0 - color.a) * alpha;
                        
                        // Early exit if we've accumulated enough
                        if (color.a > 0.95) break;
                    }
                    
                    // Output color with proper alpha blending
                    fragColor = color;
                }
            '''
        )
        
        # Shader for particle rendering
        self.particle_program = self.ctx.program(
            vertex_shader='''
                #version 430
                
                in vec3 in_position;
                
                uniform mat4 m_proj;
                uniform mat4 m_view;
                uniform float point_size;
                
                void main() {
                    gl_Position = m_proj * m_view * vec4(in_position, 1.0);
                    gl_PointSize = point_size;
                }
            ''',
            fragment_shader='''
                #version 430
                
                out vec4 fragColor;
                
                void main() {
                    // Create circular point
                    vec2 coord = gl_PointCoord - vec2(0.5);
                    if (length(coord) > 0.5) {
                        discard;
                    }
                    
                    // Add some lighting effect
                    float dist = length(coord);
                    float intensity = 1.0 - dist * 2.0;
                    
                    fragColor = vec4(1.0, 0.7, 0.3, intensity * 0.8);
                }
            '''
        )
    
    def setup_particles(self):
        """Set up particle rendering."""
        particles = self.particle_frames[0]
        self.particles_vbo = self.ctx.buffer(particles.astype('f4').tobytes())
        self.particles_vao = self.ctx.vertex_array(
            self.particle_program,
            [(self.particles_vbo, '3f', 'in_position')]
        )
    
    def create_grid(self):
        """Create a reference grid."""
        # Simple cube as reference
        vertices = np.array([
            # Front face
            -0.5, -0.5,  0.5, 0.0, 0.0, 1.0, 0.0, 0.0,  # Bottom-left
             0.5, -0.5,  0.5, 0.0, 0.0, 1.0, 1.0, 0.0,  # Bottom-right
             0.5,  0.5,  0.5, 0.0, 0.0, 1.0, 1.0, 1.0,  # Top-right
            -0.5,  0.5,  0.5, 0.0, 0.0, 1.0, 0.0, 1.0,  # Top-left
            
            # Back face
            -0.5, -0.5, -0.5, 0.0, 0.0, -1.0, 1.0, 0.0,  # Bottom-right
             0.5, -0.5, -0.5, 0.0, 0.0, -1.0, 0.0, 0.0,  # Bottom-left
             0.5,  0.5, -0.5, 0.0, 0.0, -1.0, 0.0, 1.0,  # Top-left
            -0.5,  0.5, -0.5, 0.0, 0.0, -1.0, 1.0, 1.0,  # Top-right
            
            # Left face
            -0.5, -0.5, -0.5, -1.0, 0.0, 0.0, 0.0, 0.0,  # Bottom-left
            -0.5, -0.5,  0.5, -1.0, 0.0, 0.0, 1.0, 0.0,  # Bottom-right
            -0.5,  0.5,  0.5, -1.0, 0.0, 0.0, 1.0, 1.0,  # Top-right
            -0.5,  0.5, -0.5, -1.0, 0.0, 0.0, 0.0, 1.0,  # Top-left
            
            # Right face
             0.5, -0.5,  0.5, 1.0, 0.0, 0.0, 0.0, 0.0,  # Bottom-left
             0.5, -0.5, -0.5, 1.0, 0.0, 0.0, 1.0, 0.0,  # Bottom-right
             0.5,  0.5, -0.5, 1.0, 0.0, 0.0, 1.0, 1.0,  # Top-right
             0.5,  0.5,  0.5, 1.0, 0.0, 0.0, 0.0, 1.0,  # Top-left
            
            # Bottom face
            -0.5, -0.5, -0.5, 0.0, -1.0, 0.0, 0.0, 0.0,  # Bottom-left
             0.5, -0.5, -0.5, 0.0, -1.0, 0.0, 1.0, 0.0,  # Bottom-right
             0.5, -0.5,  0.5, 0.0, -1.0, 0.0, 1.0, 1.0,  # Top-right
            -0.5, -0.5,  0.5, 0.0, -1.0, 0.0, 0.0, 1.0,  # Top-left
            
            # Top face
            -0.5,  0.5,  0.5, 0.0, 1.0, 0.0, 0.0, 0.0,  # Bottom-left
             0.5,  0.5,  0.5, 0.0, 1.0, 0.0, 1.0, 0.0,  # Bottom-right
             0.5,  0.5, -0.5, 0.0, 1.0, 0.0, 1.0, 1.0,  # Top-right
            -0.5,  0.5, -0.5, 0.0, 1.0, 0.0, 0.0, 1.0,  # Top-left
        ], dtype='f4')
        
        indices = np.array([
            0, 1, 2, 2, 3, 0,       # Front face
            4, 5, 6, 6, 7, 4,       # Back face
            8, 9, 10, 10, 11, 8,    # Left face
            12, 13, 14, 14, 15, 12, # Right face
            16, 17, 18, 18, 19, 16, # Bottom face
            20, 21, 22, 22, 23, 20  # Top face
        ], dtype='i4')
        
        self.grid_vbo = self.ctx.buffer(vertices)
        self.grid_ibo = self.ctx.buffer(indices)
        self.grid_vao = self.ctx.vertex_array(
            self.volume_program,
            [
                (self.grid_vbo, '3f 3f 2f', 'in_position', 'in_normal', 'in_texcoord_0')
            ],
            self.grid_ibo
        )
    
    def render(self, time, frame_time):
        """Render a frame."""
        # Clear the framebuffer
        self.ctx.clear(0.05, 0.05, 0.05, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        
        # Check if we've finished rendering all frames
        if self.current_frame >= self.frame_count:
            # Close video writer and exit when done
            if self.video_writer is not None:
                self.video_writer.release()
                print(f"Video saved to {self.output_video_path}")
            self.wnd.close()
            return
            
        # Update density texture with current frame data
        density_data = self.density_frames[self.current_frame]
        self.density_texture.write(density_data.astype('f4').tobytes())
        
        # Update particle positions
        particles = self.particle_frames[self.current_frame]
        self.particles_vbo.write(particles.astype('f4').tobytes())
        
        # Set up camera and view
        proj = self.camera.projection.matrix
        view = self.camera.matrix
        
        # Fixed camera position looking at the center of the simulation
        eye_pos = np.array([1.5, 1.5, 1.5], dtype='f4')
        target = np.array([0.0, 0.0, 0.0], dtype='f4')
        up = np.array([0.0, 1.0, 0.0], dtype='f4')
        
        # Build view matrix manually for fixed camera position
        z_axis = eye_pos - target
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        y_axis = np.cross(z_axis, x_axis)
        
        view = np.array([
            [x_axis[0], y_axis[0], z_axis[0], 0.0],
            [x_axis[1], y_axis[1], z_axis[1], 0.0],
            [x_axis[2], y_axis[2], z_axis[2], 0.0],
            [-np.dot(x_axis, eye_pos), -np.dot(y_axis, eye_pos), -np.dot(z_axis, eye_pos), 1.0]
        ], dtype='f4')
        
        # Render the volume
        self.volume_program['m_proj'].write(proj.astype('f4').tobytes())
        self.volume_program['m_view'].write(view.astype('f4').tobytes())
        self.volume_program['m_model'].write(np.eye(4, dtype='f4').tobytes())
        self.volume_program['grid_size'].write(np.array(self.grid_size, dtype='f4').tobytes())
        self.volume_program['eye_position'].write(eye_pos.tobytes())
        
        # Bind the density texture
        self.density_texture.use(0)
        self.volume_program['volume_texture'] = 0
        
        # Draw the cube for volume rendering (wireframe)
        self.ctx.wireframe = True
        self.grid_vao.render()
        self.ctx.wireframe = False
        
        # Draw particles
        self.particle_program['m_proj'].write(proj.astype('f4').tobytes())
        self.particle_program['m_view'].write(view.astype('f4').tobytes())
        self.particle_program['point_size'] = 10.0  # Adjust size as needed
        
        # Enable point sprites and draw particles
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.particles_vao.render(moderngl.POINTS)
        
        # Capture frame for video
        self.capture_frame()
        
        # Move to next frame
        self.current_frame += 1
        
    def capture_frame(self):
        """Capture the current frame for video output."""
        # Read pixels from framebuffer
        pixels = self.ctx.fbo.read(components=4, attachment=0)
        img = np.frombuffer(pixels, dtype=np.uint8).reshape(self.window_size[1], self.window_size[0], 4)
        
        # Convert RGBA to BGR for OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
        # Flip the image vertically (OpenGL vs OpenCV coordinates)
        img = cv2.flip(img, 0)
        
        # Write frame to video file
        if self.video_writer is not None:
            self.video_writer.write(img)
        
        # Save individual frame if desired
        # cv2.imwrite(f"frames/frame_{self.current_frame:04d}.png", img)

if __name__ == "__main__":
    # Run the application
    mglw.run_window_config(FluidRenderer)