# run_simulation.py
# This file handles the GPU-accelerated ocean simulation

import sys
import json
import numpy as np
import torch
import cv2
import time
import base64
from io import BytesIO
from PIL import Image

# Check for GPU availability
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}", file=sys.stderr)
else:
    device = torch.device('cpu')
    print("CUDA not available, using CPU", file=sys.stderr)

class OceanSimulator:
    def __init__(self, width, height, params):
        self.width = width
        self.height = height
        self.params = params
        
        # Initialize simulation parameters
        self.wave_scale = params.get('waveScale', 0.5)
        self.choppiness = params.get('choppiness', 1.5)
        self.wind_direction = params.get('windDirection', [1.0, 1.0])
        self.wind_speed = params.get('windSpeed', 8.0)
        
        # Create height and displacement maps
        self.height_map = torch.zeros((height, width), device=device)
        self.displacement_x = torch.zeros((height, width), device=device)
        self.displacement_z = torch.zeros((height, width), device=device)
        
        # Initialize FFT plans - use PyTorch's FFT functionality
        self.initialize_spectrum()
    
    def initialize_spectrum(self):
        # Create frequency grid
        x = torch.arange(-self.width/2, self.width/2, device=device) * (2.0 * np.pi / self.width)
        z = torch.arange(-self.height/2, self.height/2, device=device) * (2.0 * np.pi / self.height)
        self.k_x = x.repeat(self.height, 1)
        self.k_z = z.view(-1, 1).repeat(1, self.width)
        
        # Compute wave spectrum
        self.compute_phillips_spectrum()
        
    def compute_phillips_spectrum(self):
        # Wind direction normalization
        wind_x, wind_z = self.wind_direction
        wind_norm = np.sqrt(wind_x**2 + wind_z**2)
        wind_x /= wind_norm
        wind_z /= wind_norm
        
        # Avoid division by zero
        k_mag = torch.sqrt(self.k_x**2 + self.k_z**2)
        k_mag[k_mag < 1e-7] = 1e-7
        
        # Compute wind alignment term
        wind_alignment = self.k_x * wind_x + self.k_z * wind_z
        wind_alignment = wind_alignment / k_mag
        
        # Phillips spectrum
        L = self.wind_speed**2 / 9.81  # Largest wave from wind speed
        phillips = torch.zeros_like(k_mag)
        
        # Main spectrum calculation
        mask = (torch.abs(wind_alignment) > 0.01) & (k_mag > 0)
        phillips[mask] = (
            0.0081 * torch.exp(-1.0 / (k_mag[mask] * L)**2) / 
            k_mag[mask]**4 * 
            wind_alignment[mask]**2
        )
        
        # Generate random complex spectrum
        h0_real = torch.randn_like(phillips) * torch.sqrt(phillips * 0.5)
        h0_imag = torch.randn_like(phillips) * torch.sqrt(phillips * 0.5)
        self.h0 = torch.complex(h0_real, h0_imag)
        
    def simulate_step(self, t):
        # Time-dependent phase
        omega = torch.sqrt(9.81 * k_mag)
        phase = omega * t
        
        # Generate height field at time t
        h_tilde = self.h0 * torch.exp(torch.complex(torch.zeros_like(phase), phase))
        h_tilde_conj = torch.conj(torch.flip(torch.flip(h_tilde, [0]), [1]))
        
        # Ensure symmetry for real output
        h_tilde_conj[0, 0] = torch.complex(0.0, 0.0)
        
        ht = torch.fft.ifft2(h_tilde + h_tilde_conj).real * self.wave_scale
        
        # Compute displacement
        kx_ht = torch.fft.ifft2(torch.complex(0.0, 1.0) * self.k_x * h_tilde).real
        kz_ht = torch.fft.ifft2(torch.complex(0.0, 1.0) * self.k_z * h_tilde).real
        
        # Apply choppiness
        dx = kx_ht * self.choppiness
        dz = kz_ht * self.choppiness
        
        return ht, dx, dz
        
    def render_frame(self, t):
        # Simulate ocean at time t
        height, disp_x, disp_z = self.simulate_step(t)
        
        # Convert to numpy arrays for rendering
        height_np = height.cpu().numpy()
        disp_x_np = disp_x.cpu().numpy()
        disp_z_np = disp_z.cpu().numpy()
        
        # Normalize for visualization
        height_vis = ((height_np - height_np.min()) / (height_np.max() - height_np.min()) * 255).astype(np.uint8)
        
        # Create a colored height map
        colored_map = cv2.applyColorMap(height_vis, cv2.COLORMAP_OCEAN)
        
        # Create normal map for lighting
        normal_x = cv2.Sobel(height_np, cv2.CV_32F, 1, 0, ksize=3)
        normal_z = cv2.Sobel(height_np, cv2.CV_32F, 0, 1, ksize=3)
        normal_y = np.ones_like(normal_x)
        
        # Normalize normal vectors
        norm = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
        normal_x /= norm
        normal_y /= norm
        normal_z /= norm
        
        # Create RGB normal map
        normal_map = np.stack([
            ((normal_x + 1) / 2 * 255).astype(np.uint8),
            ((normal_y + 1) / 2 * 255).astype(np.uint8),
            ((normal_z + 1) / 2 * 255).astype(np.uint8)
        ], axis=2)
        
        # Combine height and normal maps for visualization
        # In a real app, you'd use this data for advanced rendering
        result = cv2.addWeighted(colored_map, 0.6, normal_map, 0.4, 0)
        
        # Convert to base64 for sending over websocket
        img = Image.fromarray(result)
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=70)
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Create metadata
        metadata = {
            'time': t,
            'min_height': float(height_np.min()),
            'max_height': float(height_np.max()),
            'avg_displacement_x': float(np.mean(np.abs(disp_x_np))),
            'avg_displacement_z': float(np.mean(np.abs(disp_z_np)))
        }
        
        return {
            'image': img_str,
            'metadata': metadata
        }

def main():
    # Parse parameters from command line
    if len(sys.argv) > 1:
        params = json.loads(sys.argv[1])
    else:
        # Default parameters
        params = {
            'width': 512,
            'height': 512,
            'waveScale': 0.5,
            'choppiness': 1.5,
            'windDirection': [1.0, 1.0],
            'windSpeed': 8.0,
            'frames': 10,
            'fps': 30
        }

    width = params.get('width', 512)
    height = params.get('height', 512)
    frames = params.get('frames', 10)
    fps = params.get('fps', 30)
    
    # Initialize simulator
    simulator = OceanSimulator(width, height, params)
    
    # Generate and output frames
    for i in range(frames):
        t = i / fps
        frame_data = simulator.render_frame(t)
        
        # Output as JSON for the Node.js server to pick up
        print(json.dumps(frame_data))
        sys.stdout.flush()
        
        # Control the simulation speed if needed
        time.sleep(1/fps)

if __name__ == "__main__":
    main()