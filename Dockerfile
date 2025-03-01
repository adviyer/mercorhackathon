FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

LABEL maintainer="Fluid Simulation"
LABEL description="3D Fluid Simulation for H100 GPUs on Northflank"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libegl1 \
    ffmpeg \
    wget \
    git \
    unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel
RUN pip3 install --no-cache-dir numpy taichi opencv-python-headless

# Set up working directory
WORKDIR /app

# Copy simulation files
COPY fluid_physics.py /app/
COPY frames_renderer.py /app/
COPY run_simulation.py /app/

# Create directories for output
RUN mkdir -p /app/simulation_data /app/frames

# Default command to run the simulation
ENTRYPOINT ["python3", "run_simulation.py"]
CMD ["--grid", "128", "--particles", "1000", "--duration", "5.0", "--save-interval", "3.0", "--output-dir", "/app/frames"]