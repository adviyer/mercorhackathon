# Dockerfile for WebGPU-Ocean on Northflank with H100 GPUs
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV NODE_VERSION=18.x
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python-is-python3 \  # This explicitly creates the python->python3 symlink
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_${NODE_VERSION} | bash - \
    && apt-get update \
    && apt-get install -y nodejs \
    && npm install -g npm \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install Python dependencies
RUN pip install opencv-python Pillow

# Copy files
COPY . .

# Create the expected directory structure and ensure script is there
RUN mkdir -p /app/gpu_simulation && \
    cp /app/run_simulation.py /app/gpu_simulation/ && \
    chmod +x /app/gpu_simulation/run_simulation.py

# Install Node.js dependencies
RUN npm install --production

# Debug: verify everything is in place
RUN echo "Python version:" && python --version && \
    echo "Python3 version:" && python3 --version && \
    echo "CUDA available:" && python -c "import torch; print(torch.cuda.is_available())" && \
    echo "GPU_SIMULATION DIR:" && ls -la /app/gpu_simulation

# Expose port
EXPOSE 3000

# Start the application
CMD ["npm", "start"]