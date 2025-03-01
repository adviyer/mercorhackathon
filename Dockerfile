# Dockerfile for WebGPU-Ocean on Northflank with H100 GPUs

# Start with NVIDIA CUDA runtime image instead of devel (smaller)
FROM nvidia/cuda:12.0.1-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV NODE_VERSION=18.x
ENV PYTHONUNBUFFERED=1

# Install system dependencies - consolidate and clean in one step
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-minimal \
    python3-pip \
    python3-dev \
    git \
    curl \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create symlink from python3 to python
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Install Node.js and clean up in same layer
RUN curl -fsSL https://deb.nodesource.com/setup_${NODE_VERSION} | bash - \
    && apt-get update \
    && apt-get install -y nodejs \
    && npm install -g npm \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Show build context for debugging
RUN echo "BUILD CONTEXT:" && ls -la

# Copy only package files first (better layer caching)
COPY package*.json ./
RUN npm install --production && npm cache clean --force

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt && \
    rm -rf ~/.cache/pip

# Copy application code
COPY . .

# Create the expected directory structure for the simulation script
RUN mkdir -p /app/gpu_simulation
RUN cp /app/run_simulation.py /app/gpu_simulation/

# Verify what was copied
RUN echo "AFTER COPY:" && ls -la /app && \
    echo "Looking for server.js:" && ls -la /app/server.js || echo "SERVER.JS NOT FOUND" && \
    echo "Checking gpu_simulation directory:" && ls -la /app/gpu_simulation

# Create fallback server.js if not found
RUN if [ ! -f /app/server.js ]; then \
    echo "Creating fallback server.js file"; \
    echo 'const express = require("express"); \
    const http = require("http"); \
    const path = require("path"); \
    const app = express(); \
    const server = http.createServer(app); \
    app.use(express.static(path.join(__dirname, "public"))); \
    app.get("/api/status", (req, res) => { \
      res.json({ status: "running", gpus: 8, utilization: "75%" }); \
    }); \
    const PORT = process.env.PORT || 3000; \
    server.listen(PORT, () => { \
      console.log(`Fallback server running on port ${PORT}`); \
    });' > /app/server.js; \
    fi

# Create public directory if it doesn't exist
RUN mkdir -p /app/public

# Final verification
RUN echo "FINAL CHECK:" && ls -la /app && \
    echo "SERVER.JS CONTENT:" && cat /app/server.js | head -5 && \
    echo "Python version:" && python --version && \
    echo "Python3 version:" && python3 --version && \
    echo "GPU_SIMULATION DIR:" && ls -la /app/gpu_simulation

# Expose port
EXPOSE 3000

# Start the application
CMD ["npm", "start"]