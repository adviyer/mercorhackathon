FROM nvidia/cuda:12.0.1-runtime-ubuntu22.04

# Set up working directory
WORKDIR /app

# Install Python and required packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    nvcc \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create directories for simulation, results, and versions
RUN mkdir -p /app/fluid_simulation /app/optimization_results /app/optimization_results/versions /app/logs

# Create requirements.txt file
RUN echo "fastapi>=0.95.0\nuvicorn>=0.21.0\nhttpx>=0.24.0\npydantic>=1.10.7" > /app/requirements.txt

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY optimization_orchestrator.py /app/

# Expose the service port (3000 for Northflank)
EXPOSE 3000

# Set environment variables (these will be overridden at runtime via Northflank)
ENV CODE_GENERATOR_URL="http://code-generator-service:3000" \
    EVALUATOR_URL="http://evaluator-service:3000" \
    SIMULATION_DIR="/app/fluid_simulation" \
    RESULTS_DIR="/app/optimization_results" \
    MAX_ITERATIONS=10 \
    PERFORMANCE_THRESHOLD=1.05

# Create volume mount points for persistent data
VOLUME ["/app/fluid_simulation", "/app/optimization_results"]

# Run the service
CMD ["uvicorn", "optimization_orchestrator:app", "--host", "0.0.0.0", "--port", "3000"]