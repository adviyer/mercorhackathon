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

# Install Node.js and clean up in same layer
RUN curl -fsSL https://deb.nodesource.com/setup_${NODE_VERSION} | bash - \
    && apt-get update \
    && apt-get install -y nodejs \
    && npm install -g npm \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# For debugging - Let's see what's in the build context
RUN echo "Listing build context files:" && ls -la

# Copy only package files first (better layer caching)
COPY package*.json ./
RUN npm install --production && npm cache clean --force

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt && \
    rm -rf ~/.cache/pip

# Copy application code - explicitly copy the server.js file
COPY server.js /app/server.js
COPY . .

# For debugging - Let's verify server.js exists
RUN ls -la /app && echo "Content of server.js:" && cat /app/server.js | head -5

# Expose port
EXPOSE 3000

# Start the application
CMD ["npm", "start"]