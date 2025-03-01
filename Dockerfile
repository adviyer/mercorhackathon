# Dockerfile for Code Generator Service
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=3000

# Create app directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir fastapi uvicorn httpx pydantic

# Create the code_generator_service.py file
COPY code_generator_service.py /app/

# Create a simple server.js file for Node.js compatibility
RUN echo 'const { spawn } = require("child_process"); \
console.log("Starting Python FastAPI application..."); \
const pythonProcess = spawn("python3", ["-m", "uvicorn", "code_generator_service:app", "--host", "0.0.0.0", "--port", process.env.PORT || 3000]); \
pythonProcess.stdout.on("data", (data) => { console.log(`${data}`); }); \
pythonProcess.stderr.on("data", (data) => { console.error(`${data}`); }); \
pythonProcess.on("close", (code) => { console.log(`Python process exited with code ${code}`); process.exit(code); });' > /app/server.js

# Create a package.json file for Node.js compatibility
RUN echo '{"name":"code-generator-service","version":"1.0.0","scripts":{"start":"node server.js"}}' > /app/package.json

# Install Node.js
RUN apt-get update && apt-get install -y --no-install-recommends \
    nodejs \
    npm \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Expose port
EXPOSE 3000

# Set environment variables
ENV DEEPSEEK_API_URL="https://p01--deepseek-ollama--8bsw8fx29k5g.code.run:11434/v1/chat/completions"
ENV MAX_RETRIES=3
ENV TIMEOUT_SECONDS=120

# Start command
CMD ["node", "server.js"]