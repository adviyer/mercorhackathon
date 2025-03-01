FROM python:3.9-slim

WORKDIR /app

# Create directories for logs and data
RUN mkdir -p /app/logs /app/data

# Create requirements.txt file
RUN echo "fastapi>=0.95.0\nuvicorn>=0.21.0\nhttpx>=0.24.0\npydantic>=1.10.7" > /app/requirements.txt

# Install required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY evaluator_service.py /app/

# Expose the service port (3000 for Northflank)
EXPOSE 3000

# Set environment variables (these can be overridden at runtime)
ENV DEEPSEEK_API_URL="https://p01--deepseek-ollama--8bsw8fx29k5g.code.run:11434/v1/chat/completions"
ENV MAX_RETRIES=3
ENV TIMEOUT_SECONDS=180

# Run the service
CMD ["uvicorn", "evaluator_service:app", "--host", "0.0.0.0", "--port", "3000"]