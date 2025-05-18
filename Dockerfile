# Use an official Python runtime with build tools
FROM python:3.9 as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Final lightweight image
FROM python:3.9-slim

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy application files
COPY . .

# Download the Ollama model during build (optional)
RUN python -c "from langchain_community.llms import Ollama; Ollama(model='phi3:mini')"

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "api_service:app", "--host", "0.0.0.0", "--port", "8000"]