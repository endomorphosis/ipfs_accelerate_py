# Enterprise-Grade IPFS Accelerate Python Dockerfile
FROM python:3.12-slim

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV IPFS_ACCELERATE_ENV=production

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    libgmp-dev \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .
COPY setup.py .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -e .

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 ipfs_user && \
    chown -R ipfs_user:ipfs_user /app
USER ipfs_user

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD python deployments/health_check.py

# Expose default port
EXPOSE 8000

# Default command
CMD ["python", "-m", "ipfs_accelerate_py"]