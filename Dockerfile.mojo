# Mojo CI/CD Docker Configuration
# 
# Multi-stage Docker build for testing Mojo integration
# Supports both mock mode and real Mojo when available

# Stage 1: Base testing environment
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install testing dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    pytest-xdist \
    pytest-timeout \
    aiohttp \
    numpy \
    bandit \
    safety \
    mypy \
    black \
    flake8

# Stage 2: Development/Testing environment
FROM base as testing

# Copy source code
COPY . .

# Create test directories
RUN mkdir -p test-results/mojo logs

# Set environment variables for testing
ENV PYTHONPATH=/app
ENV MOJO_MOCK_MODE=true
ENV TEST_TIMEOUT=300
ENV PYTEST_WORKERS=4

# Expose MCP server port
EXPOSE 8004

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8004/health || exit 1

# Default command runs all tests
CMD ["python", "-m", "pytest", "tests/", "-v", "--tb=short"]

# Stage 3: Mojo-enabled environment (when Mojo is available)
FROM base as mojo-enabled

# This stage would include actual Mojo installation
# For now, it's a placeholder for when Mojo becomes available

# Install Mojo (placeholder - actual installation would go here)
# RUN curl -ssL https://get.modular.com | sh -
# RUN modular install mojo

# Set Mojo environment
ENV MOJO_MOCK_MODE=false
ENV MODULAR_HOME=/opt/modular
ENV PATH=$PATH:$MODULAR_HOME/bin

# Copy source code
COPY . .

# Create test directories
RUN mkdir -p test-results/mojo logs

# Set environment variables
ENV PYTHONPATH=/app

# Default command with real Mojo tests
CMD ["./scripts/run_mojo_tests.sh", "--level", "all", "--real"]

# Stage 4: Production-ready image
FROM base as production

# Copy only necessary files for production
COPY final_mcp_server.py .
COPY src/ src/
COPY templates/ templates/

# Set production environment
ENV PYTHONPATH=/app
ENV ENVIRONMENT=production
ENV LOG_LEVEL=INFO

# Expose MCP server port
EXPOSE 8004

# Run MCP server
CMD ["python", "final_mcp_server.py", "--host", "0.0.0.0", "--port", "8004", "--timeout", "3600"]
