FROM python:3.12-slim

LABEL maintainer="SaurabhJalendra"
LABEL description="Simulating Anything: Multi-agent scientific discovery engine"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ src/
COPY configs/ configs/
COPY tests/ tests/

# Install the package with all dependencies
RUN pip install --no-cache-dir -e ".[dev]"

# Default command: run tests
CMD ["python", "-m", "pytest", "tests/unit/", "-q", "--tb=short"]
