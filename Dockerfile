FROM python:3.11-slim

LABEL org.opencontainers.image.title="benchb0t"
LABEL org.opencontainers.image.description="LLM agent benchmark framework"
LABEL org.opencontainers.image.source="https://github.com/benchb0t/benchb0t"

# curl is needed by some level evaluation checks run in this image
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[dev]" 2>/dev/null || pip install --no-cache-dir -e .

# Copy project (levels, harnesses, framework, config)
COPY . .

# Verify install
RUN benchbot --version

EXPOSE 7860

CMD ["benchbot", "dash", "--host", "0.0.0.0", "--port", "7860"]
