FROM python:3.11-slim

LABEL org.opencontainers.image.title="benchb0t"
LABEL org.opencontainers.image.description="LLM agent benchmark framework"
LABEL org.opencontainers.image.source="https://github.com/benchb0t/benchb0t"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# curl is needed by some level evaluation checks run in this image
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only the files needed to install the package first so dependency layers cache well.
COPY pyproject.toml README.md ./
COPY framework ./framework

RUN python -m pip install --no-cache-dir .

# Copy runtime data files after install.
COPY config.yaml ./
COPY levels ./levels
COPY harnesses ./harnesses

# Verify install
RUN benchbot --version

EXPOSE 7860

CMD ["benchbot", "dash", "--host", "0.0.0.0", "--port", "7860"]
