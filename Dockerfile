FROM python:3.11-slim

# System libraries required for Pillow, PyMuPDF, and pandas wheels
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg62-turbo \
    libjpeg62-turbo-dev \
    zlib1g \
    zlib1g-dev \
    libpng16-16 \
    libpng-dev \
    libtiff6 \
    libtiff-dev \
    libfreetype6 \
    libfreetype6-dev \
    libwebp7 \
    libwebp-dev \
    liblcms2-2 \
    libopenjp2-7 \
    libxml2 \
    libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy source
COPY src /app/src
COPY prompts /app/prompts

# Default envs (override at runtime)
ENV PYTHONUNBUFFERED=1 \
    OPENAI_API_KEY="" \
    OUT_DIR=/app/out \
    MODEL_NAME=gpt-5 \
    SERVICE_TIER=priority

RUN mkdir -p /app/in /app/out

ENTRYPOINT ["python", "-m", "src.run_batch_pipeline"]
