FROM python:3.9-slim

# Set up non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:${PATH}"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN sudo apt-get update && \
    sudo apt-get install -y --no-install-recommends gcc python3-dev && \
    sudo rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy application files
COPY --chown=user . .

# Environment variables
ENV PYTHONPATH=/app \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HOME=/app/.cache/huggingface \
    GRADIO_ALLOW_FLAGGING=never

# Expose Hugging Face Space port
EXPOSE 7860

# Start application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]