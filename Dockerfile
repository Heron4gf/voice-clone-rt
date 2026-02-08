FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git curl sox libsox-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/QwenLM/Qwen3-TTS.git /app/Qwen3-TTS
WORKDIR /app/Qwen3-TTS
RUN pip install --no-cache-dir -e .
RUN pip install --no-cache-dir fastapi uvicorn python-multipart soundfile safetensors requests

COPY server.py /app/Qwen3-TTS/server.py

EXPOSE 8000
CMD ["python", "server.py"]