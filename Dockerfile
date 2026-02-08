FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    curl \
    sox \
    libsox-dev \
    && rm -rf /var/lib/apt/lists/*

# Clona il repository Qwen3-TTS
RUN git clone https://github.com/QwenLM/Qwen3-TTS.git /app/Qwen3-TTS

# Imposta la workdir dentro il repo clonato (importante per i path relativi)
WORKDIR /app/Qwen3-TTS

RUN pip install --no-cache-dir -e .
RUN pip install --no-cache-dir fastapi uvicorn python-multipart soundfile safetensors requests

COPY server.py /app/Qwen3-TTS/server.py

EXPOSE 8000

ENV VOICE_URL=""

CMD ["python", "server.py"]