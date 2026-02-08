FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y git curl sox libsox-dev build-essential && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir flash-attn --no-build-isolation

RUN git clone https://github.com/groxaxo/Qwen3-TTS-Openai-Fastapi.git .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir requests

COPY start.py /app/start.py

EXPOSE 8000

CMD ["python", "start.py"]