# Qwen3-TTS Voice Cloning API

High-performance text-to-speech engine with zero-shot voice cloning capabilities based on Qwen3-TTS.

---

## Prerequisites

* **Docker** installed.
* **NVIDIA Container Toolkit** (for GPU acceleration).
* A `.safetensors` voice embedding file.

---

## Usage

### 1. Build the Image

Run this command from the directory containing the `Dockerfile` and the script:

```bash
docker build -t qwen-tts-rt .

```

### 2. Run - Option A: Remote Download (Recommended for Cloud)

Use this option to automatically download the voice file from a URL at startup. This is ideal for serverless deployments like RunPod or Lambda.

```bash
docker run --gpus all -p 8000:8000 \
  -e VOICE_URL="https://your-storage.com/voice.safetensors" \
  qwen-tts-rt

```

### 3. Run - Option B: Local File

Use this option if you already have the `voice.safetensors` file on your local machine.

```bash
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/voice.safetensors:/app/voice.safetensors \
  qwen-tts-rt

```

---

## API Endpoint

**POST** `/tts`

**Payload:**

```json
{
  "text": "Hello, this is a cloned voice testing the system.",
  "language": "en"
}

```

**Response:**
Returns a `StreamingResponse` containing the generated `.wav` audio file.