import os
import requests
import subprocess

VOICE_URL = os.environ.get("VOICE_URL")
VOICE_LIB_DIR = "./voice_library"
VOICE_NAME = "custom_voice" 

os.makedirs(VOICE_LIB_DIR, exist_ok=True)

target_file = os.path.join(VOICE_LIB_DIR, f"{VOICE_NAME}.safetensors")

if VOICE_URL:
    print(f"⬇️ Downloading voice from {VOICE_URL}...")
    try:
        with requests.get(VOICE_URL, stream=True) as r:
            r.raise_for_status()
            with open(target_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"✅ Voice saved as '{VOICE_NAME}' in library.")
    except Exception as e:
        print(f"❌ Error downloading voice: {e}")
else:
    print("⚠️ No VOICE_URL provided. Using default voices only.")

print("🚀 Starting High-Performance TTS Server...")

cmd = [
    "python", "-m", "api.main",
]

env = os.environ.copy()
env["HOST"] = "0.0.0.0"
env["PORT"] = "8000"
env["TTS_MODEL_NAME"] = "Qwen/Qwen3-TTS-12Hz-1.7B-Base" 
env["TTS_BACKEND"] = "official" # Usa Flash Attention 2 automaticamente

subprocess.run(cmd, env=env)