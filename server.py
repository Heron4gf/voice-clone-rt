import os
import sys
import subprocess
import requests
import torch
from qwen_tts import Qwen3TTSModel
import safetensors.torch
import io
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import traceback

VOICE_FILENAME = "voice.safetensors"
VOICE_URL = os.environ.get("VOICE_URL")

if not os.path.exists(VOICE_FILENAME):
    if VOICE_URL:
        print(f"⬇️ {VOICE_FILENAME} not found. Downloading from URL...")
        try:
            with requests.get(VOICE_URL, stream=True) as r:
                r.raise_for_status()
                with open(VOICE_FILENAME, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192): 
                        f.write(chunk)
            print("✅ Download complete.")
        except Exception as e:
            print(f"❌ Critical Error: Failed to download voice file. {e}")
            sys.exit(1)
    else:
        print(f"❌ Error: {VOICE_FILENAME} missing and VOICE_URL environment variable is not set.")
        sys.exit(1)
else:
    print(f"✅ Found local {VOICE_FILENAME}.")

try:
    subprocess.run(["sox", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
except FileNotFoundError:
    print("⚠️ Warning: SoX not found. Audio generation might fail.")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Device: {device}")

print("⏳ Loading model (0.6B)...")
try:
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device
    )
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    sys.exit(1)

print("⏳ Loading voice embedding...")
try:
    voice_embedding = safetensors.torch.load_file(VOICE_FILENAME)
except Exception as e:
    print(f"❌ Failed to load safetensors: {e}")
    sys.exit(1)

voice_clone_prompt = {}
print("🔧 Patching Tensor Shapes:")

for k, v in voice_embedding.items():
    if isinstance(v, torch.Tensor):
        v = v.to(device)
        old_shape = str(list(v.shape))
        
        if v.ndim == 1:
            v = v.unsqueeze(0) 
        elif v.ndim == 2 and "id" not in k and "mask" not in k:
            v = v.unsqueeze(0) 
            
        if str(list(v.shape)) != old_shape:
            print(f"   - {k}: {old_shape} -> {list(v.shape)}")
        
    voice_clone_prompt[k] = v

if "x_vector_only_mode" not in voice_clone_prompt:
    voice_clone_prompt["x_vector_only_mode"] = [False]
if "icl_mode" not in voice_clone_prompt:
    voice_clone_prompt["icl_mode"] = [False]

print("✅ Voice prompt ready.")

app = FastAPI()

class TTSRequest(BaseModel):
    text: str
    language: str = "en"

@app.post("/tts")
async def tts(request: TTSRequest):
    print(f"📩 Req: {request.text[:40]}...")
    try:
        wavs, sr = model.generate_voice_clone(
            voice_clone_prompt=voice_clone_prompt,
            text=request.text
        )
        
        audio_data = wavs[0] if isinstance(wavs, list) else wavs
        if isinstance(audio_data, torch.Tensor):
            audio_data = audio_data.float().cpu().numpy()
            
        buf = io.BytesIO()
        sf.write(buf, audio_data, sr, format="WAV")
        buf.seek(0)
        
        return StreamingResponse(buf, media_type="audio/wav")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 Server starting on port 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)