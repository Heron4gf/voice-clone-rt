import os, sys, requests, torch, safetensors.torch, io, soundfile as sf
from qwen_tts import Qwen3TTSModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

VOICE_URL = os.environ.get("VOICE_URL") # URL from Docker env var, e.g. "https://huggingface.co/your-username/your-repo/resolve/main/voice.safetensors"
VOICE_FILE = "voice.safetensors"

if not os.path.exists(VOICE_FILE):
    if VOICE_URL:
        print(f"⬇️ Downloading voice from {VOICE_URL}...")
        try:
            with requests.get(VOICE_URL, stream=True) as r:
                r.raise_for_status()
                with open(VOICE_FILE, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
            print("✅ Download done.")
        except Exception as e:
            print(f"❌ Download failed: {e}"); sys.exit(1)
    else:
        print("❌ No voice file and no VOICE_URL env var set."); sys.exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("⏳ Loading Model...")
model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", dtype=torch.float16, device_map=device)

print("⏳ Loading Voice...")
voice_data = safetensors.torch.load_file(VOICE_FILE)
prompt = {}
for k, v in voice_data.items():
    if isinstance(v, torch.Tensor):
        v = v.to(device)
        if v.ndim == 1: v = v.unsqueeze(0)
        elif v.ndim == 2 and "id" not in k and "mask" not in k: v = v.unsqueeze(0)
    prompt[k] = v
prompt["x_vector_only_mode"] = [False]
prompt["icl_mode"] = [False]

app = FastAPI()
class Req(BaseModel): text: str

@app.post("/tts")
def tts(r: Req):
    try:
        wavs, sr = model.generate_voice_clone(voice_clone_prompt=prompt, text=r.text)
        audio = wavs[0].float().cpu().numpy() if isinstance(wavs[0], torch.Tensor) else wavs[0]
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV")
        buf.seek(0)
        return StreamingResponse(buf, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)