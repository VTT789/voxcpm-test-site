import io
import os
import threading
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

app = FastAPI(title="VoxCPM2 TTS Tester")
app.mount("/static", StaticFiles(directory="static"), name="static")

_model = None
_model_lock = threading.Lock()

class GenerateRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.get("/api/health")
def health():
    return {
        "ok": True,
        "model_loaded": _model is not None,
        "model_id": "openbmb/VoxCPM2",
        "note": "The model loads on the first generation request."
    }

def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                try:
                    from voxcpm import VoxCPM
                    _model = VoxCPM.from_pretrained(
                        "openbmb/VoxCPM2",
                        load_denoiser=False
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "Could not load VoxCPM2. This may be due to insufficient "
                        "RAM/CPU, missing compatible dependencies, or a model download "
                        f"problem. Details: {exc}"
                    ) from exc
    return _model

@app.post("/api/generate")
def generate(req: GenerateRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Please enter some text.")
    try:
        model = get_model()
        wav = model.generate(
            text=text,
            cfg_value=2.0,
            inference_timesteps=10
        )
        audio = np.asarray(wav).squeeze()
        buffer = io.BytesIO()
        sf.write(buffer, audio, model.tts_model.sample_rate, format="WAV")
        buffer.seek(0)
        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": 'inline; filename="voxcpm2-test.wav"'}
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}")
