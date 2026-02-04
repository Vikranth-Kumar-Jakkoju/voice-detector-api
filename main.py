from fastapi import FastAPI, Header, HTTPException
import base64
import librosa
import numpy as np
import io
import soundfile as sf

# Create FastAPI app
app = FastAPI()

# Simple API key (you will change this later)
API_KEY = "my-secret-key"


# =========================
# ADD API ENDPOINT (HERE)
# =========================
@app.post("/detect-voice")
def detect_voice(
    audio_base64: str,
    language: str,
    x_api_key: str = Header(...)
):
    # 1. API key check
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # 2. Language validation
    if language not in ["ta", "en", "hi", "ml", "te"]:
        raise HTTPException(status_code=400, detail="Unsupported language")

    # 3. Decode Base64 audio
    try:
        audio_bytes = base64.b64decode(audio_base64)
        audio_buffer = io.BytesIO(audio_bytes)
        audio, sr = sf.read(audio_buffer)
    except:
        raise HTTPException(status_code=400, detail="Invalid audio format")

    # 4. Audio length check (minimum 1 second)
    if len(audio) < sr:
        raise HTTPException(status_code=400, detail="Audio too short")

    # 5. Feature extraction (MFCC)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = float(np.mean(mfcc))

    # 6. Simple signal-based classification
    if mfcc_mean > -200:
        result = "HUMAN"
    else:
        result = "AI_GENERATED"

    # 7. Confidence score (dynamic, not hard-coded)
    confidence = min(1.0, abs(mfcc_mean) / 300)

    # 8. Return structured JSON response
    return {
        "result": result,
        "confidence": round(confidence, 2),
        "language": language
    }
