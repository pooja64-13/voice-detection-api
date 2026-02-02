from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
import base64
import tempfile
import os
import joblib

from src.features import extract_basic_features

# ----------------------
# Configuration
# ----------------------

API_KEY = os.getenv("VOICE_API_KEY")
MODEL_PATH = "voice_detector.pkl"

if not API_KEY:
    raise RuntimeError("VOICE_API_KEY environment variable not set")

# ----------------------
# App init
# ----------------------

app = FastAPI(title="AI Generated Voice Detection API")

# ----------------------
# Load model once
# ----------------------

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("Model file not found. Train the model first.")

model = joblib.load(MODEL_PATH)

# ----------------------
# Request / Response schema
# ----------------------

class VoiceRequest(BaseModel):
    # Accept BOTH audio_base64 and audioBase64
    audio_base64: str = Field(..., alias="audioBase64")
    language: str

    class Config:
        populate_by_name = True


class VoiceResponse(BaseModel):
    classification: str
    confidence: float
    explanation: dict

# ----------------------
# Helper function
# ----------------------

def decode_audio(audio_base64: str) -> str:
    try:
        audio_bytes = base64.b64decode(audio_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp.write(audio_bytes)
    tmp.close()
    return tmp.name

# ----------------------
# API endpoint
# ----------------------

@app.post("/detect-voice", response_model=VoiceResponse)
def detect_voice(
    request: VoiceRequest,
    authorization: str = Header(None, alias="Authorization"),
    x_api_key: str = Header(None, alias="x-api-key")
):
    # Accept BOTH Authorization and x-api-key
    key = authorization or x_api_key

    if key not in (API_KEY, f"Bearer {API_KEY}"):
        raise HTTPException(status_code=401, detail="Unauthorized")

    audio_path = decode_audio(request.audio_base64)

    try:
        features = extract_basic_features(audio_path)
        probs = model.predict_proba([features])[0]
    except Exception:
        raise HTTPException(status_code=400, detail="Audio processing failed")
    finally:
        os.remove(audio_path)

    ai_prob = float(probs[1])
    classification = "AI_GENERATED" if ai_prob >= 0.5 else "HUMAN"

    explanation = {
        "reason": "Decision based on acoustic smoothness and temporal stability",
        "features_used": [
            "MFCC statistics",
            "Zero Crossing Rate",
            "RMS Energy"
        ]
    }

    return {
        "classification": classification,
        "confidence": round(ai_prob, 3),
        "explanation": explanation
    }
