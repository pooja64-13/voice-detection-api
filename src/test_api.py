import base64
import requests

# Read audio and encode
with open("test_audio/sample.mp3", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode()

url = "http://127.0.0.1:8000/detect-voice"

headers = {
    "Authorization": "Bearer ai_impact_voice_2026_key",
    "Content-Type": "application/json"
}

payload = {
    "audio_base64": audio_base64,
    "language": "en"
}

response = requests.post(url, json=payload, headers=headers)

print("Status code:", response.status_code)
print("Response:", response.text)
