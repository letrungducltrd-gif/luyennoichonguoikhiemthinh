from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import librosa, numpy as np, os
import socket
import uvicorn
from typing import Optional

# --- Tạo thư mục dữ liệu ---
os.makedirs("data/samples", exist_ok=True)
os.makedirs("data/users", exist_ok=True)

# --- FastAPI app ---
app = FastAPI()

# --- CORS: cho phép mọi client gọi API trong LAN ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Hàm lấy IP cục bộ an toàn ---
def get_local_ip() -> Optional[str]:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return None

# --- Endpoints cơ bản ---
@app.get("/")
def read_root():
    return {"message": "Server is running"}

@app.get("/ping")
def ping():
    return {"pong": True}

@app.get("/whoami")
def whoami():
    ip = get_local_ip()
    return {"ip": ip, "port": 8000}

# ==== Hàm trích xuất đặc trưng âm thanh ====
def extract_features(path: str):
    y, sr = librosa.load(path, sr=16000)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    pitch, _ = librosa.piptrack(y=y, sr=sr)
    positive_pitch = pitch[pitch > 0]
    pitch_mean = float(np.mean(positive_pitch)) if positive_pitch.size > 0 else 0.0
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return {"mfcc": mfcc, "pitch": pitch_mean, "tempo": float(tempo)}

# ==== So sánh âm thanh ====
def compare_features(f1, f2):
    mfcc_dist = float(np.linalg.norm(f1["mfcc"] - f2["mfcc"]))
    pitch_diff = float(abs(f1["pitch"] - f2["pitch"]))
    tempo_diff = float(abs(f1["tempo"] - f2["tempo"]))
    return mfcc_dist, pitch_diff, tempo_diff

@app.post("/compare")
async def compare(sample: UploadFile = File(...), user: UploadFile = File(...)):
    sample_path = os.path.join("data", "samples", sample.filename)
    user_path = os.path.join("data", "users", user.filename)

    with open(sample_path, "wb") as f:
        f.write(await sample.read())
    with open(user_path, "wb") as f:
        f.write(await user.read())

    f1 = extract_features(sample_path)
    f2 = extract_features(user_path)

    mfcc_dist, pitch_diff, tempo_diff = compare_features(f1, f2)

    if mfcc_dist < 40:
        feedback = "Rất giống! Bạn phát âm tốt."
    elif mfcc_dist < 80:
        feedback = "Khá giống, cần điều chỉnh một chút."
    else:
        feedback = "Cần luyện thêm — âm khác nhiều so với mẫu."

    if pitch_diff > 30:
        feedback += "\nCao độ khác nhiều, hãy nói cao/trầm hơn."
    if tempo_diff > 20:
        feedback += "\nTốc độ nói khác, thử chậm hoặc nhanh hơn chút."

    return {
        "mfcc_distance": mfcc_dist,
        "pitch_diff": pitch_diff,
        "tempo_diff": tempo_diff,
        "feedback": feedback
    }

# --- Chạy trực tiếp ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)