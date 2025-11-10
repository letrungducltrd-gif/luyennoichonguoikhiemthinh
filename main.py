# main.py
import os
import socket
import subprocess
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import librosa
import numpy as np
import uvicorn

# --- Cấu hình ---
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB
SAMPLES_DIR = "data/samples"
USERS_DIR = "data/users"
STATIC_DIR = "data"
FFMPEG_BIN = "ffmpeg"

os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(USERS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

app = FastAPI(title="Speech Practice API (Server-side comparison)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev: cho phép mọi origin; production: giới hạn lại
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (ví dụ: /static/samples/<file>)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


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


def safe_filename(name: Optional[str]) -> str:
    """Return a safe filename (no path components, no traversal).

    - Strips any directory components (both Unix and Windows styles)
    - Removes parent directory sequences
    """
    if not name:
        return ""
    base = os.path.basename(name)
    # Normalize potential Windows backslashes and re-strip
    base = base.replace("\\", "/").split("/")[-1]
    # prevent simple traversal artifacts
    return base.replace("..", "")


@app.get("/")
def read_root():
    return {"message": "Server is running"}


@app.get("/ping")
def ping():
    return {"pong": True}


@app.get("/whoami")
def whoami():
    return {"ip": get_local_ip(), "port": 8000}


# -------------------------
# Sample data (ví dụ) - chuyển sang DB/JSON khi cần
# -------------------------
LESSONS = [
    {"id": "1", "title": "Chào hỏi cơ bản", "description": "Học cách chào hỏi", "progress": 80},
    {"id": "2", "title": "Gia đình", "description": "Từ vựng về gia đình", "progress": 60},
    {"id": "3", "title": "Thức ăn & Đồ uống", "description": "Từ vựng đồ ăn", "progress": 45},
]

VOCAB = {
    "1": [
        {"id": "1_1", "word": "Xin chào", "meaning": "Hello", "example": "Xin chào bạn!", "audio_filename": "xin_chao.wav"},
        {"id": "1_2", "word": "Tạm biệt", "meaning": "Goodbye", "example": "Tạm biệt nhé!", "audio_filename": "tam_biet.wav"},
    ],
    "2": [
        {"id": "2_1", "word": "Mẹ", "meaning": "Mother", "example": "Mẹ của tôi...", "audio_filename": "me.wav"},
        {"id": "2_2", "word": "Cha", "meaning": "Father", "example": "Cha làm nghề...", "audio_filename": "cha.wav"},
    ],
    "3": [
        {"id": "3_1", "word": "Phở", "meaning": "Pho (noodle soup)", "example": "Tôi thích phở.", "audio_filename": "pho.wav"},
    ],
}


def vocab_item_with_url(item):
    fn = item.get("audio_filename")
    audio_url = f"/static/samples/{fn}" if fn else None
    return {**item, "audio_url": audio_url}


@app.get("/lessons")
def get_lessons():
    return LESSONS


@app.get("/lessons/{lesson_id}/vocab")
def get_vocab(lesson_id: str):
    if lesson_id not in VOCAB:
        raise HTTPException(status_code=404, detail="Lesson not found")
    return [vocab_item_with_url(x) for x in VOCAB[lesson_id]]


@app.put("/lessons/{lesson_id}/progress")
def update_progress(lesson_id: str, progress: int):
    for l in LESSONS:
        if l["id"] == lesson_id:
            l["progress"] = max(0, min(100, int(progress)))
            return {"ok": True, "lesson": l}
    raise HTTPException(status_code=404, detail="Lesson not found")


# -------------------------
# Utility: convert to WAV16 mono via ffmpeg
# -------------------------
def convert_to_wav16_mono(src_path: str, dst_path: str) -> None:
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-i", src_path,
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        dst_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# -------------------------
# Feature extraction & compare
# -------------------------
def extract_features(path: str):
    y, sr = librosa.load(path, sr=16000)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    pitch, _ = librosa.piptrack(y=y, sr=sr)
    positive_pitch = pitch[pitch > 0]
    pitch_mean = float(np.mean(positive_pitch)) if positive_pitch.size > 0 else 0.0
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return {"mfcc": mfcc.tolist(), "pitch": pitch_mean, "tempo": float(tempo)}


def compare_features_dicts(f1: dict, f2: dict):
    a = np.array(f1["mfcc"])
    b = np.array(f2["mfcc"])
    mfcc_dist = float(np.linalg.norm(a - b))
    pitch_diff = float(abs(f1["pitch"] - f2["pitch"]))
    tempo_diff = float(abs(f1["tempo"] - f2["tempo"]))
    return mfcc_dist, pitch_diff, tempo_diff


class FeaturePayload(BaseModel):
    mfcc: list
    pitch: float
    tempo: float


@app.post("/compare_features")
def compare_features_endpoint(sample: FeaturePayload, user: FeaturePayload):
    mfcc_dist, pitch_diff, tempo_diff = compare_features_dicts(sample.dict(), user.dict())
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
    return {"mfcc_distance": mfcc_dist, "pitch_diff": pitch_diff, "tempo_diff": tempo_diff, "feedback": feedback}


# -------------------------
# POST /compare (server-side comparison)
# -------------------------
@app.post("/compare")
async def compare_endpoint(
    user: UploadFile = File(...),
    sample: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
):
    # validate user upload
    user_bytes = await user.read()
    if len(user_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty user file")
    if len(user_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="User file too large")

    user_name = safe_filename(user.filename)
    user_ext = os.path.splitext(user_name)[1]
    user_temp_path = os.path.join(USERS_DIR, f"u_{uuid4().hex}{user_ext}")
    with open(user_temp_path, "wb") as f:
        f.write(user_bytes)

    sample_path = None
    sample_uploaded_temp = False
    if sample is not None:
        sample_bytes = await sample.read()
        if len(sample_bytes) > MAX_UPLOAD_BYTES:
            try:
                os.remove(user_temp_path)
            except Exception:
                pass
            raise HTTPException(status_code=413, detail="Sample file too large")
        sample_name = safe_filename(sample.filename)
        sample_ext = os.path.splitext(sample_name)[1]
        sample_temp_path = os.path.join(SAMPLES_DIR, f"s_{uuid4().hex}{sample_ext}")
        with open(sample_temp_path, "wb") as f:
            f.write(sample_bytes)
        sample_path = sample_temp_path
        sample_uploaded_temp = True
    elif sample_id:
        found = None
        for lesson_vocab in VOCAB.values():
            for item in lesson_vocab:
                if item["id"] == sample_id:
                    found = item
                    break
            if found:
                break
        if not found:
            try:
                os.remove(user_temp_path)
            except Exception:
                pass
            raise HTTPException(status_code=404, detail="sample_id not found")
        fn = found.get("audio_filename")
        if not fn:
            try:
                os.remove(user_temp_path)
            except Exception:
                pass
            raise HTTPException(status_code=404, detail="No audio file available for this sample_id")
        sample_path = os.path.join(SAMPLES_DIR, fn)
        if not os.path.exists(sample_path):
            try:
                os.remove(user_temp_path)
            except Exception:
                pass
            raise HTTPException(status_code=404, detail=f"Sample file missing on server: {fn}")
    else:
        try:
            os.remove(user_temp_path)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail="Either sample file or sample_id must be provided")

    # convert to wav16 mono temp files
    sample_conv = sample_path + ".conv.wav"
    user_conv = user_temp_path + ".conv.wav"
    try:
        convert_to_wav16_mono(sample_path, sample_conv)
        convert_to_wav16_mono(user_temp_path, user_conv)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        try:
            os.remove(user_temp_path)
        except Exception:
            pass
        if sample_uploaded_temp:
            try:
                os.remove(sample_path)
            except Exception:
                pass
        detail = "ffmpeg not found on server" if isinstance(e, FileNotFoundError) else "Error converting audio (ffmpeg required)"
        # attempt to remove any partially created conv files
        for p in (user_conv, sample_conv):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=detail)

    # extract features & compare
    try:
        f1 = extract_features(sample_conv)
        f2 = extract_features(user_conv)
    except Exception:
        # cleanup before erroring
        for p in (user_temp_path, user_conv, sample_conv):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        if sample_uploaded_temp:
            try:
                if os.path.exists(sample_path):
                    os.remove(sample_path)
            except Exception:
                pass
        raise HTTPException(status_code=400, detail="Failed to analyze audio. Ensure the files are valid speech recordings.")
    mfcc_dist, pitch_diff, tempo_diff = compare_features_dicts(f1, f2)

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

    # cleanup temp files
    for p in (user_temp_path, user_conv, sample_conv):
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass
    if sample_uploaded_temp:
        try:
            if os.path.exists(sample_path):
                os.remove(sample_path)
        except Exception:
            pass

    return {
        "mfcc_distance": mfcc_dist,
        "pitch_diff": pitch_diff,
        "tempo_diff": tempo_diff,
        "feedback": feedback,
        "features_sample": f1,
        "features_user": f2,
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
