# main.py
import os
import socket
import subprocess
import uuid
import json
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import librosa
import numpy as np
import uvicorn

# --- Configuration ---
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB
SAMPLES_DIR = "data/samples"
USERS_DIR = "data/users"
STATIC_DIR = "data"
VOCAB_STORE_FILE = "data/vocab_store.json"
FFMPEG_BIN = "ffmpeg"

os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(USERS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(os.path.dirname(VOCAB_STORE_FILE) or ".", exist_ok=True)
if not os.path.exists(VOCAB_STORE_FILE):
    with open(VOCAB_STORE_FILE, "w", encoding="utf-8") as f:
        json.dump({"lessons": {}, "samples": {}}, f, ensure_ascii=False, indent=2)

app = FastAPI(title="Speech Practice API (Server-side comparison)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev: allow all; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (e.g. /static/samples/<file>)
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
    """Sanitize a provided filename to avoid path traversal and odd separators.

    - Keeps only the basename
    - Normalizes backslashes to forward slashes before splitting
    - Removes parent directory sequences
    """
    if not name:
        return ""
    base = os.path.basename(name)
    base = base.replace("\\", "/").split("/")[-1]
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
# Static sample lessons & VOCAB (default built-in)
# -------------------------
LESSONS = [
    {"id": "1", "title": "Chào hỏi cơ bản", "description": "Học cách chào hỏi", "progress": 80},
    {"id": "2", "title": "Gia đình", "description": "Từ vựng về gia đình", "progress": 60},
    {"id": "3", "title": "Thức ăn & Đồ uống", "description": "Từ vựng đồ ăn", "progress": 45},
    {"id": "4", "title": "Số đếm", "description": "Từ vựng về số và cách đếm", "progress": 0},
    {"id": "5", "title": "Màu sắc", "description": "Từ vựng các màu cơ bản", "progress": 0},
    {"id": "6", "title": "Động vật", "description": "Từ vựng về các động vật thường gặp", "progress": 0},
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
    # lessons 4/5/6 initially empty; may be filled from persistent store
}

# In-memory store for persisted vocab and samples (loaded from VOCAB_STORE_FILE)
PERSISTED_STORE = {"lessons": {}, "samples": {}}


def load_persisted_store():
    global PERSISTED_STORE
    try:
        with open(VOCAB_STORE_FILE, "r", encoding="utf-8") as f:
            PERSISTED_STORE = json.load(f)
    except Exception:
        PERSISTED_STORE = {"lessons": {}, "samples": {}}


def save_persisted_store():
    with open(VOCAB_STORE_FILE, "w", encoding="utf-8") as f:
        json.dump(PERSISTED_STORE, f, ensure_ascii=False, indent=2)


def merged_vocab_for_lesson(lesson_id: str):
    """
    Return list of vocab items for a lesson by merging built-in VOCAB and persisted store.
    Persisted items come after built-in ones.
    """
    result = []
    # built-in first
    if lesson_id in VOCAB:
        result.extend(VOCAB[lesson_id])
    # persisted next
    lessons_store = PERSISTED_STORE.get("lessons", {})
    if lesson_id in lessons_store:
        for v in lessons_store[lesson_id]:
            # attach audio_url if audio_filename exists
            fn = v.get("audio_filename")
            v2 = {**v}
            v2["audio_url"] = f"/static/samples/{fn}" if fn else None
            result.append(v2)
    return result


# Load persisted store at startup
load_persisted_store()


@app.get("/lessons")
def get_lessons():
    return LESSONS


@app.get("/lessons/{lesson_id}/vocab")
def get_vocab(lesson_id: str):
    return merged_vocab_for_lesson(lesson_id)


@app.put("/lessons/{lesson_id}/progress")
def update_progress(lesson_id: str, progress: int):
    for l in LESSONS:
        if l["id"] == lesson_id:
            l["progress"] = max(0, min(100, int(progress)))
            return {"ok": True, "lesson": l}
    raise HTTPException(status_code=404, detail="Lesson not found")


# -------------------------
# Sample uploads and vocab management (persisted)
# -------------------------
class VocabItemIn(BaseModel):
    id: Optional[str] = None
    word: str
    meaning: Optional[str] = ""
    example: Optional[str] = ""
    audio_filename: Optional[str] = None  # filename existing in data/samples
    sample_id: Optional[str] = None       # sample_id returned by upload


def convert_to_wav16_mono(src_path: str, dst_path: str) -> None:
    cmd = [FFMPEG_BIN, "-y", "-i", src_path, "-ac", "1", "-ar", "16000", "-vn", dst_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


@app.get("/samples", summary="List sample files available on server")
def list_samples():
    files = []
    for fn in sorted(os.listdir(SAMPLES_DIR)):
        path = os.path.join(SAMPLES_DIR, fn)
        if os.path.isfile(path):
            files.append({"filename": fn, "audio_url": f"/static/samples/{fn}"})
    return {"count": len(files), "files": files}


@app.post("/samples/upload-simple", summary="Upload sample file and register it (simple)")
async def upload_simple(file: UploadFile = File(...), lesson_id: Optional[str] = Form(None), vocab_id: Optional[str] = Form(None)):
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large")
    _, ext = os.path.splitext(file.filename)
    ext = ext or ".wav"
    sample_id = str(uuid.uuid4())[:8]
    raw_fname = f"{sample_id}{ext}"
    raw_path = os.path.join(SAMPLES_DIR, raw_fname)
    with open(raw_path, "wb") as f:
        f.write(contents)

    # convert to wav16 mono for consistency
    conv_fname = f"{sample_id}.wav"
    conv_path = os.path.join(SAMPLES_DIR, conv_fname)
    try:
        convert_to_wav16_mono(raw_path, conv_path)
        if raw_path != conv_path:
            try:
                os.remove(raw_path)
            except Exception:
                pass
    except Exception:
        # if conversion fails, keep raw file and proceed (optional)
        conv_fname = raw_fname
        conv_path = raw_path

    # register in persisted store
    PERSISTED_STORE.setdefault("samples", {})
    PERSISTED_STORE["samples"][sample_id] = {"filename": conv_fname, "lesson_id": lesson_id, "vocab_id": vocab_id}
    save_persisted_store()
    audio_url = f"/static/samples/{conv_fname}"
    return {"ok": True, "sample_id": sample_id, "audio_filename": conv_fname, "audio_url": audio_url}


@app.post("/lessons/{lesson_id}/vocab", summary="Add a vocab item to persisted store and link to a sample")
def add_vocab(lesson_id: str, item: VocabItemIn):
    lessons_store = PERSISTED_STORE.setdefault("lessons", {})
    if lesson_id not in lessons_store:
        lessons_store[lesson_id] = []
    # resolve audio_filename from sample_id if provided
    audio_filename = item.audio_filename
    if item.sample_id:
        sample_meta = PERSISTED_STORE.get("samples", {}).get(item.sample_id)
        if not sample_meta:
            raise HTTPException(status_code=404, detail="sample_id not found")
        audio_filename = sample_meta.get("filename")
    vid = item.id or f"{lesson_id}_{len(lessons_store[lesson_id]) + 1}"
    vocab_obj = {
        "id": vid,
        "word": item.word,
        "meaning": item.meaning or "",
        "example": item.example or "",
        "audio_filename": audio_filename
    }
    lessons_store[lesson_id].append(vocab_obj)
    save_persisted_store()
    # return the merged view (including audio_url)
    vocab_with_url = {**vocab_obj, "audio_url": f"/static/samples/{audio_filename}" if audio_filename else None}
    return {"ok": True, "vocab": vocab_with_url}


@app.get("/lessons/{lesson_id}/vocab/store", summary="Get persisted vocab for lesson")
def get_vocab_store(lesson_id: str):
    lessons_store = PERSISTED_STORE.get("lessons", {})
    items = lessons_store.get(lesson_id, [])
    result = []
    for v in items:
        fn = v.get("audio_filename")
        result.append({**v, "audio_url": f"/static/samples/{fn}" if fn else None})
    return {"count": len(result), "vocab": result}


# -------------------------
# Feature extraction & server-side compare (existing logic)
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


@app.post("/compare")
async def compare_endpoint(
    user: UploadFile = File(...),
    sample: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
):
    user_bytes = await user.read()
    if len(user_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty user file")
    if len(user_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="User file too large")

    user_name = safe_filename(user.filename)
    user_ext = os.path.splitext(user_name)[1]
    user_temp_path = os.path.join(USERS_DIR, f"u_{uuid.uuid4().hex}{user_ext}")
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
        sample_temp_path = os.path.join(SAMPLES_DIR, f"s_{uuid.uuid4().hex}{sample_ext}")
        with open(sample_temp_path, "wb") as f:
            f.write(sample_bytes)
        sample_path = sample_temp_path
        sample_uploaded_temp = True
    elif sample_id:
        sample_meta = PERSISTED_STORE.get("samples", {}).get(sample_id)
        if not sample_meta:
            try:
                os.remove(user_temp_path)
            except Exception:
                pass
            raise HTTPException(status_code=404, detail="sample_id not found")
        fn = sample_meta.get("filename")
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
        # try to remove any partial conv files
        for p in (user_conv, sample_conv):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=detail)

    f1 = extract_features(sample_conv)
    f2 = extract_features(user_conv)
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