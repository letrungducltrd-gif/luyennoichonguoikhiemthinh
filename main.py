#!/usr/bin/env python3
import os
import socket
import subprocess
import uuid
import json
import re
import zipfile
import shutil
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
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
TMP_DIR = "data/tmp"
FFMPEG_BIN = "ffmpeg"
ALLOWED_EXTS = {".wav", ".mp3", ".m4a", ".ogg"}

# ensure directories exist
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(USERS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(os.path.dirname(VOCAB_STORE_FILE) or ".", exist_ok=True)

# create empty store if missing
if not os.path.exists(VOCAB_STORE_FILE):
    with open(VOCAB_STORE_FILE, "w", encoding="utf-8") as f:
        json.dump({"lessons": {}, "samples": {}}, f, ensure_ascii=False, indent=2)

app = FastAPI(title="Speech Practice API (Complete Server)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# serve static files (audio)
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
    """Return a sanitized filename (no path components / traversal)."""
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
# Built-in lessons & vocab (fallback/demo)
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
    # lessons 4/5/6 initially empty; may be filled from persisted store
}

# persisted store (file-backed)
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
    Merge built-in VOCAB and persisted store for a lesson.
    Persisted items are appended after built-in ones.
    """
    result = []
    if lesson_id in VOCAB:
        for v in VOCAB[lesson_id]:
            fn = v.get("audio_filename")
            v2 = {**v}
            v2["audio_url"] = f"/static/samples/{fn}" if fn else None
            result.append(v2)
    lessons_store = PERSISTED_STORE.get("lessons", {})
    if lesson_id in lessons_store:
        for v in lessons_store[lesson_id]:
            fn = v.get("audio_filename")
            v2 = {**v}
            v2["audio_url"] = f"/static/samples/{fn}" if fn else None
            result.append(v2)
    return result


# -------------------------
# Utility: audio conversion & feature extraction
# -------------------------
def convert_to_wav16_mono(src_path: str, dst_path: str) -> None:
    # requires ffmpeg installed; raises CalledProcessError on failure
    cmd = [FFMPEG_BIN, "-y", "-i", src_path, "-ac", "1", "-ar", "16000", "-vn", dst_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


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


# -------------------------
# Samples rescan / auto-register
# -------------------------
def rescan_samples():
    """
    Scan SAMPLES_DIR for audio files. Register any new files into PERSISTED_STORE['samples'].
    If filename matches pattern '<lessonId>_<rest>...' a vocab item is auto-created in that lesson.
    Returns a report dict.
    """
    load_persisted_store()
    samples_meta = PERSISTED_STORE.setdefault("samples", {})
    lessons_store = PERSISTED_STORE.setdefault("lessons", {})

    existing_filenames = {m["filename"] for m in samples_meta.values()}
    files = sorted([f for f in os.listdir(SAMPLES_DIR) if os.path.isfile(os.path.join(SAMPLES_DIR, f))])
    new_samples = []
    new_vocabs = []

    for fn in files:
        _, ext = os.path.splitext(fn)
        if ext.lower() not in ALLOWED_EXTS:
            continue
        if fn in existing_filenames:
            continue
        # register sample
        sample_id = str(uuid.uuid4())[:8]
        samples_meta[sample_id] = {"filename": fn, "lesson_id": None, "vocab_id": None}
        new_samples.append({"sample_id": sample_id, "filename": fn})

        # try auto-create vocab if filename starts with lesson id like '2_me.wav' or '3_1_pho.wav'
        m = re.match(r'^(\d+)[-_](.+)$', fn)
        if m:
            lesson_id = m.group(1)
            remainder = os.path.splitext(m.group(2))[0]
            word = re.sub(r'[_\-]+', ' ', remainder).strip().capitalize()
            lesson_ids = {l["id"] for l in LESSONS}
            if lesson_id in lesson_ids:
                lessons_store.setdefault(lesson_id, lessons_store.get(lesson_id, []))
                exists = any(v.get("audio_filename") == fn for v in lessons_store[lesson_id])
                if not exists:
                    vid = f"{lesson_id}_{len(lessons_store[lesson_id]) + 1}"
                    vocab_obj = {"id": vid, "word": word or vid, "meaning": "", "example": "", "audio_filename": fn}
                    lessons_store[lesson_id].append(vocab_obj)
                    samples_meta[sample_id]["lesson_id"] = lesson_id
                    samples_meta[sample_id]["vocab_id"] = vid
                    new_vocabs.append({"lesson_id": lesson_id, "vocab_id": vid, "word": word, "audio_filename": fn})

    save_persisted_store()
    report = {
        "total_files_on_disk": len(files),
        "registered_samples_total": len(samples_meta),
        "new_samples_found": len(new_samples),
        "new_samples": new_samples,
        "new_vocabs_added": len(new_vocabs),
        "new_vocabs": new_vocabs,
    }
    return report


# perform initial load + rescan at startup
load_persisted_store()
try:
    startup_report = rescan_samples()
    print("startup rescan report:", startup_report)
except Exception as e:
    print("startup rescan failed:", e)


# -------------------------
# API: lessons & vocab
# -------------------------
@app.get("/lessons")
def api_get_lessons():
    return LESSONS


@app.get("/lessons/{lesson_id}/vocab")
def api_get_vocab(lesson_id: str):
    return merged_vocab_for_lesson(lesson_id)


@app.put("/lessons/{lesson_id}/progress")
def api_update_progress(lesson_id: str, progress: int):
    for l in LESSONS:
        if l["id"] == lesson_id:
            l["progress"] = max(0, min(100, int(progress)))
            return {"ok": True, "lesson": l}
    raise HTTPException(status_code=404, detail="Lesson not found")


# -------------------------
# API: samples listing, upload, rescan
# -------------------------
@app.get("/samples", summary="List sample files available on server (auto-rescan is triggered)")
def api_list_samples():
    report = rescan_samples()
    files = []
    for fn in sorted(os.listdir(SAMPLES_DIR)):
        path = os.path.join(SAMPLES_DIR, fn)
        if os.path.isfile(path):
            files.append({"filename": fn, "audio_url": f"/static/samples/{fn}"})
    return {"count": len(files), "files": files, "rescan_report": report}


@app.post("/samples/upload-simple", summary="Upload sample and register (simple). Returns sample_id and audio_url.")
async def api_upload_simple(file: UploadFile = File(...), lesson_id: Optional[str] = Form(None), vocab_id: Optional[str] = Form(None)):
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

    # convert to wav16 mono
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
        # conversion failed, keep raw file as registered name
        conv_fname = raw_fname
        conv_path = raw_path

    load_persisted_store()
    PERSISTED_STORE.setdefault("samples", {})
    PERSISTED_STORE["samples"][sample_id] = {"filename": conv_fname, "lesson_id": lesson_id, "vocab_id": vocab_id}
    save_persisted_store()
    audio_url = f"/static/samples/{conv_fname}"
    return {"ok": True, "sample_id": sample_id, "audio_filename": conv_fname, "audio_url": audio_url}


@app.post("/samples/rescan", summary="Rescan samples directory and auto-register new files")
def api_samples_rescan():
    report = rescan_samples()
    return {"ok": True, "report": report}


# export / import package for offline distribution
@app.get("/samples/export-package", summary="Export samples + vocab_store.json as a zip for offline use")
def api_export_package():
    package_name = f"speech_package_{uuid.uuid4().hex[:8]}.zip"
    package_path = os.path.join(TMP_DIR, package_name)
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        if os.path.exists(VOCAB_STORE_FILE):
            z.write(VOCAB_STORE_FILE, arcname=os.path.basename(VOCAB_STORE_FILE))
        if os.path.isdir(SAMPLES_DIR):
            for f in os.listdir(SAMPLES_DIR):
                full = os.path.join(SAMPLES_DIR, f)
                if os.path.isfile(full):
                    z.write(full, arcname=os.path.join("samples", f))
    def iterfile():
        with open(package_path, "rb") as fp:
            yield from fp
        try:
            os.remove(package_path)
        except Exception:
            pass
    return StreamingResponse(iterfile(), media_type="application/zip", headers={"Content-Disposition": f"attachment; filename={package_name}"})


@app.post("/samples/import-package", summary="Import zip (samples + vocab_store.json). Overwrites persisted store and copies samples.")
async def api_import_package(file: UploadFile = File(...)):
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty upload")
    temp_zip = os.path.join(TMP_DIR, f"import_{uuid.uuid4().hex[:8]}.zip")
    with open(temp_zip, "wb") as f:
        f.write(contents)
    extract_dir = os.path.join(TMP_DIR, f"extract_{uuid.uuid4().hex[:8]}")
    os.makedirs(extract_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(temp_zip, 'r') as z:
            z.extractall(extract_dir)
        maybe_vocab = os.path.join(extract_dir, os.path.basename(VOCAB_STORE_FILE))
        if os.path.exists(maybe_vocab):
            shutil.copy2(maybe_vocab, VOCAB_STORE_FILE)
        extracted_samples_dir = os.path.join(extract_dir, "samples")
        if os.path.isdir(extracted_samples_dir):
            for f in os.listdir(extracted_samples_dir):
                src = os.path.join(extracted_samples_dir, f)
                dst = os.path.join(SAMPLES_DIR, f)
                shutil.copy2(src, dst)
        try:
            os.remove(temp_zip)
        except Exception:
            pass
        report = rescan_samples()
        return {"ok": True, "report": report}
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid zip file")
    finally:
        try:
            shutil.rmtree(extract_dir)
        except Exception:
            pass


# -------------------------
# Persisted vocab management
# -------------------------
class VocabItemIn(BaseModel):
    id: Optional[str] = None
    word: str
    meaning: Optional[str] = ""
    example: Optional[str] = ""
    audio_filename: Optional[str] = None  # filename existing in data/samples
    sample_id: Optional[str] = None       # sample_id returned by upload


@app.post("/lessons/{lesson_id}/vocab", summary="Add a vocab item to persisted store and link to a sample")
def api_add_vocab(lesson_id: str, item: VocabItemIn):
    load_persisted_store()
    lessons_store = PERSISTED_STORE.setdefault("lessons", {})
    if lesson_id not in lessons_store:
        lessons_store[lesson_id] = []
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
    vocab_with_url = {**vocab_obj, "audio_url": f"/static/samples/{audio_filename}" if audio_filename else None}
    return {"ok": True, "vocab": vocab_with_url}


@app.get("/lessons/{lesson_id}/vocab/store", summary="Get persisted vocab for lesson")
def api_get_vocab_store(lesson_id: str):
    load_persisted_store()
    lessons_store = PERSISTED_STORE.get("lessons", {})
    items = lessons_store.get(lesson_id, [])
    result = []
    for v in items:
        fn = v.get("audio_filename")
        result.append({**v, "audio_url": f"/static/samples/{fn}" if fn else None})
    return {"count": len(result), "vocab": result}


# -------------------------
# Comparison endpoints
# -------------------------
@app.post("/compare_features")
def api_compare_features(sample: dict, user: dict):
    # sample and user are JSON payloads matching FeaturePayload
    f1 = {"mfcc": sample.get("mfcc", []), "pitch": sample.get("pitch", 0.0), "tempo": sample.get("tempo", 0.0)}
    f2 = {"mfcc": user.get("mfcc", []), "pitch": user.get("pitch", 0.0), "tempo": user.get("tempo", 0.0)}
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
    return {"mfcc_distance": mfcc_dist, "pitch_diff": pitch_diff, "tempo_diff": tempo_diff, "feedback": feedback}


@app.post("/compare", summary="Compare user audio with sample (upload sample file or provide sample_id)")
async def api_compare(user: UploadFile = File(...), sample: Optional[UploadFile] = File(None), sample_id: Optional[str] = Form(None)):
    user_bytes = await user.read()
    if len(user_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty user file")
    if len(user_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="User file too large")

    user_name = safe_filename(user.filename)
    user_ext = os.path.splitext(user_name)[1] or ".wav"
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
        sample_ext = os.path.splitext(sample_name)[1] or ".wav"
        sample_temp_path = os.path.join(SAMPLES_DIR, f"s_{uuid.uuid4().hex}{sample_ext}")
        with open(sample_temp_path, "wb") as f:
            f.write(sample_bytes)
        sample_path = sample_temp_path
        sample_uploaded_temp = True
    elif sample_id:
        load_persisted_store()
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
        # cleanup any partially created conv files
        for p in (user_conv, sample_conv):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        detail = "ffmpeg not found on server" if isinstance(e, FileNotFoundError) else "Error converting audio (ffmpeg required)"
        raise HTTPException(status_code=500, detail=detail)

    try:
        f1 = extract_features(sample_conv)
        f2 = extract_features(user_conv)
    except Exception:
        # cleanup before returning
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
        raise HTTPException(status_code=400, detail="Failed to analyze audio. Ensure files are valid speech recordings.")
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
