import os
import uuid
import json
import re
import subprocess
from typing import Optional
import librosa
import numpy as np

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

# Built-in lessons & vocab (fallback/demo)
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

def convert_to_wav16_mono(src_path: str, dst_path: str) -> None:
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

def rescan_samples():
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
		sample_id = str(uuid.uuid4())[:8]
		samples_meta[sample_id] = {"filename": fn, "lesson_id": None, "vocab_id": None}
		new_samples.append({"sample_id": sample_id, "filename": fn})

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

# Multimodal helpers are provided in a separate module helpers_multimodal.py.
# Attempt relative import first, fall back to top-level import, else set to None.
try:
    from . import helpers_multimodal as helpers_multimodal
except Exception:
    try:
        import helpers_multimodal as helpers_multimodal
    except Exception:
        helpers_multimodal = None
