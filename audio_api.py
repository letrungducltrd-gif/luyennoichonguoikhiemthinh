from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from starlette.responses import StreamingResponse
from typing import Optional
import os
import uuid
import zipfile
import shutil

from . import helpers
from pydantic import BaseModel, validator

router = APIRouter()

class VocabItemIn(BaseModel):
	id: Optional[str] = None
	word: str
	meaning: Optional[str] = ""
	example: Optional[str] = ""
	audio_filename: Optional[str] = None
	sample_id: Optional[str] = None
	
	@validator('word')
	def word_not_empty(cls, v):
		if not v or not v.strip():
			raise ValueError('Word cannot be empty')
		return v.strip()

# -------------------------
# API: lessons & vocab
# -------------------------
@router.get("/lessons")
def api_get_lessons():
	return helpers.LESSONS

@router.get("/lessons/{lesson_id}/vocab")
def api_get_vocab(lesson_id: str):
	return helpers.merged_vocab_for_lesson(lesson_id)

@router.put("/lessons/{lesson_id}/progress")
def api_update_progress(lesson_id: str, progress: int):
	# Persist progress to JSON instead of memory
	helpers.load_persisted_store()
	helpers.PERSISTED_STORE.setdefault("progress", {})
	helpers.PERSISTED_STORE["progress"][lesson_id] = max(0, min(100, int(progress)))
	helpers.save_persisted_store()
	
	# Also update in-memory for compatibility
	for l in helpers.LESSONS:
		if l["id"] == lesson_id:
			l["progress"] = helpers.PERSISTED_STORE["progress"][lesson_id]
			return {"ok": True, "lesson": l}
	raise HTTPException(status_code=404, detail="Lesson not found")

# -------------------------
# API: samples listing, upload, rescan
# -------------------------
@router.get("/samples", summary="List sample files available on server (auto-rescan is triggered)")
def api_list_samples():
	# ensure samples dir exists
	os.makedirs(helpers.SAMPLES_DIR, exist_ok=True)
	report = helpers.rescan_samples()
	files = []
	for fn in sorted(os.listdir(helpers.SAMPLES_DIR)):
		path = os.path.join(helpers.SAMPLES_DIR, fn)
		if os.path.isfile(path):
			files.append({"filename": fn, "audio_url": f"/static/samples/{fn}"})
	return {"count": len(files), "files": files, "rescan_report": report}

@router.post("/samples/upload-simple", summary="Upload sample and register (simple). Returns sample_id and audio_url.")
async def api_upload_simple(file: UploadFile = File(...), lesson_id: Optional[str] = Form(None), vocab_id: Optional[str] = Form(None)):
	contents = await file.read()
	if len(contents) == 0:
		raise HTTPException(status_code=400, detail="Empty file")
	if len(contents) > helpers.MAX_UPLOAD_BYTES:
		raise HTTPException(status_code=413, detail="File too large")
	_, ext = os.path.splitext(file.filename)
	ext = ext or ".wav"
	sample_id = str(uuid.uuid4())[:8]
	raw_fname = f"{sample_id}{ext}"
	# ensure samples dir exists before writing
	os.makedirs(helpers.SAMPLES_DIR, exist_ok=True)
	raw_path = os.path.join(helpers.SAMPLES_DIR, raw_fname)
	with open(raw_path, "wb") as f:
		f.write(contents)

	conv_fname = f"{sample_id}.wav"
	conv_path = os.path.join(helpers.SAMPLES_DIR, conv_fname)
	try:
		helpers.convert_to_wav16_mono(raw_path, conv_path)
		if raw_path != conv_path:
			try:
				os.remove(raw_path)
			except Exception:
				pass
	except Exception:
		conv_fname = raw_fname
		conv_path = raw_path

	helpers.load_persisted_store()
	helpers.PERSISTED_STORE.setdefault("samples", {})
	helpers.PERSISTED_STORE["samples"][sample_id] = {"filename": conv_fname, "lesson_id": lesson_id, "vocab_id": vocab_id}
	helpers.save_persisted_store()
	audio_url = f"/static/samples/{conv_fname}"
	return {"ok": True, "sample_id": sample_id, "audio_filename": conv_fname, "audio_url": audio_url}

@router.post("/samples/rescan", summary="Rescan samples directory and auto-register new files")
def api_samples_rescan():
	report = helpers.rescan_samples()
	return {"ok": True, "report": report}

@router.get("/samples/export-package", summary="Export samples + vocab_store.json as a zip for offline use")
def api_export_package():
	package_name = f"speech_package_{uuid.uuid4().hex[:8]}.zip"
	# ensure tmp dir exists
	os.makedirs(helpers.TMP_DIR, exist_ok=True)
	package_path = os.path.join(helpers.TMP_DIR, package_name)
	with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
		if os.path.exists(helpers.VOCAB_STORE_FILE):
			z.write(helpers.VOCAB_STORE_FILE, arcname=os.path.basename(helpers.VOCAB_STORE_FILE))
		if os.path.isdir(helpers.SAMPLES_DIR):
			for f in os.listdir(helpers.SAMPLES_DIR):
				full = os.path.join(helpers.SAMPLES_DIR, f)
				if os.path.isfile(full):
					z.write(full, arcname=os.path.join("samples", f))
	def iterfile():
		# stream in fixed-size chunks to avoid line-based iteration issues
		try:
			with open(package_path, "rb") as fp:
				while True:
					chunk = fp.read(8192)
					if not chunk:
						break
					yield chunk
		finally:
			try:
				os.remove(package_path)
			except Exception:
				pass
	# use proper Content-Disposition header
	headers = {"Content-Disposition": f'attachment; filename="{package_name}"'}
	return StreamingResponse(iterfile(), media_type="application/zip", headers=headers)

@router.post("/samples/import-package", summary="Import zip (samples + vocab_store.json). Overwrites persisted store and copies samples.")
async def api_import_package(file: UploadFile = File(...)):
	contents = await file.read()
	if not contents:
		raise HTTPException(status_code=400, detail="Empty upload")
	# ensure tmp dir exists
	os.makedirs(helpers.TMP_DIR, exist_ok=True)
	temp_zip = os.path.join(helpers.TMP_DIR, f"import_{uuid.uuid4().hex[:8]}.zip")
	with open(temp_zip, "wb") as f:
		f.write(contents)
	extract_dir = os.path.join(helpers.TMP_DIR, f"extract_{uuid.uuid4().hex[:8]}")
	os.makedirs(extract_dir, exist_ok=True)
	try:
		with zipfile.ZipFile(temp_zip, 'r') as z:
			z.extractall(extract_dir)
		maybe_vocab = os.path.join(extract_dir, os.path.basename(helpers.VOCAB_STORE_FILE))
		if os.path.exists(maybe_vocab):
			shutil.copy2(maybe_vocab, helpers.VOCAB_STORE_FILE)
		extracted_samples_dir = os.path.join(extract_dir, "samples")
		if os.path.isdir(extracted_samples_dir):
			# ensure samples dir exists before copying
			os.makedirs(helpers.SAMPLES_DIR, exist_ok=True)
			for f in os.listdir(extracted_samples_dir):
				src = os.path.join(extracted_samples_dir, f)
				dst = os.path.join(helpers.SAMPLES_DIR, f)
				shutil.copy2(src, dst)
		try:
			os.remove(temp_zip)
		except Exception:
			pass
		report = helpers.rescan_samples()
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
@router.post("/lessons/{lesson_id}/vocab", summary="Add a vocab item to persisted store and link to a sample")
def api_add_vocab(lesson_id: str, item: VocabItemIn):
	helpers.load_persisted_store()
	lessons_store = helpers.PERSISTED_STORE.setdefault("lessons", {})
	if lesson_id not in lessons_store:
		lessons_store[lesson_id] = []
	audio_filename = item.audio_filename
	if item.sample_id:
		sample_meta = helpers.PERSISTED_STORE.get("samples", {}).get(item.sample_id)
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
	helpers.save_persisted_store()
	vocab_with_url = {**vocab_obj, "audio_url": f"/static/samples/{audio_filename}" if audio_filename else None}
	return {"ok": True, "vocab": vocab_with_url}

@router.get("/lessons/{lesson_id}/vocab/store", summary="Get persisted vocab for lesson")
def api_get_vocab_store(lesson_id: str):
	helpers.load_persisted_store()
	lessons_store = helpers.PERSISTED_STORE.get("lessons", {})
	items = lessons_store.get(lesson_id, [])
	result = []
	for v in items:
		fn = v.get("audio_filename")
		result.append({**v, "audio_url": f"/static/samples/{fn}" if fn else None})
	return {"count": len(result), "vocab": result}

# -------------------------
# API: Pronunciation Analysis
# -------------------------
@router.post("/analyze/compare", summary="Compare user audio with reference sample")
async def api_analyze_compare(
	user_audio: UploadFile = File(..., description="User's pronunciation audio"),
	sample_id: str = Form(..., description="Reference sample ID to compare against")
):
	"""
	Upload user audio and compare with a reference sample.
	Returns similarity scores for pronunciation accuracy.
	"""
	# Validate user audio
	user_contents = await user_audio.read()
	if len(user_contents) == 0:
		raise HTTPException(status_code=400, detail="Empty audio file")
	if len(user_contents) > helpers.MAX_UPLOAD_BYTES:
		raise HTTPException(status_code=413, detail="File too large")
	
	# Get reference sample info
	helpers.load_persisted_store()
	sample_meta = helpers.PERSISTED_STORE.get("samples", {}).get(sample_id)
	if not sample_meta:
		raise HTTPException(status_code=404, detail=f"Reference sample '{sample_id}' not found")
	
	ref_filename = sample_meta.get("filename")
	ref_path = os.path.join(helpers.SAMPLES_DIR, ref_filename)
	if not os.path.exists(ref_path):
		raise HTTPException(status_code=404, detail=f"Reference audio file not found: {ref_filename}")
	
	# Save user audio temporarily
	os.makedirs(helpers.TMP_DIR, exist_ok=True)
	user_id = str(uuid.uuid4())[:8]
	_, ext = os.path.splitext(user_audio.filename)
	user_raw_path = os.path.join(helpers.TMP_DIR, f"user_{user_id}{ext or '.wav'}")
	with open(user_raw_path, "wb") as f:
		f.write(user_contents)
	
	# Convert user audio to WAV 16kHz mono
	user_wav_path = os.path.join(helpers.TMP_DIR, f"user_{user_id}.wav")
	try:
		helpers.convert_to_wav16_mono(user_raw_path, user_wav_path)
		if user_raw_path != user_wav_path:
			try:
				os.remove(user_raw_path)
			except Exception:
				pass
	except Exception as e:
		helpers.logger.error(f"Audio conversion failed: {e}")
		try:
			os.remove(user_raw_path)
		except Exception:
			pass
		raise HTTPException(status_code=500, detail=f"Audio conversion failed: {str(e)}")
	
	# Extract features from both audios
	try:
		user_features = helpers.extract_features(user_wav_path)
		ref_features = helpers.extract_features(ref_path)
	except Exception as e:
		helpers.logger.error(f"Feature extraction failed: {e}")
		try:
			os.remove(user_wav_path)
		except Exception:
			pass
		raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")
	
	# Compare features
	try:
		mfcc_dist, pitch_diff, tempo_diff = helpers.compare_features_dicts(user_features, ref_features)
	except Exception as e:
		helpers.logger.error(f"Comparison failed: {e}")
		try:
			os.remove(user_wav_path)
		except Exception:
			pass
		raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")
	
	# Calculate similarity score (0-100)
	# Lower distance = higher similarity
	mfcc_score = max(0, 100 - mfcc_dist * 2)  # Adjust multiplier as needed
	pitch_score = max(0, 100 - pitch_diff * 0.5)
	tempo_score = max(0, 100 - tempo_diff * 0.5)
	overall_score = (mfcc_score * 0.6 + pitch_score * 0.3 + tempo_score * 0.1)
	
	# Cleanup user audio
	try:
		os.remove(user_wav_path)
	except Exception:
		pass
	
	return {
		"ok": True,
		"sample_id": sample_id,
		"reference_file": ref_filename,
		"scores": {
			"overall": round(overall_score, 2),
			"mfcc": round(mfcc_score, 2),
			"pitch": round(pitch_score, 2),
			"tempo": round(tempo_score, 2)
		},
		"details": {
			"mfcc_distance": round(mfcc_dist, 4),
			"pitch_difference_hz": round(pitch_diff, 2),
			"tempo_difference_bpm": round(tempo_diff, 2)
		},
		"feedback": _get_feedback(overall_score)
	}

def _get_feedback(score: float) -> str:
	"""Generate feedback message based on score"""
	if score >= 90:
		return "Xuất sắc! Phát âm rất chuẩn."
	elif score >= 75:
		return "Tốt! Phát âm khá chính xác."
	elif score >= 60:
		return "Khá! Cần luyện tập thêm một chút."
	elif score >= 40:
		return "Cần cải thiện. Hãy nghe kỹ mẫu và thử lại."
	else:
		return "Cần luyện tập nhiều hơn. Nghe lại mẫu và phát âm chậm rãi."
