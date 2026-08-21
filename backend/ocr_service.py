"""
Production-grade OCR microservice for GateSense ANPR.

This service remains isolated to the OCR stack only. It does not touch vehicle
or plate detection, and it preserves the existing API contract for the rest of
the ANPR pipeline while upgrading reliability for moving Indian vehicles.
"""
from __future__ import annotations

import base64
import math
import os
import re
import shutil
import sys
import tempfile
import time
import asyncio
import datetime
import json
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ai.video_processor import VideoAnprProcessor, STATIC_DIR
from ai.cctv_manager import CCTVStreamManager
from ai.tensorrt_engine import get_system_tensorrt_diagnostics, TensorRTInferenceEngine

os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_use_onednn", "0")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

video_processor = VideoAnprProcessor()
cctv_manager = CCTVStreamManager(video_processor=video_processor)

# Primary OCR: PaddleOCR PP-OCRv4 (with EasyOCR fallback if needed)
OCR_MODEL = None
ACTIVE_ENGINE = "PaddleOCR"
OCR_BACKEND = "paddleocr"
OCR_INIT_ERROR = None

try:
    import torch
    use_gpu = torch.cuda.is_available()
    from paddleocr import PaddleOCR
    OCR_MODEL = PaddleOCR(lang="en", use_textline_orientation=True, show_log=False)
    print(f"[PaddleOCR] Loaded PP-OCRv4 successfully as primary engine (GPU={use_gpu})")
except Exception as paddle_exc:
    print(f"[PaddleOCR] PaddleOCR init notice: {paddle_exc}, falling back to EasyOCR...")
    try:
        import easyocr
        import torch
        use_gpu = torch.cuda.is_available()
        OCR_MODEL = easyocr.Reader(['en'], gpu=use_gpu)
        ACTIVE_ENGINE = "EasyOCR"
        OCR_BACKEND = "easyocr"
        print(f"[EasyOCR] Loaded as fallback successfully (GPU={use_gpu})")
    except Exception as exc:
        OCR_INIT_ERROR = f"All OCR initializations failed: PaddleOCR ({paddle_exc}), EasyOCR ({exc})"
        print(f"[OCR ERROR] {OCR_INIT_ERROR}", file=sys.stderr)
        raise RuntimeError(OCR_INIT_ERROR) from exc

CHAR_WHITELIST = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

# Import ANPRX unified operations router and database management
import routes_gatesense
from db.session import init_db, verify_connectivity
from db.seed import seed_master_data

app = FastAPI(title="ANPRX Edge ANPR & Trip Management Backend", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
async def on_startup():
    print("[Startup] Initializing ANPRX MySQL production database...")
    is_connected = verify_connectivity()
    if is_connected:
        init_db()
        seed_master_data()
        print("[Startup] MySQL Database and Master Data ready!")
    else:
        print("[Startup ERROR] Could not connect to MySQL server. Please verify database is running.", file=sys.stderr)

# Mount ANPRX Operations API routes
app.include_router(routes_gatesense.router, prefix="/api")

# Mount static files for video streaming
STATIC_PARENT_DIR = os.path.dirname(STATIC_DIR)
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_PARENT_DIR), name="static")


class OcrRequest(BaseModel):
    image: str
    source: str | None = None


def decode_base64_image(raw_b64: str) -> np.ndarray:
    if not isinstance(raw_b64, str) or not raw_b64:
        raise ValueError("Image payload is required")
    if "," in raw_b64:
        raw_b64 = raw_b64.split(",", 1)[1]
    try:
        img_bytes = base64.b64decode(raw_b64, validate=False)
    except Exception as exc:  # pragma: no cover - validation path
        raise ValueError(f"Invalid base64 image payload: {exc}") from exc
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image from base64 payload")
    return img


def deskew_image(gray: np.ndarray) -> np.ndarray:
    coords = np.column_stack(np.where(gray > 0))
    if coords.shape[0] < 10:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 1.5 or abs(angle) > 15.0:
        return gray
    h, w = gray.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def gamma_correction(img: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)], dtype="uint8")
    return cv2.LUT(img, table)


def contrast_stretch(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    min_val, max_val = img.min(), img.max()
    if max_val == min_val:
        return img
    stretched = ((img - min_val) * 255.0 / (max_val - min_val)).astype(np.uint8)
    return stretched


def sharpen(img: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(img, -1, kernel)


def apply_morphology(img: np.ndarray, *, close: bool = False, open: bool = False) -> np.ndarray:
    result = img
    if close:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    if open:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    return result


def adaptive_equalize(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    return clahe.apply(gray)


def resize_to_ocr_height(img: np.ndarray, target_height: int = 120) -> np.ndarray:
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return img
    scale = target_height / float(h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


VALID_INDIAN_STATES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ",
    "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP",
    "WB", "BH"
}

STATE_CONFUSION_MAP = {
    "HH": "MH", "NH": "MH", "KH": "MH", "VH": "MH", "WH": "MH", "MM": "MH",
    "MI": "MH", "MN": "MH", "MR": "MH", "HM": "MH", "TH": "MH", "WW": "MH",
    "0L": "DL", "OL": "DL", "1L": "DL", "IL": "DL", "QL": "DL", "DI": "DL", "D1": "DL",
    "HA": "HR", "HB": "HR", "HD": "HR",
    "KB": "KA", "KP": "KL", "G3": "GJ", "R3": "RJ", "W8": "WB",
    "MB": "MP", "MD": "MP", "UB": "UP", "UF": "UP", "TJ": "TN", "TM": "TN"
}

CHAR_TO_DIGIT = {
    "O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "T": "7"
}

CHAR_TO_LETTER = {
    "0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G", "7": "T"
}


def normalize_plate_text(raw: str) -> str:
    cleaned = re.sub(r"\s+", "", (raw or "").upper())
    cleaned = re.sub(r"[^A-Z0-9]", "", cleaned)
    if len(cleaned) < 4:
        return ""

    # Strip leading 'IND' or 'IN'
    if cleaned.startswith("IND") and len(cleaned) >= 7:
        cleaned = cleaned[3:]
    elif cleaned.startswith("IN") and len(cleaned) >= 8 and (cleaned[2:4] in VALID_INDIAN_STATES or cleaned[2:4] in STATE_CONFUSION_MAP):
        cleaned = cleaned[2:]
    elif len(cleaned) >= 11 and cleaned[0] in {"1", "I", "L", "T"} and (cleaned[1:3] in VALID_INDIAN_STATES or cleaned[1:3] in STATE_CONFUSION_MAP):
        cleaned = cleaned[1:]

    # Remove trailing border artifacts (e.g. trailing 1, I, L, 7 after 4-digit number)
    m = re.match(r"^([A-Z0-9]{2}[A-Z0-9]{2}[A-Z0-9]{1,3}[A-Z0-9]{4})([1IL7TI])$", cleaned)
    if m:
        cleaned = m.group(1)
    else:
        m2 = re.match(r"^([A-Z0-9]{2}\d{1,2}[A-Z0-9]{1,3}\d{4})\d$", cleaned)
        if m2:
            cleaned = m2.group(1)

    return cleaned


def smart_indian_correction(plate: str) -> str:
    if not plate:
        return ""
    cleaned = normalize_plate_text(plate)
    chars = list(cleaned)
    
    # Standard 10-character Indian plate format (e.g. MH 20 DV 2366)
    if len(chars) == 10:
        chars[0] = CHAR_TO_LETTER.get(chars[0], chars[0])
        chars[1] = CHAR_TO_LETTER.get(chars[1], chars[1])
        st = chars[0] + chars[1]
        if st not in VALID_INDIAN_STATES and st in STATE_CONFUSION_MAP:
            chars[0], chars[1] = STATE_CONFUSION_MAP[st][0], STATE_CONFUSION_MAP[st][1]
        chars[2] = CHAR_TO_DIGIT.get(chars[2], chars[2])
        chars[3] = CHAR_TO_DIGIT.get(chars[3], chars[3])
        chars[4] = CHAR_TO_LETTER.get(chars[4], chars[4])
        chars[5] = CHAR_TO_LETTER.get(chars[5], chars[5])
        if chars[0] == "T" and chars[1] == "S" and chars[4] == "B" and chars[5] == "S":
            chars[4] = "J"
        for i in [6, 7, 8, 9]:
            chars[i] = CHAR_TO_DIGIT.get(chars[i], chars[i])
        return "".join(chars)

    # 9-character format (e.g. MH 02 D 1365)
    if len(chars) == 9:
        chars[0] = CHAR_TO_LETTER.get(chars[0], chars[0])
        chars[1] = CHAR_TO_LETTER.get(chars[1], chars[1])
        st = chars[0] + chars[1]
        if st not in VALID_INDIAN_STATES and st in STATE_CONFUSION_MAP:
            chars[0], chars[1] = STATE_CONFUSION_MAP[st][0], STATE_CONFUSION_MAP[st][1]
        chars[2] = CHAR_TO_DIGIT.get(chars[2], chars[2])
        chars[3] = CHAR_TO_DIGIT.get(chars[3], chars[3])
        chars[4] = CHAR_TO_LETTER.get(chars[4], chars[4])
        for i in [5, 6, 7, 8]:
            chars[i] = CHAR_TO_DIGIT.get(chars[i], chars[i])
        return "".join(chars)

    # Position-aware fallback corrections
    for idx, ch in enumerate(chars):
        if idx < 2:
            chars[idx] = CHAR_TO_LETTER.get(ch, ch)
        elif 2 <= idx < 4:
            chars[idx] = CHAR_TO_DIGIT.get(ch, ch)
        elif idx >= len(chars) - 4:
            chars[idx] = CHAR_TO_DIGIT.get(ch, ch)
        else:
            chars[idx] = CHAR_TO_LETTER.get(ch, ch)
    st = chars[0] + chars[1] if len(chars) >= 2 else ""
    if st in STATE_CONFUSION_MAP:
        chars[0], chars[1] = STATE_CONFUSION_MAP[st][0], STATE_CONFUSION_MAP[st][1]
    return "".join(chars)


def is_valid_indian_plate(candidate: str) -> Tuple[bool, float, str]:
    plate = normalize_plate_text(candidate)
    if not plate:
        return False, 0.0, "No text"
    corrected = smart_indian_correction(plate)
    state = corrected[:2] if len(corrected) >= 2 else ""
    is_valid_state = state in VALID_INDIAN_STATES

    patterns = [
        re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$"),
        re.compile(r"^\d{2}BH\d{4}[A-Z]{1,2}$"),
        re.compile(r"^[A-Z]{2}\d{1,2}\d[A-Z]{0,3}\d{4}$"),
        re.compile(r"^[A-Z]{2}\d{1,2}T[A-Z]{1,3}\d{4}$"),
        re.compile(r"^[A-Z]{2}\d{1,2}G[A-Z]{1,3}\d{4}$"),
        re.compile(r"^[A-Z]{2}\d{1,2}\d{4}$"),
        re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]\d{4}$"),
    ]
    if any(p.fullmatch(corrected) for p in patterns):
        return True, 0.98 if is_valid_state else 0.85, corrected
    
    if len(corrected) >= 8 and len(corrected) <= 11 and is_valid_state:
        return True, 0.80, corrected
    return False, 0.20, corrected


def preprocess_variant_a(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = deskew_image(gray)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    gray = adaptive_equalize(gray)
    gray = resize_to_ocr_height(gray)
    return gray


def preprocess_variant_b(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = deskew_image(gray)
    gray = cv2.medianBlur(gray, 3)
    gray = adaptive_equalize(gray)
    gray = sharpen(gray)
    gray = resize_to_ocr_height(gray)
    return gray


def preprocess_variant_c(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = deskew_image(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return resize_to_ocr_height(binary)


def preprocess_variant_d(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = deskew_image(gray)
    gray = cv2.medianBlur(gray, 3)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
    gray = apply_morphology(gray, close=True, open=True)
    return resize_to_ocr_height(gray)


def preprocess_variant_e(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = adaptive_equalize(gray)
    gray = contrast_stretch(gray)
    gray = gamma_correction(gray, 1.35)
    gray = sharpen(gray)
    return resize_to_ocr_height(gray)


def preprocess_variant_f(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = deskew_image(gray)
    gray = cv2.bilateralFilter(gray, 9, 60, 60)
    gray = cv2.medianBlur(gray, 5)
    gray = contrast_stretch(gray)
    return resize_to_ocr_height(gray)


def preprocess_variant_g(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = deskew_image(gray)
    gray = adaptive_equalize(gray)
    gray = apply_morphology(gray, close=True)
    gray = sharpen(gray)
    return resize_to_ocr_height(gray)


def preprocess_variant_h(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = adaptive_equalize(gray)
    gray = cv2.medianBlur(gray, 3)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = apply_morphology(binary, open=True)
    return resize_to_ocr_height(binary)


def build_variants(img: np.ndarray) -> List[Dict[str, Any]]:
    variants = [
        ("Raw", "Natural Enhanced Crop", lambda x: resize_to_ocr_height(x)),
        ("A", "CLAHE + Bilateral + Adaptive Threshold", preprocess_variant_a),
        ("B", "Median + CLAHE + Sharpen", preprocess_variant_b),
        ("C", "Otsu Binary", preprocess_variant_c),
        ("D", "Adaptive Threshold + Morphology", preprocess_variant_d),
        ("E", "Gamma + Contrast Stretch + Sharpen", preprocess_variant_e),
        ("F", "Bilateral + Median + Contrast Stretch", preprocess_variant_f),
        ("G", "CLAHE + Close + Sharpen", preprocess_variant_g),
        ("H", "Otsu + Open + Median", preprocess_variant_h),
    ]
    prepared: List[Dict[str, Any]] = []
    for variant_id, name, fn in variants:
        candidate = fn(img)
        if len(candidate.shape) == 2:
            candidate = cv2.cvtColor(candidate, cv2.COLOR_GRAY2BGR)
        prepared.append({"variantId": variant_id, "variantName": name, "image": candidate})
    return prepared


def parse_ocr_result(raw_result: Any) -> List[Tuple[str, float]]:
    if raw_result is None:
        return []
    parsed: List[Tuple[str, float]] = []

    # Unwrap outer page list if present: e.g. [[box1, box2]]
    items_to_process = raw_result
    if isinstance(items_to_process, list) and len(items_to_process) == 1 and isinstance(items_to_process[0], list):
        items_to_process = items_to_process[0]

    if isinstance(raw_result, dict):
        text = raw_result.get("rec_text") or raw_result.get("text") or raw_result.get("pred_text") or ""
        score = raw_result.get("rec_score") or raw_result.get("score") or raw_result.get("confidence") or 0.0
        normalized = normalize_plate_text(str(text))
        if normalized:
            parsed.append((normalized, float(score)))
        return parsed

    if isinstance(items_to_process, list):
        for item in items_to_process:
            if item is None:
                continue
            if isinstance(item, dict):
                text = item.get("rec_text") or item.get("text") or item.get("pred_text") or ""
                score = item.get("rec_score") or item.get("score") or item.get("confidence") or 0.0
                normalized = normalize_plate_text(str(text))
                if normalized:
                    parsed.append((normalized, float(score)))
                continue
            if isinstance(item, (list, tuple)):
                if len(item) >= 2 and isinstance(item[1], (list, tuple)) and len(item[1]) >= 2:
                    try:
                        _, text_score = item
                        text = text_score[0]
                        score = text_score[1]
                        normalized = normalize_plate_text(str(text))
                        if normalized:
                            parsed.append((normalized, float(score)))
                    except Exception:
                        pass
                elif len(item) == 2 and isinstance(item[1], (float, int, str)):
                    try:
                        text, score = item
                        normalized = normalize_plate_text(str(text))
                        if normalized:
                            parsed.append((normalized, float(score)))
                    except Exception:
                        pass
    return parsed


def run_ocr_for_variant(img: np.ndarray) -> Dict[str, Any]:
    try:
        if OCR_BACKEND == "easyocr":
            # EasyOCR readtext
            raw_results = OCR_MODEL.readtext(img, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            if not raw_results:
                # Retry without allowlist in case of contrast variation
                raw_results = OCR_MODEL.readtext(img)
            parsed = []
            for item in raw_results:
                if len(item) >= 3:
                    _, text, score = item[0], item[1], item[2]
                    norm = normalize_plate_text(str(text))
                    if norm:
                        parsed.append((norm, float(score)))
                elif len(item) == 2:
                    text, score = item[0], item[1]
                    norm = normalize_plate_text(str(text))
                    if norm:
                        parsed.append((norm, float(score)))
        elif OCR_BACKEND == "paddleocr":
            results = OCR_MODEL.ocr(img)
            parsed = parse_ocr_result(results)
        else:
            raise RuntimeError(f"Unsupported OCR backend: {OCR_BACKEND}")
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc

    texts: List[str] = []
    confidences: List[float] = []
    for text, score in parsed:
        cleaned = normalize_plate_text(text)
        if cleaned:
            texts.append(cleaned)
            confidences.append(float(score))

    if not texts:
        return {"text": "", "confidence": 0.0, "reason": "No OCR text extracted"}

    best_text = max(texts, key=lambda x: (len(x), texts.count(x)))
    best_confidence = max(confidences) if confidences else 0.0
    is_valid, validation_confidence, corrected = is_valid_indian_plate(best_text)
    if not is_valid:
        # Keep the best OCR result for debugging output instead of silently discarding it.
        return {
            "text": best_text,
            "confidence": float(best_confidence),
            "corrected": corrected,
            "reason": "Text extracted but failed Indian plate validation",
            "validation_confidence": float(validation_confidence),
            "valid": False,
        }
    return {
        "text": corrected,
        "confidence": float(best_confidence),
        "corrected": corrected,
        "reason": "Valid Indian plate pattern matched",
        "validation_confidence": float(validation_confidence),
        "valid": True,
    }


@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    return {"status": "ok", "engine": ACTIVE_ENGINE, "python": sys.version.split()[0], "port": 5001}


@app.get("/api/ai/health")
def ai_health() -> Dict[str, Any]:
    health = video_processor.check_health()
    if isinstance(health, dict):
        health["status"] = "healthy"
    return health


@app.get("/api/ai/tensorrt/status")
def tensorrt_status() -> Dict[str, Any]:
    """Returns real-time TensorRT acceleration diagnostics and engine file status."""
    return get_system_tensorrt_diagnostics()


@app.post("/api/ai/tensorrt/benchmark")
def tensorrt_benchmark(iterations: int = 20, model_name: str = "yolov8n") -> Dict[str, Any]:
    """Triggers an on-demand latency & FPS benchmark for TensorRT/PyTorch."""
    engine = TensorRTInferenceEngine(model_name=model_name, imgsz=384)
    return engine.benchmark(num_iterations=iterations, warmup_iterations=3)


INFERENCE_LOCK = asyncio.Lock()


@app.post("/api/video/process")
async def process_video(
    video: UploadFile = File(...),
    conf_threshold: float = Form(0.30),
    plate_conf_threshold: float = Form(0.25),
    process_fps: int = Form(5),
    vehicle_img_size: int = Form(384),
    max_process_fps: Optional[int] = Form(None)
) -> Dict[str, Any]:
    # 1. Save uploaded video to temporary file
    temp_dir = os.path.join(os.path.dirname(__file__), "temp_videos")
    os.makedirs(temp_dir, exist_ok=True)
    safe_filename = os.path.basename(video.filename or "uploaded_video.mp4")
    temp_path = os.path.join(temp_dir, f"upload_{int(time.time())}_{safe_filename}")

    async with INFERENCE_LOCK:
        try:
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(video.file, buffer)

            # 2. Process video frame-by-frame with optimized YOLO and EasyOCR models
            result = video_processor.process_video(
                video_path=temp_path,
                conf_threshold=conf_threshold,
                plate_conf_threshold=plate_conf_threshold,
                process_fps=process_fps,
                vehicle_img_size=vehicle_img_size,
                max_process_fps=max_process_fps
            )
            return result
        except Exception as exc:
            print(f"[ERROR] Video processing failed: {exc}", file=sys.stderr)
            raise HTTPException(status_code=500, detail=str(exc))
        finally:
            # Clean up temporary uploaded file after processing
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass


class CctvConnectRequest(BaseModel):
    rtsp_url: str


@app.post("/api/cctv/connect")
def cctv_connect(req: CctvConnectRequest) -> Dict[str, Any]:
    """Connect to RTSP / CCTV stream in background and start real-time ANPR inference."""
    return cctv_manager.connect(req.rtsp_url)


@app.post("/api/cctv/stop")
def cctv_stop() -> Dict[str, Any]:
    """Stop the active CCTV stream and release video capture."""
    return cctv_manager.stop()


@app.get("/api/cctv/status")
def cctv_status() -> Dict[str, Any]:
    """Get live status, FPS, and real-time detection list from the active CCTV stream."""
    return cctv_manager.get_status()


@app.get("/api/cctv/detections")
def cctv_detections() -> List[Dict[str, Any]]:
    """Get list of active and recent vehicle detections from the CCTV stream."""
    return cctv_manager.get_status().get("detections", [])


@app.get("/api/cctv/stream")
def cctv_stream():
    """High-speed MJPEG video stream with real-time bounding boxes and plate HUD for web browsers."""
    return StreamingResponse(
        cctv_manager.generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/api/ocr")
def run_paddle_ocr(req: OcrRequest) -> Dict[str, Any]:
    t0 = time.perf_counter()
    try:
        img = decode_base64_image(req.image)
    except Exception as exc:  # pragma: no cover - API validation path
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {exc}") from exc

    if img is None or img.size == 0:
        raise HTTPException(status_code=400, detail="Decoded image is empty")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    debug_dir = os.path.join(os.path.dirname(__file__), "debug")
    os.makedirs(debug_dir, exist_ok=True)
    img_path = os.path.join(debug_dir, f"crop_{timestamp}.jpg")
    cv2.imwrite(img_path, img)

    h, w = img.shape[:2]
    if w < 16 or h < 8:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "success": False,
            "plate": "",
            "text": "",
            "confidence": 0.0,
            "engine": ACTIVE_ENGINE,
            "processing_time_ms": elapsed_ms,
            "error": f"Crop size too small ({w}x{h}px)",
        }

    # ──────────────────────────────────────────────────────────────
    # BUG FIX: Run OCR directly on the received image.
    #
    # The frontend already applies its own preprocessing variants
    # (A–I) before calling this endpoint.  Each request contains a
    # SINGLE preprocessed crop.  Previously, build_variants() was
    # called here, generating 8 additional Python-side variants for
    # every incoming variant — resulting in ~72 double-preprocessed
    # OCR attempts per plate, which destroyed text quality.
    # ──────────────────────────────────────────────────────────────
    try:
        result = run_ocr_for_variant(img)
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "success": False,
            "plate": "",
            "text": "",
            "confidence": 0.0,
            "engine": ACTIVE_ENGINE,
            "processing_time_ms": elapsed_ms,
            "error": f"OCR error: {exc}",
        }

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    plate_text = str(result.get("text") or "")
    corrected = str(result.get("corrected") or plate_text)
    confidence = float(result.get("confidence", 0.0))

    if plate_text:
        result_payload = {
            "success": True,
            "plate": corrected or plate_text,
            "text": corrected or plate_text,
            "confidence": confidence,
            "engine": ACTIVE_ENGINE,
            "processing_time_ms": elapsed_ms,
            "cropWidth": w,
            "cropHeight": h,
        }
        
        with open(os.path.join(debug_dir, "ocr_log.txt"), "a") as f:
            log_entry = {
                "timestamp": timestamp,
                "width": w,
                "height": h,
                "success": result_payload["success"],
                "plate": result_payload["plate"],
                "confidence": result_payload["confidence"],
                "error": "",
                "raw_result": result
            }
            f.write(json.dumps(log_entry) + "\n")
            
        return result_payload

    result_payload = {
        "success": False,
        "plate": "",
        "text": "",
        "confidence": 0.0,
        "engine": ACTIVE_ENGINE,
        "processing_time_ms": elapsed_ms,
        "error": result.get("reason", "No text extracted"),
        "cropWidth": w,
        "cropHeight": h,
    }
    
    with open(os.path.join(debug_dir, "ocr_log.txt"), "a") as f:
        log_entry = {
            "timestamp": timestamp,
            "width": w,
            "height": h,
            "success": result_payload["success"],
            "plate": result_payload["plate"],
            "confidence": result_payload["confidence"],
            "error": result_payload.get("error", ""),
            "raw_result": result
        }
        f.write(json.dumps(log_entry) + "\n")
        
    return result_payload


@app.post("/api/debug/save_variant")
def debug_save_variant(req: OcrRequest, variant_id: str = "unknown") -> Dict[str, Any]:
    t0 = time.perf_counter()
    try:
        img = decode_base64_image(req.image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {exc}") from exc

    if img is None or img.size == 0:
        raise HTTPException(status_code=400, detail="Decoded image is empty")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    debug_dir = os.path.join(os.path.dirname(__file__), "debug")
    os.makedirs(debug_dir, exist_ok=True)
    img_path = os.path.join(debug_dir, f"variant_{variant_id}_{timestamp}.jpg")
    cv2.imwrite(img_path, img)

    try:
        result = run_ocr_for_variant(img)
    except Exception as exc:
        return {"error": str(exc)}

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    log_entry = {
        "timestamp": timestamp,
        "variant": variant_id,
        "success": result.get("valid", False),
        "plate": result.get("corrected") or result.get("text", ""),
        "confidence": result.get("confidence", 0.0),
        "time_ms": elapsed_ms,
        "raw_result": result
    }
    
    with open(os.path.join(debug_dir, "ocr_log.txt"), "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return log_entry

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001, log_level="info")
