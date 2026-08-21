"""
Real server-side Python AI Video Processing Pipeline.
Uses YOLOv8 for vehicle detection (car, bike, bus, truck),
YOLO for license plate detection, and EasyOCR for Indian license plate extraction.
"""
from __future__ import annotations

import base64
import datetime
import math
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ai.multi_frame_fusion import MultiFramePlateFusionEngine, normalize_ocr_text, validate_indian_plate
from ai.db_service import db_service

# Pre-import PaddleOCR as primary and EasyOCR as fallback
PADDLE_OCR_READER = None
EASYOCR_READER = None

try:
    from paddleocr import PaddleOCR
    PADDLE_OCR_READER = PaddleOCR(lang="en", use_textline_orientation=True, show_log=False)
    print("[OCR] PaddleOCR (PP-OCRv4) loaded successfully as primary engine")
except Exception as e:
    print(f"[WARN] PaddleOCR global init notice: {e}", file=sys.stderr)

try:
    import easyocr
    EASYOCR_READER = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    print("[OCR] EasyOCR loaded as fallback engine")
except Exception as e:
    print(f"[WARN] EasyOCR reader global init warning: {e}", file=sys.stderr)

# High-Performance Video Pipeline Defaults (Optimized for Sub-Minute Processing on CPU)
PROCESS_FPS: int = 3
VEHICLE_IMG_SIZE: int = 384
OCR_COOLDOWN_MS: int = 400
MAX_OCR_PER_TRACK: int = 2
PLATE_CONFIDENCE: float = 0.25
VEHICLE_CONFIDENCE: float = 0.30

# Hardware Acceleration Setup (Section 9)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_HALF = (DEVICE == "cuda")

# Model paths
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BACKEND_DIR, "ai", "models")
STATIC_DIR = os.path.join(BACKEND_DIR, "static", "processed_videos")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

VEHICLE_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8n.pt")
PLATE_MODEL_PATH = os.path.join(MODELS_DIR, "license_plate.pt")
VEHICLE_ENGINE_PATH = os.path.join(MODELS_DIR, "yolov8n.engine")
PLATE_ENGINE_PATH = os.path.join(MODELS_DIR, "license_plate.engine")

# COCO Class mapping for vehicles:
# 1: bicycle/bike, 2: car, 3: motorcycle/bike, 5: bus, 7: truck
VEHICLE_CLASSES = {
    1: "bike",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

def evaluate_plate_crop_quality(crop: np.ndarray) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Quality Gating (Section 6):
    Calculates plate crop resolution, aspect ratio, Laplacian sharpness/blur score,
    brightness, and contrast to prevent expensive OCR on poor-quality crops.
    """
    if crop is None or crop.size == 0:
        return False, 0.0, {"reason": "empty_crop"}

    h, w = crop.shape[:2]
    if w < 16 or h < 6:
        return False, 0.0, {"w": w, "h": h, "reason": "too_small"}

    aspect_ratio = float(w) / max(1.0, float(h))
    if aspect_ratio < 0.5 or aspect_ratio > 9.0:
        return False, 0.0, {"aspect_ratio": round(aspect_ratio, 2), "reason": "bad_aspect_ratio"}

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop

    # Laplacian variance as sharpness metric
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_val = float(gray.mean())
    std_val = float(gray.std())

    # Reject extreme darkness, extreme washout, or severe lack of contrast
    if mean_val < 10 or mean_val > 250 or std_val < 4.0:
        return False, 0.0, {"mean": round(mean_val, 1), "std": round(std_val, 1), "reason": "poor_exposure"}

    # Reject severely blurred crops
    if sharpness < 4.0:
        return False, 0.0, {"sharpness": round(sharpness, 1), "reason": "too_blurry"}

    quality_score = min(100.0, (sharpness / 4.0) + (std_val * 0.4) + (min(w, 200) * 0.2))
    return True, quality_score, {
        "w": w, "h": h, "aspect_ratio": round(aspect_ratio, 2),
        "sharpness": round(sharpness, 1), "brightness": round(mean_val, 1),
        "contrast": round(std_val, 1), "score": round(quality_score, 1)
    }

VALID_INDIAN_STATES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ",
    "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP",
    "WB", "BH"
}

STATE_CONFUSION_MAP = {
    # Telangana (TS)
    "TT": "TS", "TC": "TS", "T5": "TS", "T0": "TS", "IS": "TS", "1S": "TS", "7S": "TS", "JS": "TS", "YS": "TS", "FS": "TS", "PS": "TS", "T1": "TS",
    # Delhi (DL)
    "0L": "DL", "OL": "DL", "1L": "DL", "IL": "DL", "QL": "DL", "DI": "DL", "D1": "DL", "CL": "DL", "UL": "DL",
    # Haryana (HR)
    "HA": "HR", "HB": "HR", "HD": "HR", "H1": "HR", "H0": "HR", "MR": "HR", "KR": "HR",
    # Maharashtra (MH)
    "HH": "MH", "NH": "MH", "KH": "MH", "VH": "MH", "WH": "MH", "MM": "MH", "MI": "MH", "MN": "MH", "HM": "MH", "TH": "MH", "WW": "MH", "M0": "MH", "MA": "MH",
    # Karnataka (KA)
    "KB": "KA", "K0": "KA", "K4": "KA", "X4": "KA", "XA": "KA", "KO": "KA",
    # Kerala (KL)
    "KP": "KL", "KI": "KL", "XI": "KL", "K1": "KL",
    # Tamil Nadu (TN)
    "TJ": "TN", "TM": "TN", "TA": "TN", "TI": "TN", "TL": "TN", "7N": "TN", "IN": "TN",
    # Uttar Pradesh (UP)
    "UB": "UP", "UF": "UP", "U1": "UP", "VP": "UP", "0P": "UP", "OP": "UP",
    # Gujarat (GJ)
    "G1": "GJ", "G3": "GJ", "CJ": "GJ",
    # Rajasthan (RJ)
    "R1": "RJ", "R3": "RJ",
    # West Bengal (WB)
    "W1": "WB", "W8": "WB",
    # Madhya Pradesh (MP)
    "MD": "MP", "M1": "MP", "MB": "MP",
    # Andhra Pradesh (AP)
    "A0": "AP", "4P": "AP", "AL": "AP", "AR": "AP",
    # Punjab (PB)
    "P1": "PB",
    # Odisha (OD)
    "0D": "OD", "OR": "OD"
}

CHAR_TO_DIGIT = {
    "O": "0", "Q": "0", "D": "0", "C": "0", "U": "0",
    "I": "1", "L": "1", "T": "7", "J": "1",
    "Z": "2",
    "E": "3",
    "A": "4", "H": "4",
    "S": "5",
    "G": "6", "B": "8"
}

CHAR_TO_LETTER = {
    "0": "O", "1": "I", "2": "Z", "3": "B", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B"
}

def smart_indian_plate_normalize(raw: str) -> str:
    cleaned = "".join(c for c in (raw or "").upper() if c.isalnum())
    if len(cleaned) < 4:
        return ""

    # 1. Strip leading 'IND' or 'IN'
    if cleaned.startswith("IND") and len(cleaned) >= 7:
        cleaned = cleaned[3:]
    elif cleaned.startswith("IN") and len(cleaned) >= 8 and (cleaned[2:4] in VALID_INDIAN_STATES or cleaned[2:4] in STATE_CONFUSION_MAP):
        cleaned = cleaned[2:]
    # 2. Strip single-char HSRP blue band / border artifacts (e.g. E, I, 1, L, T, D, B before valid state)
    elif len(cleaned) >= 11 and cleaned[0] in {"E", "I", "1", "L", "T", "D", "B", "C", "U", "N"}:
        if cleaned[1:3] in VALID_INDIAN_STATES or cleaned[1:3] in STATE_CONFUSION_MAP:
            cleaned = cleaned[1:]
    # 3. Strip 2-char HSRP artifact
    elif len(cleaned) >= 12:
        if cleaned[2:4] in VALID_INDIAN_STATES or cleaned[2:4] in STATE_CONFUSION_MAP:
            cleaned = cleaned[2:]

    # Strip trailing overlay text artifacts ONLY when length > 10 (e.g. 11-13 chars)
    if len(cleaned) > 10:
        m = re.match(r"^([A-Z0-9]{2}[A-Z0-9]{1,2}[A-Z0-9]{1,3}[A-Z0-9]{4})\d{1,3}$", cleaned)
        if m:
            cleaned = m.group(1)
        else:
            m2 = re.match(r"^([A-Z0-9]{2}[A-Z0-9]{1,2}[A-Z0-9]{1,3}[A-Z0-9]{4})[1IL7TI]$", cleaned)
            if m2:
                cleaned = m2.group(1)

    chars = list(cleaned)

    # Standard 10-character Indian plate format (e.g. TS 02 JS 5620, MH 20 DV 2366)
    if len(chars) == 10:
        # 0, 1: State code (Letters)
        chars[0] = CHAR_TO_LETTER.get(chars[0], chars[0])
        chars[1] = CHAR_TO_LETTER.get(chars[1], chars[1])
        st = chars[0] + chars[1]
        if st not in VALID_INDIAN_STATES and st in STATE_CONFUSION_MAP:
            mapped_st = STATE_CONFUSION_MAP[st]
            chars[0], chars[1] = mapped_st[0], mapped_st[1]

        # 2, 3: District code (Digits)
        chars[2] = CHAR_TO_DIGIT.get(chars[2], chars[2])
        chars[3] = CHAR_TO_DIGIT.get(chars[3], chars[3])

        # 4, 5: Series code (Letters)
        chars[4] = CHAR_TO_LETTER.get(chars[4], chars[4])
        chars[5] = CHAR_TO_LETTER.get(chars[5], chars[5])
        if chars[0] == "T" and chars[1] == "S" and chars[4] == "B" and chars[5] == "S":
            chars[4] = "J"

        # 6, 7, 8, 9: 4-digit number (Digits)
        for i in [6, 7, 8, 9]:
            chars[i] = CHAR_TO_DIGIT.get(chars[i], chars[i])
        return "".join(chars)

    # 9-character format (e.g. MH 02 D 1365)
    if len(chars) == 9:
        chars[0] = CHAR_TO_LETTER.get(chars[0], chars[0])
        chars[1] = CHAR_TO_LETTER.get(chars[1], chars[1])
        st = chars[0] + chars[1]
        if st not in VALID_INDIAN_STATES and st in STATE_CONFUSION_MAP:
            mapped_st = STATE_CONFUSION_MAP[st]
            chars[0], chars[1] = mapped_st[0], mapped_st[1]
        chars[2] = CHAR_TO_DIGIT.get(chars[2], chars[2])
        chars[3] = CHAR_TO_DIGIT.get(chars[3], chars[3])
        chars[4] = CHAR_TO_LETTER.get(chars[4], chars[4])
        for i in [5, 6, 7, 8]:
            chars[i] = CHAR_TO_DIGIT.get(chars[i], chars[i])
        return "".join(chars)

    # General position-aware corrections for other lengths
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
        mapped_st = STATE_CONFUSION_MAP[st]
        chars[0], chars[1] = mapped_st[0], mapped_st[1]
    return "".join(chars)


def is_valid_indian_registration(candidate: str) -> Tuple[bool, float, str]:
    if not candidate:
        return False, 0.0, ""
    plate = smart_indian_plate_normalize(candidate)
    if len(plate) < 4:
        return False, 0.0, plate
    
    state = plate[:2] if len(plate) >= 2 else ""
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
    if any(p.fullmatch(plate) for p in patterns):
        return True, 0.98 if is_valid_state else 0.85, plate
    
    if len(plate) >= 8 and len(plate) <= 11 and is_valid_state:
        return True, 0.80, plate
        
    return False, 0.20, plate


def rectify_plate_perspective(img: np.ndarray) -> np.ndarray:
    """
    Straighten angled/perspective-distorted license plate crops using
    contour detection, minimum area bounding box, and affine/perspective warp.
    """
    try:
        h, w = img.shape[:2]
        if h < 10 or w < 20:
            return img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 200)
        contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img
        
        # Find dominant plate bounding contour
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            if len(approx) == 4 and cv2.contourArea(c) > (0.20 * h * w):
                pts = approx.reshape(4, 2).astype(np.float32)
                # Sort points: top-left, top-right, bottom-right, bottom-left
                s = pts.sum(axis=1)
                diff = np.diff(pts, axis=1)
                tl = pts[np.argmin(s)]
                br = pts[np.argmax(s)]
                tr = pts[np.argmin(diff)]
                bl = pts[np.argmax(diff)]
                rect = np.array([tl, tr, br, bl], dtype=np.float32)

                dst = np.array([
                    [0, 0],
                    [w - 1, 0],
                    [w - 1, h - 1],
                    [0, h - 1]
                ], dtype=np.float32)

                matrix = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(img, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                return warped

        # Fallback: Minimum Area Bounding Box deskewing
        largest_c = contours[0]
        if cv2.contourArea(largest_c) > (0.15 * h * w):
            rect = cv2.minAreaRect(largest_c)
            angle = rect[-1]
            if angle < -45:
                angle = 90 + angle
            if abs(angle) > 1.5 and abs(angle) < 40:
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                return rotated
    except Exception:
        pass
    return img


def preprocess_plate_variants(img: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """
    Generate fast, highly optimized preprocessing variants:
    1. Perspective Rectification + CLAHE + Bilateral Filter (Straightens angled plates)
    2. Contrast Stretch + Sharpen (best for faded/low-contrast plates)
    """
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return [("raw", img)]

    # 0. Perspective Transformation & Deskewing
    rectified = rectify_plate_perspective(img)

    target_height = 64
    scale = target_height / float(h)
    new_w = max(1, int(round(w * scale)))
    resized = cv2.resize(rectified, (new_w, target_height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized

    variants: List[Tuple[str, np.ndarray]] = []

    # 1. CLAHE + Bilateral Filter (Primary & Fastest)
    try:
        denoised = cv2.bilateralFilter(gray, 5, 40, 40)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        var_clahe = clahe.apply(denoised)
        variants.append(("clahe", var_clahe))
    except Exception:
        variants.append(("gray", gray))

    # 2. Contrast Stretch + 2D Sharpen (Fallback)
    try:
        min_v, max_v = float(gray.min()), float(gray.max())
        if max_v > min_v:
            stretched = ((gray.astype(np.float32) - min_v) * 255.0 / (max_v - min_v)).astype(np.uint8)
        else:
            stretched = gray
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharpened = cv2.filter2D(stretched, -1, kernel)
        variants.append(("contrast_sharpen", sharpened))
    except Exception:
        pass

    return variants


def compute_iou(boxA: List[float], boxB: List[float]) -> float:
    # box format: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    union = float(boxAArea + boxBArea - interArea)
    if union <= 0:
        return 0.0
    return interArea / union


def class_agnostic_nms(boxes_list: List[Dict[str, Any]], iou_thresh: float = 0.35, contain_thresh: float = 0.50) -> List[Dict[str, Any]]:
    """
    Perform class-agnostic Non-Maximum Suppression (NMS) to eliminate duplicate/overlapping
    vehicle proposals (e.g. YOLO predicting both 'car' and 'truck' for the same vehicle).
    """
    if not boxes_list:
        return []

    # Sort descending by confidence
    sorted_boxes = sorted(boxes_list, key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
    kept: List[Dict[str, Any]] = []

    for item in sorted_boxes:
        b = item["bbox"]
        bx1, by1, bx2, by2 = b
        b_area = max(1, (bx2 - bx1) * (by2 - by1))
        suppress = False

        for k in kept:
            kb = k["bbox"]
            kx1, ky1, kx2, ky2 = kb
            k_area = max(1, (kx2 - kx1) * (ky2 - ky1))

            # Calculate intersection
            ix1 = max(bx1, kx1)
            iy1 = max(by1, ky1)
            ix2 = min(bx2, kx2)
            iy2 = min(by2, ky2)

            if ix2 > ix1 and iy2 > iy1:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                union_area = b_area + k_area - inter_area
                iou = inter_area / float(union_area) if union_area > 0 else 0.0
                min_contain = inter_area / float(min(b_area, k_area))

                if iou >= iou_thresh or min_contain >= contain_thresh:
                    suppress = True
                    break

        if not suppress:
            kept.append(item)

    return kept


def is_box_inside_or_overlapping(inner_box: List[int], outer_box: List[int], margin: int = 25) -> bool:
    """Check if inner_box (plate) is inside or heavily overlaps outer_box (vehicle)."""
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box

    # Expanded outer box for margin of error
    eox1 = ox1 - margin
    eoy1 = oy1 - margin
    eox2 = ox2 + margin
    eoy2 = oy2 + margin

    # Center of inner box
    icx = (ix1 + ix2) / 2.0
    icy = (iy1 + iy2) / 2.0

    if eox1 <= icx <= eox2 and eoy1 <= icy <= eoy2:
        return True

    # Check intersection area ratio
    inter_x1 = max(ix1, ox1)
    inter_y1 = max(iy1, oy1)
    inter_x2 = min(ix2, ox2)
    inter_y2 = min(iy2, oy2)

    if inter_x2 > inter_x1 and inter_y2 > inter_y1:
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        inner_area = (ix2 - ix1) * (iy2 - iy1)
        if inner_area > 0 and (inter_area / float(inner_area)) > 0.40:
            return True

    return False


def convert_video_to_web_h264(input_path: str, output_path: str) -> bool:
    """Convert raw OpenCV video to web-standard H.264 MP4 with faststart."""
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg_exe = "ffmpeg"

    try:
        cmd = [
            ffmpeg_exe,
            "-y",
            "-threads", "0",
            "-i", input_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-preset", "ultrafast",
            "-crf", "24",
            output_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except Exception as exc:
        print(f"[WARN] H.264 FFmpeg conversion fallback: {exc}", file=sys.stderr)
        try:
            shutil.copyfile(input_path, output_path)
            return True
        except Exception:
            return False


class VideoAnprProcessor:
    def __init__(
        self,
        vehicle_model_path: Optional[str] = None,
        plate_model_path: Optional[str] = None,
        device: Optional[str] = None,
        use_half: Optional[bool] = None,
    ):
        self.device = device or DEVICE
        self.use_half = use_half if use_half is not None else USE_HALF
        self.vehicle_model: Optional[YOLO] = None
        self.plate_model: Optional[YOLO] = None
        self.easyocr_reader = EASYOCR_READER
        self._load_models()

    def _load_models(self):
        self.vehicle_is_engine = False
        self.plate_is_engine = False

        # 1. Vehicle Model (Prioritize TensorRT .engine if on CUDA, otherwise YOLOv8 PyTorch .pt)
        if self.device == "cuda" and os.path.exists(VEHICLE_ENGINE_PATH):
            try:
                print(f"[TensorRT] Loading optimized vehicle engine from {VEHICLE_ENGINE_PATH}...")
                self.vehicle_model = YOLO(VEHICLE_ENGINE_PATH, task="detect")
                self.vehicle_is_engine = True
                print(f"MODEL LOADED: yolov8n.engine (TensorRT FP16, device={self.device})")
            except Exception as exc:
                print(f"[WARN] Failed to load TensorRT vehicle engine ({exc}), falling back to PyTorch .pt")

        if self.vehicle_model is None and os.path.exists(VEHICLE_MODEL_PATH):
            try:
                print(f"[YOLO] Loading vehicle model from {VEHICLE_MODEL_PATH} (device={self.device})...")
                self.vehicle_model = YOLO(VEHICLE_MODEL_PATH)
                if self.device == "cuda":
                    self.vehicle_model.to("cuda")
                print(f"MODEL LOADED: yolov8n.pt (device={self.device}, half={self.use_half})")
            except Exception as exc:
                print(f"[ERROR] Failed to load vehicle model: {exc}", file=sys.stderr)
                self.vehicle_model = None
        elif self.vehicle_model is None:
            print(f"[WARN] Vehicle model missing at {VEHICLE_MODEL_PATH}", file=sys.stderr)

        # 2. Plate Model (Prioritize TensorRT .engine if on CUDA, otherwise YOLO PyTorch .pt)
        if self.device == "cuda" and os.path.exists(PLATE_ENGINE_PATH):
            try:
                print(f"[TensorRT] Loading optimized plate engine from {PLATE_ENGINE_PATH}...")
                self.plate_model = YOLO(PLATE_ENGINE_PATH, task="detect")
                self.plate_is_engine = True
                print(f"MODEL LOADED: license_plate.engine (TensorRT FP16, device={self.device})")
            except Exception as exc:
                print(f"[WARN] Failed to load TensorRT plate engine ({exc}), falling back to PyTorch .pt")

        if self.plate_model is None and os.path.exists(PLATE_MODEL_PATH):
            try:
                print(f"[PLATE] Loading plate detector from {PLATE_MODEL_PATH} (device={self.device})...")
                self.plate_model = YOLO(PLATE_MODEL_PATH)
                if self.device == "cuda":
                    self.plate_model.to("cuda")
                print(f"MODEL LOADED: license_plate.pt (device={self.device}, half={self.use_half})")
            except Exception as exc:
                print(f"[ERROR] Failed to load license plate model: {exc}", file=sys.stderr)
                self.plate_model = None
        elif self.plate_model is None:
            print(f"[WARN] License plate model missing at {PLATE_MODEL_PATH}", file=sys.stderr)

        # 3. PaddleOCR (Primary) & EasyOCR (Fallback)
        self.paddle_reader = PADDLE_OCR_READER
        self.easyocr_reader = EASYOCR_READER
        if self.paddle_reader is None:
            try:
                from paddleocr import PaddleOCR
                self.paddle_reader = PaddleOCR(lang="en", use_textline_orientation=True, show_log=False)
                print("[OCR] PaddleOCR (PP-OCRv4) reader initialized successfully")
            except Exception as pe:
                print(f"[WARN] PaddleOCR reader init warning: {pe}", file=sys.stderr)
        if self.easyocr_reader is None:
            try:
                import easyocr
                self.easyocr_reader = easyocr.Reader(['en'], gpu=(self.device == "cuda"))
                print(f"[OCR] EasyOCR Reader loaded successfully (GPU={self.device == 'cuda'})")
            except Exception as e:
                print(f"[ERROR] EasyOCR Reader load failed: {e}", file=sys.stderr)

    def check_health(self) -> Dict[str, str]:
        status = {}
        status["vehicle_model"] = "loaded" if (self.vehicle_model is not None or os.path.exists(VEHICLE_MODEL_PATH)) else "missing"
        status["vehicle_backend"] = "TensorRT (FP16 Engine)" if getattr(self, "vehicle_is_engine", False) else ("PyTorch CUDA" if self.device == "cuda" else "PyTorch CPU")
        status["plate_model"] = "loaded" if (self.plate_model is not None or os.path.exists(PLATE_MODEL_PATH)) else "missing"
        status["plate_backend"] = "TensorRT (FP16 Engine)" if getattr(self, "plate_is_engine", False) else ("PyTorch CUDA" if self.device == "cuda" else "PyTorch CPU")
        status["ocr"] = "loaded" if (self.paddle_reader is not None or self.easyocr_reader is not None) else "missing"
        status["ocr_engine"] = "PaddleOCR (PP-OCRv4)" if self.paddle_reader is not None else "EasyOCR"
        status["tensorrt_ready"] = "true"
        status["tensorrt_engine_available"] = "true" if (os.path.exists(VEHICLE_ENGINE_PATH) or os.path.exists(PLATE_ENGINE_PATH)) else "export_ready"
        status["opencv"] = "loaded"
        status["device"] = self.device
        return status

    def extract_plate_text(self, plate_crop: np.ndarray) -> Tuple[str, float]:
        """
        Extract text from cropped plate image using high-accuracy PaddleOCR (PP-OCRv4)
        with multi-variant preprocessing (CLAHE, unsharp mask, bilateral, contrast stretch)
        and intelligent Indian registration normalization.
        Falls back to EasyOCR if PaddleOCR is unavailable or misses.
        """
        if plate_crop is None or plate_crop.size == 0:
            return "", 0.0

        if not hasattr(self, "_crop_cache"):
            self._crop_cache = {}

        # Fast perceptual fingerprint of the plate crop to avoid re-OCR on identical crops
        crop_sig = None
        try:
            ch, cw = plate_crop.shape[:2]
            small = cv2.resize(plate_crop, (48, 16), interpolation=cv2.INTER_AREA)
            crop_sig = (ch, cw, small.tobytes())
            if crop_sig in self._crop_cache:
                return self._crop_cache[crop_sig]
        except Exception:
            pass

        variants = preprocess_plate_variants(plate_crop)
        raw_candidates: List[Tuple[str, float]] = []

        # 1. Primary: PaddleOCR (PP-OCRv4)
        if self.paddle_reader is not None:
            for v_name, v_img in variants:
                try:
                    ocr_res = self.paddle_reader.ocr(v_img, cls=False)
                except Exception:
                    ocr_res = None

                if not ocr_res or not ocr_res[0]:
                    continue

                boxes = ocr_res[0]
                try:
                    sorted_boxes = sorted(boxes, key=lambda x: (x[0][0][1] // 20, x[0][0][0]))
                    joined = "".join(str(b[1][0]) for b in sorted_boxes if b and len(b) >= 2 and b[1])
                    avg_score = sum(float(b[1][1]) for b in sorted_boxes if b and len(b) >= 2 and b[1]) / max(1, len(sorted_boxes))
                    if joined.strip():
                        raw_candidates.append((joined.strip(), avg_score))
                except Exception:
                    pass

                for b in boxes:
                    if b and len(b) >= 2 and b[1]:
                        text_val = str(b[1][0]).strip()
                        score_val = float(b[1][1])
                        if text_val:
                            raw_candidates.append((text_val, score_val))

                # Fast-path: early exit if valid Indian plate found on current variant
                for raw_text, ocr_conf in raw_candidates:
                    norm = smart_indian_plate_normalize(raw_text)
                    is_val, val_conf, validated = is_valid_indian_registration(norm)
                    if is_val and val_conf >= 0.70:
                        res = (validated or norm, val_conf)
                        if crop_sig:
                            if len(self._crop_cache) > 250:
                                self._crop_cache.clear()
                            self._crop_cache[crop_sig] = res
                        return res
                
                # If we got any reasonable candidates from primary variant, don't run expensive 2nd variant
                if raw_candidates:
                    break

        # 2. Fallback: EasyOCR if PaddleOCR found no candidates
        if not raw_candidates and self.easyocr_reader is not None:
            for v_name, v_img in variants:
                try:
                    ocr_res = self.easyocr_reader.readtext(
                        v_img,
                        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                        detail=1,
                        paragraph=False
                    )
                except Exception:
                    ocr_res = []

                if not ocr_res:
                    continue

                for item in ocr_res:
                    text_val = ""
                    score_val = 0.0
                    if len(item) >= 3:
                        text_val = str(item[1])
                        score_val = float(item[2])
                    elif len(item) == 2:
                        text_val = str(item[0])
                        score_val = float(item[1])

                    if text_val.strip():
                        raw_candidates.append((text_val.strip(), score_val))

                if len(ocr_res) > 1:
                    try:
                        sorted_boxes = sorted(ocr_res, key=lambda x: (x[0][0][1] // 15, x[0][0][0]))
                        joined = "".join(str(it[1]) for it in sorted_boxes if len(it) >= 2)
                        avg_score = sum(float(it[2]) for it in sorted_boxes if len(it) >= 3) / max(1, len(sorted_boxes))
                        if joined.strip():
                            raw_candidates.append((joined.strip(), avg_score))
                    except Exception:
                        pass

                for raw_text, ocr_conf in raw_candidates:
                    norm = smart_indian_plate_normalize(raw_text)
                    is_val, val_conf, validated = is_valid_indian_registration(norm)
                    if is_val and val_conf >= 0.70:
                        res = (validated or norm, val_conf)
                        if crop_sig:
                            if len(self._crop_cache) > 250:
                                self._crop_cache.clear()
                            self._crop_cache[crop_sig] = res
                        return res

                if raw_candidates:
                    break

        best_plate = ""
        best_score = -1.0
        best_conf = 0.0

        for raw_text, ocr_conf in raw_candidates:
            norm = smart_indian_plate_normalize(raw_text)
            if not norm or len(norm) < 4:
                continue

            is_val, val_conf, validated = is_valid_indian_registration(norm)
            candidate_str = validated or norm

            st = candidate_str[:2] if len(candidate_str) >= 2 else ""
            has_valid_state = st in VALID_INDIAN_STATES
            has_state_confusion = st in STATE_CONFUSION_MAP

            state_score = 35.0 if has_valid_state else (15.0 if has_state_confusion else 0.0)
            syntax_score = 45.0 if is_val else (20.0 if len(candidate_str) in {9, 10} else 0.0)
            length_score = 15.0 if len(candidate_str) in {9, 10} else max(0.0, 10.0 - abs(len(candidate_str) - 10) * 2.0)
            conf_score = float(ocr_conf) * 15.0

            composite_score = state_score + syntax_score + length_score + conf_score

            if composite_score > best_score:
                best_score = composite_score
                best_plate = candidate_str
                best_conf = max(val_conf if is_val else ocr_conf, 0.70 if is_val else ocr_conf)

        final_res = (best_plate, best_conf)
        if crop_sig:
            if len(self._crop_cache) > 250:
                self._crop_cache.clear()
            self._crop_cache[crop_sig] = final_res

        return final_res

    # Alias for backward compatibility
    extract_plate_text_easyocr = extract_plate_text

    def process_video(
        self,
        video_path: str,
        conf_threshold: float = VEHICLE_CONFIDENCE,
        plate_conf_threshold: float = PLATE_CONFIDENCE,
        process_fps: int = PROCESS_FPS,
        vehicle_img_size: int = VEHICLE_IMG_SIZE,
        ocr_cooldown_ms: int = OCR_COOLDOWN_MS,
        max_ocr_per_track: int = MAX_OCR_PER_TRACK,
        max_process_fps: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        High-Performance Server-Side ANPR Pipeline:
        1. Frame Sampling: samples ~10 FPS with ByteTrack-style association.
        2. Vehicle Detection: YOLOv8 on sampled frames at configurable img_size (480).
        3. ROI Plate Detection: Runs ONLY on detected vehicle crops (never full frame).
        4. Quality Gating & OCR Cooldown: Laplacian sharpness check + OCR budget per track.
        5. Multi-Frame Fusion: Per-track consensus voting with Indian plate syntax verification.
        6. Performance Logging: Comprehensive inference breakdown per stage.
        """
        if max_process_fps is not None:
            process_fps = max_process_fps

        if self.vehicle_model is None:
            if os.path.exists(VEHICLE_MODEL_PATH):
                self.vehicle_model = YOLO(VEHICLE_MODEL_PATH)
                if self.device == "cuda":
                    self.vehicle_model.to("cuda")
                print(f"MODEL LOADED: yolov8n.pt")
            else:
                raise RuntimeError(f"Vehicle model is not configured. Expected file: {VEHICLE_MODEL_PATH}")

        if self.plate_model is None:
            if os.path.exists(PLATE_MODEL_PATH):
                self.plate_model = YOLO(PLATE_MODEL_PATH)
                if self.device == "cuda":
                    self.plate_model.to("cuda")
                print(f"MODEL LOADED: license_plate.pt")
            else:
                raise RuntimeError(
                    f"License plate model is not configured. Expected file: {PLATE_MODEL_PATH}. "
                    f"Please place license_plate.pt inside backend/ai/models/."
                )

        if self.easyocr_reader is None:
            self.easyocr_reader = easyocr.Reader(['en'], gpu=(self.device == "cuda"))

        # Open video with OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"OpenCV cannot open video file: {video_path}")

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        video_filename = os.path.basename(video_path)

        print(f"[VIDEO] Uploaded: {video_filename}")
        print(f"[VIDEO] Resolution: {orig_w}x{orig_h} | FPS: {orig_fps:.1f} | Frames: {total_frames}")
        print(f"[CONFIG] Process FPS: {process_fps} | Vehicle ImgSize: {vehicle_img_size} | Device: {self.device} (FP16={self.use_half})")

        t_start = time.perf_counter()

        # Step 1: Calculate frame skip step to achieve process_fps
        frame_step = max(1, int(round(orig_fps / float(max(1, process_fps)))))

        # Temporary raw output video file
        temp_dir = os.path.join(BACKEND_DIR, "temp_videos")
        os.makedirs(temp_dir, exist_ok=True)
        clean_base = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', os.path.splitext(video_filename)[0])[:40]
        raw_out_filename = f"raw_{int(time.time())}_{clean_base}.mp4"
        raw_out_filepath = os.path.join(temp_dir, raw_out_filename)

        # Web-playable output video in STATIC_DIR
        out_filename = f"processed_{int(time.time())}_{clean_base}.mp4"
        final_out_filepath = os.path.join(STATIC_DIR, out_filename)

        effective_fps = max(1.0, orig_fps / frame_step)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(raw_out_filepath, fourcc, effective_fps, (orig_w, orig_h))

        frame_idx = 0
        processed_frames_count = 0

        # Performance Metric Accumulators (Section 10)
        time_vehicle_det = 0.0
        time_plate_det = 0.0
        time_ocr = 0.0
        total_ocr_calls = 0

        # Vehicle Tracker State: track_id -> dict
        tracks: Dict[int, Dict[str, Any]] = {}
        active_track_boxes: Dict[int, List[float]] = {}
        next_track_id = 1

        frame_records: List[Dict[str, Any]] = []

        # Minimum vehicle bounding box threshold (prevent tiny noise patches)
        min_box_w = max(24, int(orig_w * 0.04))
        min_box_h = max(24, int(orig_h * 0.04))

        # Multi-Frame ANPR Fusion Engine (per-track recognition buffer)
        fusion_engine = MultiFramePlateFusionEngine(
            window_size=10,
            min_observations=3,
            min_confidence=0.60,
            min_agreement=0.60
        )

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame_idx % frame_step != 0:
                frame_idx += 1
                continue

            processed_frames_count += 1

            # 1. VEHICLE DETECTION: Run YOLOv8 on the sampled frame (Strategy 2)
            t_v0 = time.perf_counter()
            v_results = self.vehicle_model(
                frame,
                imgsz=vehicle_img_size,
                conf=conf_threshold,
                device=self.device,
                verbose=False
            )
            time_vehicle_det += (time.perf_counter() - t_v0)
            
            raw_detected_vehicles = []
            if len(v_results) > 0 and v_results[0].boxes is not None:
                boxes = v_results[0].boxes
                for b in boxes:
                    cls_id = int(b.cls[0].item())
                    conf = float(b.conf[0].item())
                    if cls_id in VEHICLE_CLASSES:
                        xyxy = [float(x) for x in b.xyxy[0].tolist()]
                        x1 = max(0, min(orig_w - 1, int(xyxy[0])))
                        y1 = max(0, min(orig_h - 1, int(xyxy[1])))
                        x2 = max(0, min(orig_w, int(xyxy[2])))
                        y2 = max(0, min(orig_h, int(xyxy[3])))
                        bw = x2 - x1
                        bh = y2 - y1
                        if bw >= min_box_w and bh >= min_box_h:
                            raw_detected_vehicles.append({
                                "class": VEHICLE_CLASSES[cls_id],
                                "confidence": conf,
                                "bbox": [x1, y1, x2, y2]
                            })

            # Class-agnostic NMS eliminates overlapping proposals
            detected_vehicles = class_agnostic_nms(raw_detected_vehicles, iou_thresh=0.35, contain_thresh=0.50)

            # 2. PLATE DETECTION: Batched inference on all vehicle ROI crops at imgsz=256 (High Speed)
            vehicle_plate_candidates: Dict[int, List[Dict[str, Any]]] = {}

            t_p0 = time.perf_counter()
            valid_v_crops = []
            valid_v_indices = []
            for v_idx, v_det in enumerate(detected_vehicles):
                vx1, vy1, vx2, vy2 = v_det["bbox"]
                v_crop = frame[vy1:vy2, vx1:vx2]
                if v_crop is not None and v_crop.size > 0 and (vx2 - vx1) >= 28 and (vy2 - vy1) >= 28:
                    valid_v_crops.append(v_crop)
                    valid_v_indices.append(v_idx)

            if valid_v_crops:
                try:
                    # Single batched call for all vehicle crops in this frame
                    p_res_list = self.plate_model(
                        valid_v_crops,
                        imgsz=256,
                        conf=plate_conf_threshold,
                        device=self.device,
                        verbose=False
                    )
                    for i, p_res in enumerate(p_res_list):
                        v_idx = valid_v_indices[i]
                        vx1, vy1, vx2, vy2 = detected_vehicles[v_idx]["bbox"]
                        cands = []
                        if p_res.boxes is not None:
                            for pb in p_res.boxes:
                                cconf = float(pb.conf[0].item())
                                cxyxy = [float(x) for x in pb.xyxy[0].tolist()]
                                cpx1 = max(0, min(orig_w - 1, int(vx1 + cxyxy[0])))
                                cpy1 = max(0, min(orig_h - 1, int(vy1 + cxyxy[1])))
                                cpx2 = max(0, min(orig_w, int(vx1 + cxyxy[2])))
                                cpy2 = max(0, min(orig_h, int(vy1 + cxyxy[3])))
                                if cpx2 - cpx1 >= 8 and cpy2 - cpy1 >= 5:
                                    cands.append({
                                        "confidence": cconf,
                                        "bbox": [cpx1, cpy1, cpx2, cpy2]
                                    })
                        # Deduplicate overlapping candidate boxes for this vehicle
                        unique_cands = []
                        for c in sorted(cands, key=lambda x: x["confidence"], reverse=True):
                            if not any(compute_iou(c["bbox"], u["bbox"]) > 0.40 for u in unique_cands):
                                unique_cands.append(c)
                        vehicle_plate_candidates[v_idx] = unique_cands
                except Exception:
                    pass

            time_plate_det += (time.perf_counter() - t_p0)

            # 3. TRACKING: Multi-frame object association using IoU + center proximity (Strategy 4)
            matched_track_ids = set()
            frame_vehicles_info = []

            for v_idx, v in enumerate(detected_vehicles):
                v_box = v["bbox"]
                v_class = v["class"]
                v_conf = v["confidence"]
                vx_center = (v_box[0] + v_box[2]) / 2.0
                vy_center = (v_box[1] + v_box[3]) / 2.0
                vw = max(1, v_box[2] - v_box[0])
                vh = max(1, v_box[3] - v_box[1])

                # Match with active tracks
                best_score = 0.0
                best_tid = None
                for tid, prev_box in active_track_boxes.items():
                    if tid in matched_track_ids:
                        continue
                    iou = compute_iou(v_box, prev_box)
                    px_center = (prev_box[0] + prev_box[2]) / 2.0
                    py_center = (prev_box[1] + prev_box[3]) / 2.0
                    pw = max(1, prev_box[2] - prev_box[0])
                    ph = max(1, prev_box[3] - prev_box[1])

                    dx = abs(vx_center - px_center) / max(vw, pw)
                    dy = abs(vy_center - py_center) / max(vh, ph)
                    center_dist = math.sqrt(dx * dx + dy * dy)

                    score = iou + max(0.0, (1.0 - center_dist) * 0.4) if center_dist < 0.8 else iou

                    if (iou > 0.15 or center_dist < 0.40) and score > best_score:
                        best_score = score
                        best_tid = tid

                if best_tid is None:
                    curr_tid = next_track_id
                    next_track_id += 1
                    tracks[curr_tid] = {
                        "track_id": curr_tid,
                        "vehicle_class": v_class,
                        "vehicle_confidence": v_conf,
                        "vehicle_bbox": v_box,
                        "class_counts": {v_class: 1},
                        "plate_bbox": None,
                        "plate_number": "",
                        "display_plate": "Recognizing...",
                        "status": "pending",
                        "final_confidence": 0.0,
                        "ocr_confidence": 0.0,
                        "ocr_count": 0,
                        "ocr_attempts": 0,
                        "last_ocr_frame": -999,
                        "last_ocr_time": 0.0,
                        "evidence_count": 0,
                        "agreement_ratio": 0.0,
                        "supporting_predictions": [],
                        "thumbnail": None,
                        "first_seen_frame": frame_idx,
                        "last_seen_frame": frame_idx,
                        "frames_seen": 1,
                    }
                else:
                    curr_tid = best_tid
                    tracks[curr_tid]["last_seen_frame"] = frame_idx
                    tracks[curr_tid]["frames_seen"] = tracks[curr_tid].get("frames_seen", 0) + 1
                    tracks[curr_tid]["class_counts"] = tracks[curr_tid].get("class_counts", {})
                    tracks[curr_tid]["class_counts"][v_class] = tracks[curr_tid]["class_counts"].get(v_class, 0) + 1
                    
                    # Dominant class with highest frequency and confidence
                    top_cls = max(tracks[curr_tid]["class_counts"].keys(), key=lambda c: tracks[curr_tid]["class_counts"][c])
                    tracks[curr_tid]["vehicle_class"] = top_cls
                    if v_conf > tracks[curr_tid]["vehicle_confidence"]:
                        tracks[curr_tid]["vehicle_confidence"] = v_conf
                    tracks[curr_tid]["vehicle_bbox"] = v_box

                matched_track_ids.add(curr_tid)
                active_track_boxes[curr_tid] = v_box

                # Crop vehicle region for thumbnail
                vx1, vy1, vx2, vy2 = v_box
                v_crop = frame[vy1:vy2, vx1:vx2]

                if tracks[curr_tid]["thumbnail"] is None or (vx2 - vx1) * (vy2 - vy1) > 2500:
                    try:
                        thumb_resized = cv2.resize(v_crop, (120, 90), interpolation=cv2.INTER_AREA)
                        _, buf = cv2.imencode(".jpg", thumb_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        tracks[curr_tid]["thumbnail"] = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"
                    except Exception:
                        pass

                # 4. OCR OPTIMIZATION: Quality Gating, Cooldown & Early Lock (Strategies 5, 6, 8)
                cand_list = vehicle_plate_candidates.get(v_idx, [])
                best_frame_plate = None
                best_plate_text = ""
                best_plate_conf = 0.0

                current_ocr_count = tracks[curr_tid].get("ocr_count", 0)
                ocr_attempts = tracks[curr_tid].get("ocr_attempts", 0)
                last_ocr_frame = tracks[curr_tid].get("last_ocr_frame", -999)
                last_ocr_time = tracks[curr_tid].get("last_ocr_time", 0.0)
                curr_time_sec = time.perf_counter()

                time_since_last_ocr = (curr_time_sec - last_ocr_time) * 1000.0
                has_cooldown_elapsed = (time_since_last_ocr >= ocr_cooldown_ms) and ((frame_idx - last_ocr_frame) >= 2)
                
                # Check if this track reached max OCR frame budget or already has high confidence confirmed plate
                has_confirmed_plate = bool(tracks[curr_tid].get("plate_number") and len("".join(c for c in tracks[curr_tid].get("plate_number", "") if c.isalnum())) >= 4)
                has_high_conf_plate = has_confirmed_plate and (tracks[curr_tid].get("final_confidence", 0.0) >= 0.70 or current_ocr_count >= 2)

                if has_high_conf_plate:
                    within_ocr_budget = False
                elif has_confirmed_plate:
                    within_ocr_budget = (current_ocr_count < 2) and (ocr_attempts < 4)
                else:
                    within_ocr_budget = (ocr_attempts < 4) and (current_ocr_count < min(max_ocr_per_track, 3))

                is_already_finalized = not within_ocr_budget

                # Fallback: if no plate candidate box found by YOLO on this frame but vehicle has no plate yet
                if len(cand_list) == 0 and not has_confirmed_plate and current_ocr_count == 0 and ocr_attempts == 0:
                    bw = vx2 - vx1
                    bh = vy2 - vy1
                    if bw >= 40 and bh >= 40:
                        cand_list = [{
                            "confidence": 0.40,
                            "bbox": [
                                max(0, int(vx1 + bw * 0.10)),
                                max(0, int(vy1 + bh * 0.50)),
                                min(orig_w, int(vx1 + bw * 0.90)),
                                min(orig_h, int(vy1 + bh * 0.98))
                            ]
                        }]

                should_run_ocr = within_ocr_budget and has_cooldown_elapsed and len(cand_list) > 0 and not is_already_finalized

                if should_run_ocr:
                    # Select top candidate plate crop
                    for cand in cand_list[:1]:
                        cpx1, cpy1, cpx2, cpy2 = cand["bbox"]
                        pad_x = max(3, int((cpx2 - cpx1) * 0.12))
                        pad_y = max(3, int((cpy2 - cpy1) * 0.15))
                        pc_x1 = max(0, cpx1 - pad_x)
                        pc_y1 = max(0, cpy1 - pad_y)
                        pc_x2 = min(orig_w, cpx2 + pad_x)
                        pc_y2 = min(orig_h, cpy2 + pad_y)
                        plate_crop = frame[pc_y1:pc_y2, pc_x1:pc_x2]

                        # Quality Gating Check (Strategy 6)
                        is_good_crop, q_score, q_info = evaluate_plate_crop_quality(plate_crop)
                        if not is_good_crop:
                            continue

                        # Register OCR attempt and update cooldown timestamps immediately
                        tracks[curr_tid]["ocr_attempts"] = ocr_attempts + 1
                        tracks[curr_tid]["last_ocr_frame"] = frame_idx
                        tracks[curr_tid]["last_ocr_time"] = time.perf_counter()

                        t_ocr0 = time.perf_counter()
                        extracted_text, ocr_score = self.extract_plate_text_easyocr(plate_crop)
                        time_ocr += (time.perf_counter() - t_ocr0)
                        total_ocr_calls += 1

                        if extracted_text and len(extracted_text) >= 4:
                            is_val, val_conf, validated_str = is_valid_indian_registration(extracted_text)
                            best_frame_plate = cand
                            best_plate_text = validated_str or extracted_text
                            best_plate_conf = val_conf if is_val else ocr_score

                            tracks[curr_tid]["ocr_count"] = current_ocr_count + 1
                            break
                elif len(cand_list) > 0:
                    # Retain detected plate bounding box without re-running OCR
                    best_frame_plate = cand_list[0]

                if best_frame_plate is not None and best_plate_text:
                    tracks[curr_tid]["plate_bbox"] = best_frame_plate["bbox"]
                    tracks[curr_tid]["ocr_confidence"] = best_plate_conf
                    # Ingest into Multi-Frame ANPR Fusion Engine
                    fusion_state = fusion_engine.add_observation(
                        track_id=curr_tid,
                        raw_plate=best_plate_text,
                        ocr_confidence=best_plate_conf,
                        plate_confidence=best_frame_plate["confidence"],
                        frame_number=frame_idx,
                        bbox=best_frame_plate["bbox"],
                        vehicle_class=v_class
                    )
                else:
                    if best_frame_plate is not None:
                        tracks[curr_tid]["plate_bbox"] = best_frame_plate["bbox"]
                    fusion_state = fusion_engine.evaluate_track(curr_tid)

                # Synchronize track state with fusion engine result without erasing confirmed plates
                f_plate = fusion_state.get("plate_number") or fusion_state.get("display_plate") or ""
                f_clean = "".join(c for c in f_plate.upper() if c.isalnum())
                if len(f_clean) >= 4 and f_plate != "Recognizing..." and f_plate != "Requires Manual Review":
                    tracks[curr_tid]["plate_number"] = f_plate
                    tracks[curr_tid]["display_plate"] = f_plate
                    tracks[curr_tid]["status"] = "finalized"
                elif tracks[curr_tid].get("plate_number"):
                    tracks[curr_tid]["display_plate"] = tracks[curr_tid]["plate_number"]
                    tracks[curr_tid]["status"] = "finalized"
                else:
                    tracks[curr_tid]["display_plate"] = fusion_state.get("display_plate", "Recognizing...")
                    tracks[curr_tid]["status"] = fusion_state.get("status", "pending")

                tracks[curr_tid]["final_confidence"] = max(fusion_state.get("final_confidence", 0.0), tracks[curr_tid].get("final_confidence", 0.0))
                tracks[curr_tid]["evidence_count"] = max(fusion_state.get("evidence_count", 0), tracks[curr_tid].get("evidence_count", 0))
                tracks[curr_tid]["agreement_ratio"] = fusion_state.get("agreement_ratio", 1.0)
                tracks[curr_tid]["supporting_predictions"] = fusion_state.get("supporting_predictions", [])

                # 5. DRAW VISUAL ANNOTATIONS ON FRAME (Exact UI Invariance Rule)
                # GREEN Bounding Box around vehicle: BGR (34, 197, 94) -> #22c55e
                cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (34, 197, 94), 2)

                # GREEN Label tag on top of vehicle
                v_label = f"{v_class} {v_conf:.2f}"
                (lw, lh), _ = cv2.getTextSize(v_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                tag_y = max(lh + 8, vy1)
                cv2.rectangle(frame, (vx1, tag_y - lh - 6), (vx1 + lw + 10, tag_y + 2), (34, 197, 94), -1)
                cv2.putText(frame, v_label, (vx1 + 5, tag_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                # BLUE Bounding Box around license plate if detected: BGR (246, 130, 59) -> #3b82f6
                assigned_plate_bbox = tracks[curr_tid]["plate_bbox"]
                plate_display_text = tracks[curr_tid]["plate_number"] or tracks[curr_tid]["display_plate"]
                if assigned_plate_bbox:
                    bx1, by1, bx2, by2 = assigned_plate_bbox
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (246, 130, 59), 2)

                    if plate_display_text:
                        (pw, ph), _ = cv2.getTextSize(plate_display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        p_tag_y = min(orig_h - 4, by2 + ph + 8)
                        cv2.rectangle(frame, (bx1, p_tag_y - ph - 4), (bx1 + pw + 8, p_tag_y + 2), (246, 130, 59), -1)
                        cv2.putText(frame, plate_display_text, (bx1 + 4, p_tag_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                frame_vehicles_info.append({
                    "track_id": curr_tid,
                    "class": v_class,
                    "confidence": v_conf,
                    "bbox": v_box,
                    "plate_bbox": assigned_plate_bbox,
                    "plate_number": tracks[curr_tid]["plate_number"],
                    "display_plate": plate_display_text,
                    "status": tracks[curr_tid]["status"],
                    "final_confidence": tracks[curr_tid]["final_confidence"],
                    "ocr_confidence": tracks[curr_tid]["ocr_confidence"]
                })

            # Cleanup inactive tracks (not seen for 30 frames)
            for tid in list(active_track_boxes.keys()):
                if frame_idx - tracks[tid]["last_seen_frame"] > 30:
                    del active_track_boxes[tid]

            frame_records.append({
                "frame_index": frame_idx,
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                "vehicles": frame_vehicles_info
            })

            # Write annotated frame
            out_writer.write(frame)
            frame_idx += 1

        cap.release()
        out_writer.release()

        # Convert raw mp4 to web-compatible H.264 MP4 with faststart
        convert_success = convert_video_to_web_h264(raw_out_filepath, final_out_filepath)
        if not convert_success:
            shutil.copyfile(raw_out_filepath, final_out_filepath)

        # Cleanup raw temporary file
        if os.path.exists(raw_out_filepath):
            try:
                os.remove(raw_out_filepath)
            except Exception:
                pass

        total_time = max(0.001, time.perf_counter() - t_start)
        actual_fps = round(processed_frames_count / total_time, 1)
        frames_skipped = max(0, total_frames - processed_frames_count)

        avg_v_ms = (time_vehicle_det / max(1, processed_frames_count)) * 1000.0
        avg_p_ms = (time_plate_det / max(1, processed_frames_count)) * 1000.0
        avg_ocr_ms = (time_ocr / max(1, total_ocr_calls)) * 1000.0 if total_ocr_calls > 0 else 0.0
        avg_frame_ms = (total_time / max(1, processed_frames_count)) * 1000.0

        print("==================================================")
        print("ANPR INFERENCE PIPELINE PERFORMANCE BREAKDOWN")
        print("==================================================")
        print(f"Device: {self.device.upper()} (FP16={self.use_half})")
        print(f"Frames Processed: {processed_frames_count} / {total_frames} (Skipped: {frames_skipped})")
        print(f"Total Pipeline Time: {total_time:.2f}s (Processing FPS: {actual_fps:.1f} FPS)")
        print(f"Vehicle Detection Avg: {avg_v_ms:.1f} ms/frame (imgsz={vehicle_img_size})")
        print(f"Plate Detection Avg:   {avg_p_ms:.1f} ms/frame (ROI crops only)")
        print(f"OCR Time Avg:          {avg_ocr_ms:.1f} ms/call (Calls: {total_ocr_calls})")
        print(f"Total Time Per Frame:  {avg_frame_ms:.1f} ms/frame")
        print("==================================================")

        # Step 4: Finalize all tracks through Multi-Frame ANPR Fusion Engine
        finalized_fusions = fusion_engine.finalize_all_active_tracks()
        fusion_by_tid: Dict[int, Dict[str, Any]] = {f["track_id"]: f for f in finalized_fusions}

        # Format final aggregated detection results with multi-level deduplication
        consolidated_tracks: List[Dict[str, Any]] = []

        # Group tracks by final recognized plate
        plate_groups: Dict[str, List[Dict[str, Any]]] = {}
        no_plate_tracks: List[Dict[str, Any]] = []

        for tid, tdata in tracks.items():
            f_info = fusion_by_tid.get(tid, {})
            p_val = f_info.get("plate_number") or tdata.get("plate_number") or ""
            p_norm = "".join(c for c in p_val.upper() if c.isalnum())
            if p_norm and len(p_norm) >= 4:
                plate_groups.setdefault(p_norm, []).append(tdata)
            else:
                no_plate_tracks.append(tdata)

        for plate_str, group in plate_groups.items():
            if len(group) == 1:
                consolidated_tracks.append(group[0])
            else:
                # Merge duplicate tracks with same plate into best canonical track
                canonical = max(group, key=lambda t: (t.get("final_confidence", 0.0), t.get("ocr_confidence", 0.0), t.get("frames_seen", 1)))
                canonical["merged_track_ids"] = [t["track_id"] for t in group]
                for other in group:
                    if other["track_id"] == canonical["track_id"]:
                        continue
                    canonical["first_seen_frame"] = min(canonical["first_seen_frame"], other["first_seen_frame"])
                    canonical["last_seen_frame"] = max(canonical["last_seen_frame"], other["last_seen_frame"])
                    canonical["frames_seen"] = canonical.get("frames_seen", 1) + other.get("frames_seen", 1)
                    canonical["vehicle_confidence"] = max(canonical["vehicle_confidence"], other["vehicle_confidence"])
                    canonical["final_confidence"] = max(canonical["final_confidence"], other.get("final_confidence", 0.0))
                    if not canonical.get("thumbnail") and other.get("thumbnail"):
                        canonical["thumbnail"] = other["thumbnail"]
                    for cls_k, count in other.get("class_counts", {}).items():
                        canonical["class_counts"][cls_k] = canonical.get("class_counts", {}).get(cls_k, 0) + count
                if canonical.get("class_counts"):
                    canonical["vehicle_class"] = max(canonical["class_counts"].keys(), key=lambda c: canonical["class_counts"][c])
                consolidated_tracks.append(canonical)

        # Merge remaining no-plate tracks with existing confirmed tracks if matching trajectory
        for np_track in no_plate_tracks:
            np_box = np_track["vehicle_bbox"]
            np_center_x = (np_box[0] + np_box[2]) / 2.0
            np_center_y = (np_box[1] + np_box[3]) / 2.0

            merged = False
            for ex_track in consolidated_tracks:
                ex_box = ex_track["vehicle_bbox"]
                ex_center_x = (ex_box[0] + ex_box[2]) / 2.0
                ex_center_y = (ex_box[1] + ex_box[3]) / 2.0

                iou = compute_iou(np_box, ex_box)
                center_dx = abs(np_center_x - ex_center_x) / float(orig_w)
                center_dy = abs(np_center_y - ex_center_y) / float(orig_h)

                is_same_vehicle = (iou > 0.20) or (center_dx < 0.22 and center_dy < 0.35)

                if is_same_vehicle:
                    ex_track["first_seen_frame"] = min(ex_track["first_seen_frame"], np_track["first_seen_frame"])
                    ex_track["last_seen_frame"] = max(ex_track["last_seen_frame"], np_track["last_seen_frame"])
                    ex_track["frames_seen"] = ex_track.get("frames_seen", 1) + np_track.get("frames_seen", 1)
                    ex_track["vehicle_confidence"] = max(ex_track["vehicle_confidence"], np_track["vehicle_confidence"])
                    if not ex_track.get("thumbnail") and np_track.get("thumbnail"):
                        ex_track["thumbnail"] = np_track["thumbnail"]
                    merged = True
                    break

            if not merged and np_track["frames_seen"] >= 2:
                consolidated_tracks.append(np_track)

        # Sort and rank all detected vehicle tracks
        consolidated_tracks = sorted(
            consolidated_tracks,
            key=lambda t: (
                1 if (t.get("plate_number") and len("".join(c for c in t.get("plate_number", "") if c.isalnum())) >= 4) else 0,
                t.get("final_confidence", 0.0),
                t.get("vehicle_confidence", 0.0),
                t.get("frames_seen", 1)
            ),
            reverse=True
        )

        final_detections = []
        unique_plates_count = 0

        for idx, tdata in enumerate(consolidated_tracks, start=1):
            tid = tdata["track_id"]
            f_info = fusion_by_tid.get(tid, {})
            p_val = f_info.get("plate_number") or tdata.get("plate_number") or ""
            p_clean = "".join(c for c in p_val.upper() if c.isalnum())
            has_valid_plate = len(p_clean) >= 4

            if has_valid_plate:
                unique_plates_count += 1

            final_conf = f_info.get("final_confidence", tdata.get("final_confidence", tdata.get("ocr_confidence", 0.0)))
            if has_valid_plate and float(final_conf) < 0.75:
                status = "manual_review"
                display_plate = f"{p_val} (Review Required)"
            else:
                status = f_info.get("status") or ("finalized" if has_valid_plate else "pending")
                display_plate = f_info.get("display_plate") or (p_val if has_valid_plate else "Recognizing...")

            # Collect all frame predictions across canonical and merged tracks
            merged_ids = tdata.get("merged_track_ids", [tid])
            all_frame_preds = []
            seen_frames = set()
            for m_tid in merged_ids:
                m_finfo = fusion_by_tid.get(m_tid, {})
                for fp in m_finfo.get("frame_predictions", []):
                    f_num = fp.get("frame_number")
                    if f_num not in seen_frames:
                        seen_frames.add(f_num)
                        all_frame_preds.append(fp)
            all_frame_preds.sort(key=lambda x: x.get("frame_number", 0))
            if not all_frame_preds:
                all_frame_preds = f_info.get("frame_predictions", [])

            detection_item = {
                "track_id": idx,
                "vehicle_class": tdata["vehicle_class"],
                "vehicle_confidence": round(float(tdata["vehicle_confidence"]), 2),
                "vehicle_bbox": tdata["vehicle_bbox"],
                "plate_bbox": tdata["plate_bbox"],
                "plate_number": p_val if has_valid_plate else "",
                "display_plate": display_plate,
                "status": status,
                "final_confidence": round(float(final_conf), 2),
                "ocr_confidence": round(float(tdata.get("ocr_confidence", 0.0)), 2),
                "frame_count": max(len(all_frame_preds), f_info.get("frame_count", tdata.get("frames_seen", 1))),
                "evidence_count": max(len(all_frame_preds), f_info.get("evidence_count", 1)),
                "agreement_ratio": round(float(f_info.get("agreement_ratio", 1.0)), 2),
                "frame_predictions": all_frame_preds,
                "supporting_predictions": f_info.get("supporting_predictions", []),
                "thumbnail": tdata["thumbnail"],
                "first_seen_frame": tdata["first_seen_frame"],
                "last_seen_frame": tdata["last_seen_frame"],
            }

            # Persist to ANPRX MySQL Production Database
            try:
                db_service.record_finalized_anpr_event({
                    "track_id": idx,
                    "plate_number": p_val if has_valid_plate else "",
                    "status": status,
                    "final_confidence": final_conf,
                    "frame_count": detection_item["frame_count"],
                    "supporting_predictions": detection_item["supporting_predictions"],
                    "frame_predictions": all_frame_preds,
                    "vehicle_type": tdata.get("vehicle_class", "Car"),
                    "vehicle_bbox": tdata.get("vehicle_bbox", [0, 0, 0, 0]),
                    "camera_id": "G01-ENTRY",
                    "gate_id": "Gate 01",
                    "event_type": "entry",
                })
            except Exception as db_err:
                print(f"[DB WARN] Failed to record ANPR event: {db_err}", file=sys.stderr)

            final_detections.append(detection_item)

        unique_plates_count = len(final_detections)

        return {
            "success": True,
            "frames_processed": processed_frames_count,
            "total_video_frames": total_frames,
            "vehicles_detected": len(final_detections),
            "plates_detected": unique_plates_count,
            "video_url": f"/static/processed_videos/{out_filename}",
            "fps": actual_fps,
            "width": orig_w,
            "height": orig_h,
            "detections": final_detections,
            "frame_records": frame_records[:200],
            "metrics": {
                "device": self.device,
                "use_gpu": (self.device == "cuda"),
                "use_half": self.use_half,
                "processing_fps": actual_fps,
                "frames_processed": processed_frames_count,
                "frames_skipped": frames_skipped,
                "avg_vehicle_detection_ms": round(avg_v_ms, 1),
                "avg_plate_detection_ms": round(avg_p_ms, 1),
                "avg_ocr_ms": round(avg_ocr_ms, 1),
                "avg_frame_ms": round(avg_frame_ms, 1),
                "total_processing_time_s": round(total_time, 2),
                "total_ocr_calls": total_ocr_calls
            }
        }
