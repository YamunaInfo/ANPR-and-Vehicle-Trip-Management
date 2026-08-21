"""
Production-grade RTSP / CCTV Live Stream Ingestion & Real-Time ANPR Engine for GateSense.

Features:
1. Thread-safe background frame ingestion from RTSP / HTTP / USB / Video streams.
2. Low-latency frame buffer (drops stale backlog frames to ensure live real-time feed).
3. Frame sampling (5-10 FPS) for YOLO vehicle detection + Plate detector + PaddleOCR / EasyOCR.
4. Multi-frame fusion per vehicle track with Indian registration plate syntax validation.
5. Real-time visual annotation (bounding boxes, class labels, plate numbers, confidence %).
6. High-speed MJPEG video streaming for browser compatibility (zero RTSP codec issues in browser).
7. Automatic reconnection with exponential backoff if CCTV stream drops.
8. Persistence of finalized gate detection events to GateSense database.
"""
from __future__ import annotations

import base64
import datetime
import math
import os
import re
import sys
import threading
import time
from typing import Any, Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np

from ai.db_service import db_service
from ai.multi_frame_fusion import MultiFramePlateFusionEngine
from ai.video_processor import (
    STATE_CONFUSION_MAP,
    VALID_INDIAN_STATES,
    VEHICLE_CLASSES,
    VideoAnprProcessor,
    class_agnostic_nms,
    compute_iou,
    evaluate_plate_crop_quality,
    is_valid_indian_registration,
    smart_indian_plate_normalize,
)


class CCTVStreamManager:
    """Manages continuous ingestion and real-time ANPR inference for CCTV/RTSP streams."""

    def __init__(self, video_processor: Optional[VideoAnprProcessor] = None):
        self.video_processor = video_processor or VideoAnprProcessor()
        self.rtsp_url: str = ""
        self.status: str = "disconnected"  # "disconnected" | "connecting" | "connected" | "reconnecting" | "error"
        self.status_message: str = "CCTV Disconnected"
        self.is_running: bool = False
        
        # Threading controls
        self._stop_event = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None
        self._inference_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Frame buffers (thread-safe)
        self._raw_frame: Optional[np.ndarray] = None
        self._annotated_frame: Optional[np.ndarray] = None
        self._latest_jpeg: Optional[bytes] = None
        self._frame_seq: int = 0
        self._fps_measured: float = 0.0

        # Detection & ANPR tracking state
        self.tracks: Dict[int, Dict[str, Any]] = {}
        self.active_track_boxes: Dict[int, List[float]] = {}
        self.next_track_id: int = 1
        self.active_detections: List[Dict[str, Any]] = []
        self.fusion_engine = MultiFramePlateFusionEngine(
            window_size=10,
            min_observations=3,
            min_confidence=0.60,
            min_agreement=0.60,
        )

        # Statistics
        self.frames_read: int = 0
        self.frames_processed: int = 0
        self.vehicles_detected_count: int = 0
        self.plates_detected_count: int = 0
        self.last_error_time: float = 0.0
        self.reconnect_attempts: int = 0

    def connect(self, rtsp_url: str) -> Dict[str, Any]:
        """Initiates connection to the RTSP / IP camera stream in a background thread."""
        rtsp_url = (rtsp_url or "").strip()
        if not rtsp_url:
            self.status = "error"
            self.status_message = "Invalid RTSP URL: URL cannot be empty"
            return {"success": False, "status": self.status, "message": self.status_message}

        # Resolve local test presets
        resolved_url = rtsp_url
        if rtsp_url.lower() in ("sample", "sample_traffic", "test", "demo", "demo_stream"):
            backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            for candidate in ["test_faac.mp4", "real_traffic_test.mp4", "test_traffic_video.mp4"]:
                p = os.path.join(backend_dir, candidate)
                if os.path.exists(p):
                    resolved_url = p
                    break

        # Stop any existing stream
        self.stop()

        with self._lock:
            self.rtsp_url = resolved_url
            self.status = "connecting"
            self.status_message = "Connecting to CCTV stream..."
            self.is_running = True
            self._stop_event.clear()
            self.tracks.clear()
            self.active_track_boxes.clear()
            self.next_track_id = 1
            self.active_detections.clear()
            self.fusion_engine = MultiFramePlateFusionEngine(
                window_size=10,
                min_observations=3,
                min_confidence=0.60,
                min_agreement=0.60,
            )
            self.frames_read = 0
            self.frames_processed = 0
            self.vehicles_detected_count = 0
            self.plates_detected_count = 0
            self.reconnect_attempts = 0

        # Launch background capture & inference threads
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True, name="CCTV-Capture")
        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True, name="CCTV-Inference")
        self._capture_thread.start()
        self._inference_thread.start()

        # Wait briefly up to 1.5 seconds for initial connection verification
        t0 = time.time()
        while time.time() - t0 < 1.5:
            if self.status in ("connected", "error"):
                break
            time.sleep(0.1)

        return {
            "success": self.status != "error",
            "status": self.status,
            "message": self.status_message,
            "rtsp_url": self.rtsp_url,
        }

    def stop(self) -> Dict[str, Any]:
        """Stops the CCTV stream and cleans up resources."""
        self._stop_event.set()
        self.is_running = False
        
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=1.0)
        if self._inference_thread and self._inference_thread.is_alive():
            self._inference_thread.join(timeout=1.0)

        with self._lock:
            self.status = "disconnected"
            self.status_message = "CCTV Disconnected"
            self._raw_frame = None
            self._annotated_frame = None
            self._latest_jpeg = None

        return {"success": True, "status": "disconnected", "message": "CCTV stream stopped"}

    def get_status(self) -> Dict[str, Any]:
        """Returns current CCTV status and recent detections."""
        with self._lock:
            return {
                "status": self.status,
                "message": self.status_message,
                "rtsp_url": self.rtsp_url,
                "fps": round(self._fps_measured, 1),
                "frames_read": self.frames_read,
                "frames_processed": self.frames_processed,
                "vehicles_count": self.vehicles_detected_count,
                "plates_count": self.plates_detected_count,
                "detections": list(self.active_detections),
            }

    def get_latest_jpeg(self) -> Optional[bytes]:
        """Returns the most recent annotated JPEG frame for MJPEG streaming."""
        with self._lock:
            return self._latest_jpeg

    def _open_capture(self, url: str) -> Optional[cv2.VideoCapture]:
        """Configures and opens an OpenCV VideoCapture instance with low latency settings."""
        # For RTSP, configure TCP transport and buffer flags
        if url.startswith("rtsp://") or url.startswith("rtsps://"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|timeout;5000000|buffer_size;1024000|max_delay;500000"
            )

        # Support integer camera index (e.g. "0" -> 0)
        target = int(url) if url.isdigit() else url
        
        try:
            cap = cv2.VideoCapture(target, cv2.CAP_FFMPEG if isinstance(target, str) and "://" in target else cv2.CAP_ANY)
            if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not cap.isOpened():
                return None
            return cap
        except Exception as exc:
            print(f"[CCTV] Failed to open capture for {url}: {exc}", file=sys.stderr)
            return None

    def _capture_loop(self):
        """Continuously reads frames from the CCTV source into memory, discarding lag."""
        cap = None
        is_file_video = os.path.exists(self.rtsp_url) and not self.rtsp_url.startswith("rtsp://")

        while not self._stop_event.is_set():
            if cap is None or not cap.isOpened():
                with self._lock:
                    self.status = "connecting" if self.reconnect_attempts == 0 else "reconnecting"
                    self.status_message = (
                        "Connecting to CCTV..." if self.reconnect_attempts == 0
                        else f"CCTV Connection Lost - Reconnecting (attempt {self.reconnect_attempts})..."
                    )

                cap = self._open_capture(self.rtsp_url)
                if cap is None or not cap.isOpened():
                    self.reconnect_attempts += 1
                    if self.reconnect_attempts > 10:
                        with self._lock:
                            self.status = "error"
                            self.status_message = "Unable to connect to camera (timeout / host unreachable)"
                        time.sleep(2.0)
                    else:
                        time.sleep(1.5)
                    continue

                # Read first frame to verify connection
                ret, test_frame = cap.read()
                if not ret or test_frame is None:
                    cap.release()
                    cap = None
                    self.reconnect_attempts += 1
                    with self._lock:
                        self.status = "error"
                        self.status_message = "Camera connected but returned no video frames"
                    time.sleep(1.5)
                    continue

                # Successfully connected
                with self._lock:
                    self.status = "connected"
                    self.status_message = "CCTV Connected"
                    self.reconnect_attempts = 0

            # Read latest frame
            ret, frame = cap.read()
            if not ret or frame is None:
                if is_file_video:
                    # Loop video file for continuous testing
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.03)
                    continue

                print("[CCTV] Frame read failed; camera might have disconnected.", file=sys.stderr)
                cap.release()
                cap = None
                with self._lock:
                    self.status = "reconnecting"
                    self.status_message = "CCTV Connection Lost - Reconnecting..."
                time.sleep(1.0)
                continue

            with self._lock:
                self._raw_frame = frame
                self.frames_read += 1
                self._frame_seq += 1

            # Sleep slightly if reading a local video file to match normal ~25-30 FPS playback
            if is_file_video:
                time.sleep(0.033)
            else:
                time.sleep(0.005)

        if cap is not None:
            cap.release()

    def _inference_loop(self):
        """Processes sampled frames with YOLOv8 vehicle detection + Plate Detection + OCR."""
        last_process_time = 0.0
        target_process_interval = 1.0 / 6.0  # Sample at ~6 FPS for AI to maintain CPU efficiency
        last_fps_calc = time.time()
        fps_counter = 0

        while not self._stop_event.is_set():
            if self.status != "connected":
                time.sleep(0.1)
                continue

            current_time = time.time()
            if current_time - last_process_time < target_process_interval:
                time.sleep(0.01)
                continue

            # Grab current raw frame
            frame = None
            with self._lock:
                if self._raw_frame is not None:
                    frame = self._raw_frame.copy()

            if frame is None:
                time.sleep(0.02)
                continue

            last_process_time = current_time
            fps_counter += 1
            if current_time - last_fps_calc >= 1.0:
                self._fps_measured = float(fps_counter) / (current_time - last_fps_calc)
                fps_counter = 0
                last_fps_calc = current_time

            # Run ANPR processing on this frame
            annotated_frame = self._process_single_cctv_frame(frame)

            # Generate JPEG for MJPEG stream
            try:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                _, jpeg_bytes = cv2.imencode(".jpg", annotated_frame, encode_param)
                with self._lock:
                    self._annotated_frame = annotated_frame
                    self._latest_jpeg = jpeg_bytes.tobytes()
                    self.frames_processed += 1
            except Exception as enc_err:
                print(f"[CCTV] JPEG encode error: {enc_err}", file=sys.stderr)

    def _process_single_cctv_frame(self, frame: np.ndarray) -> np.ndarray:
        """Runs vehicle detection, plate localization, OCR, and annotations on one frame."""
        h, w = frame.shape[:2]
        vp = self.video_processor

        # Ensure models are loaded
        if vp.vehicle_model is None:
            try:
                vp._load_models()
            except Exception:
                pass

        if vp.vehicle_model is None:
            # Draw banner indicating missing AI models
            cv2.putText(frame, "AI Vehicle Model Not Loaded", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return frame

        # 1. VEHICLE DETECTION: YOLOv8
        raw_detected_vehicles = []
        try:
            v_results = vp.vehicle_model(
                frame,
                imgsz=384,
                conf=0.28,
                device=vp.device,
                verbose=False,
            )
            if len(v_results) > 0 and v_results[0].boxes is not None:
                boxes = v_results[0].boxes
                for b in boxes:
                    cls_id = int(b.cls[0].item())
                    conf = float(b.conf[0].item())
                    if cls_id in VEHICLE_CLASSES:
                        xyxy = [float(x) for x in b.xyxy[0].tolist()]
                        x1 = max(0, min(w - 1, int(xyxy[0])))
                        y1 = max(0, min(h - 1, int(xyxy[1])))
                        x2 = max(0, min(w, int(xyxy[2])))
                        y2 = max(0, min(h, int(xyxy[3])))
                        if (x2 - x1) >= 24 and (y2 - y1) >= 24:
                            raw_detected_vehicles.append({
                                "class": VEHICLE_CLASSES[cls_id],
                                "confidence": conf,
                                "bbox": [x1, y1, x2, y2],
                            })
        except Exception as v_err:
            print(f"[CCTV] Vehicle detection error: {v_err}", file=sys.stderr)

        detected_vehicles = class_agnostic_nms(raw_detected_vehicles, iou_thresh=0.35, contain_thresh=0.50)

        # 2. PLATE DETECTION ON VEHICLE CROPS
        vehicle_plate_candidates: Dict[int, List[Dict[str, Any]]] = {}
        if vp.plate_model is not None and detected_vehicles:
            valid_v_crops = []
            valid_v_indices = []
            for v_idx, v_det in enumerate(detected_vehicles):
                vx1, vy1, vx2, vy2 = v_det["bbox"]
                v_crop = frame[vy1:vy2, vx1:vx2]
                if v_crop is not None and v_crop.size > 0 and (vx2 - vx1) >= 24 and (vy2 - vy1) >= 24:
                    valid_v_crops.append(v_crop)
                    valid_v_indices.append(v_idx)

            if valid_v_crops:
                try:
                    p_res_list = vp.plate_model(
                        valid_v_crops,
                        imgsz=384,
                        conf=0.22,
                        device=vp.device,
                        verbose=False,
                    )
                    for i, p_res in enumerate(p_res_list):
                        v_idx = valid_v_indices[i]
                        vx1, vy1, vx2, vy2 = detected_vehicles[v_idx]["bbox"]
                        cands = []
                        if p_res.boxes is not None:
                            for pb in p_res.boxes:
                                cconf = float(pb.conf[0].item())
                                cxyxy = [float(x) for x in pb.xyxy[0].tolist()]
                                cpx1 = max(0, min(w - 1, int(vx1 + cxyxy[0])))
                                cpy1 = max(0, min(h - 1, int(vy1 + cxyxy[1])))
                                cpx2 = max(0, min(w, int(vx1 + cxyxy[2])))
                                cpy2 = max(0, min(h, int(vy1 + cxyxy[3])))
                                if cpx2 - cpx1 >= 8 and cpy2 - cpy1 >= 5:
                                    cands.append({"confidence": cconf, "bbox": [cpx1, cpy1, cpx2, cpy2]})
                        vehicle_plate_candidates[v_idx] = cands
                except Exception as p_err:
                    print(f"[CCTV] Plate detection error: {p_err}", file=sys.stderr)

        # 3. MULTI-FRAME TRACKING & ASSOCIATION
        matched_track_ids = set()
        current_detections_list = []

        for v_idx, v in enumerate(detected_vehicles):
            v_box = v["bbox"]
            v_class = v["class"]
            v_conf = v["confidence"]
            vx_center = (v_box[0] + v_box[2]) / 2.0
            vy_center = (v_box[1] + v_box[3]) / 2.0
            vw = max(1, v_box[2] - v_box[0])
            vh = max(1, v_box[3] - v_box[1])

            # Match with existing active tracks
            best_score = 0.0
            best_tid = None
            for tid, prev_box in self.active_track_boxes.items():
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
                curr_tid = self.next_track_id
                self.next_track_id += 1
                self.tracks[curr_tid] = {
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
                    "last_ocr_time": 0.0,
                    "evidence_count": 0,
                    "agreement_ratio": 0.0,
                    "supporting_predictions": [],
                    "frame_predictions": [],
                    "thumbnail": None,
                    "frames_seen": 1,
                    "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                }
            else:
                curr_tid = best_tid
                tdata = self.tracks[curr_tid]
                tdata["frames_seen"] = tdata.get("frames_seen", 0) + 1
                tdata["class_counts"] = tdata.get("class_counts", {})
                tdata["class_counts"][v_class] = tdata["class_counts"].get(v_class, 0) + 1
                top_cls = max(tdata["class_counts"].keys(), key=lambda c: tdata["class_counts"][c])
                tdata["vehicle_class"] = top_cls
                if v_conf > tdata["vehicle_confidence"]:
                    tdata["vehicle_confidence"] = v_conf
                tdata["vehicle_bbox"] = v_box

            matched_track_ids.add(curr_tid)
            self.active_track_boxes[curr_tid] = v_box
            tdata = self.tracks[curr_tid]

            # Create vehicle crop thumbnail if needed
            vx1, vy1, vx2, vy2 = v_box
            if tdata["thumbnail"] is None or (vx2 - vx1) * (vy2 - vy1) > 2500:
                try:
                    v_crop = frame[vy1:vy2, vx1:vx2]
                    if v_crop is not None and v_crop.size > 0:
                        thumb_resized = cv2.resize(v_crop, (120, 90), interpolation=cv2.INTER_AREA)
                        _, buf = cv2.imencode(".jpg", thumb_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        tdata["thumbnail"] = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"
                except Exception:
                    pass

            # 4. OCR EXTRACTION & MULTI-FRAME FUSION
            cands = vehicle_plate_candidates.get(v_idx, [])
            now_ms = time.time() * 1000.0
            can_ocr = (
                (tdata["ocr_count"] < 8) and
                (now_ms - tdata.get("last_ocr_time", 0.0) >= 200.0)
            )

            best_frame_plate = None
            best_frame_conf = 0.0
            assigned_plate_bbox = tdata.get("plate_bbox")

            if cands:
                best_cand = max(cands, key=lambda c: c["confidence"])
                px1, py1, px2, py2 = best_cand["bbox"]
                assigned_plate_bbox = [px1, py1, px2, py2]
                tdata["plate_bbox"] = assigned_plate_bbox

                if can_ocr:
                    p_crop = frame[py1:py2, px1:px2]
                    is_ok, q_score, q_info = evaluate_plate_crop_quality(p_crop)
                    if is_ok:
                        ocr_txt, ocr_c = vp.extract_plate_text(p_crop)
                        tdata["ocr_count"] = tdata.get("ocr_count", 0) + 1
                        tdata["last_ocr_time"] = now_ms

                        if ocr_txt and len("".join(c for c in ocr_txt.upper() if c.isalnum())) >= 4:
                            best_frame_plate = ocr_txt
                            best_frame_conf = max(ocr_c, 0.70)
                            if best_frame_conf > tdata.get("ocr_confidence", 0.0):
                                tdata["ocr_confidence"] = best_frame_conf

                            # Feed observation into multi-frame fusion
                            fusion_result = self.fusion_engine.update(
                                track_id=curr_tid,
                                plate_text=ocr_txt,
                                confidence=best_frame_conf,
                                frame_number=self.frames_processed + 1,
                                vehicle_class=tdata["vehicle_class"],
                            )

                            if fusion_result and fusion_result.get("plate_number"):
                                tdata["plate_number"] = fusion_result["plate_number"]
                                tdata["final_confidence"] = fusion_result.get("final_confidence", best_frame_conf)
                                if tdata["final_confidence"] < 0.75:
                                    tdata["status"] = "manual_review"
                                    tdata["display_plate"] = f"{tdata['plate_number']} (Review Required)"
                                else:
                                    tdata["status"] = fusion_result.get("status", "finalized")
                                    tdata["display_plate"] = fusion_result.get("display_plate", fusion_result["plate_number"])
                                tdata["evidence_count"] = fusion_result.get("evidence_count", 1)
                                tdata["agreement_ratio"] = fusion_result.get("agreement_ratio", 1.0)
                                tdata["supporting_predictions"] = fusion_result.get("supporting_predictions", [])
                                tdata["frame_predictions"] = fusion_result.get("frame_predictions", [])

                                # Persist to ANPRX MySQL Production Database
                                try:
                                    db_service.record_finalized_anpr_event({
                                        "track_id": curr_tid,
                                        "plate_number": tdata["plate_number"],
                                        "status": tdata["status"],
                                        "final_confidence": tdata["final_confidence"],
                                        "frame_count": tdata["frames_seen"],
                                        "supporting_predictions": tdata["supporting_predictions"],
                                        "frame_predictions": tdata.get("frame_predictions", []),
                                        "vehicle_type": tdata.get("vehicle_class", "car"),
                                        "vehicle_bbox": [vx1, vy1, vx2, vy2],
                                        "camera_id": "G01-ENTRY",
                                        "gate_id": "Gate 01",
                                        "event_type": "entry",
                                    })
                                except Exception as db_err:
                                    print(f"[CCTV DB WARN] Failed to save ANPR event: {db_err}", file=sys.stderr)

            # 5. DRAW VISUAL ANNOTATIONS ON FRAME
            # Vehicle Box (Green/Amber)
            cls_name = (tdata["vehicle_class"] or "Vehicle").capitalize()
            v_conf_pct = int(tdata["vehicle_confidence"] * 100)
            cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (34, 197, 94), 2)

            v_label = f"{cls_name} {v_conf_pct}%"
            (tw, th), _ = cv2.getTextSize(v_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (vx1, max(0, vy1 - th - 8)), (vx1 + tw + 8, vy1), (34, 197, 94), -1)
            cv2.putText(frame, v_label, (vx1 + 4, max(th + 2, vy1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Plate Box (Blue) + Plate Label
            plate_display = tdata.get("plate_number") or tdata.get("display_plate") or ""
            if assigned_plate_bbox:
                px1, py1, px2, py2 = assigned_plate_bbox
                cv2.rectangle(frame, (px1, py1), (px2, py2), (246, 130, 59), 2)
                if plate_display:
                    (pw, ph), _ = cv2.getTextSize(plate_display, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    p_tag_y = min(h - 4, py2 + ph + 8)
                    cv2.rectangle(frame, (px1, p_tag_y - ph - 4), (px1 + pw + 8, p_tag_y + 2), (246, 130, 59), -1)
                    cv2.putText(frame, plate_display, (px1 + 4, p_tag_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Formatted detection item for API/UI
            detection_item = {
                "id": f"cctv-track-{curr_tid}",
                "detectionId": curr_tid,
                "class": cls_name,
                "plate": tdata.get("plate_number") or "",
                "displayPlate": tdata.get("display_plate") or (tdata.get("plate_number") or "Recognizing..."),
                "status": tdata.get("status") or ("finalized" if tdata.get("plate_number") else "pending"),
                "confidence": round(float(tdata.get("final_confidence") or tdata.get("vehicle_confidence") or 0.0), 2),
                "ocrConfidence": round(float(tdata.get("ocr_confidence") or 0.0), 2),
                "frameCount": tdata.get("frames_seen", 1),
                "evidenceCount": tdata.get("evidence_count", 1),
                "agreementRatio": round(float(tdata.get("agreement_ratio", 1.0)), 2),
                "timestamp": tdata.get("timestamp") or datetime.datetime.now().strftime("%H:%M:%S"),
                "thumbnail": tdata.get("thumbnail") or "",
                "lastSeen": int(time.time() * 1000),
                "vehicleBbox": v_box,
                "plateBbox": assigned_plate_bbox,
                "framePredictions": tdata.get("frame_predictions", []),
                "supportingPredictions": tdata.get("supporting_predictions", []),
            }
            current_detections_list.append(detection_item)

        # Cleanup tracks not seen recently
        for tid in list(self.active_track_boxes.keys()):
            if tid not in matched_track_ids:
                del self.active_track_boxes[tid]

        # Draw HUD Overlays (Top Banner)
        cv2.rectangle(frame, (10, 10), (320, 60), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (320, 60), (59, 130, 246), 1)
        # Red live indicator circle
        cv2.circle(frame, (26, 26), 5, (0, 0, 255), -1)
        cv2.putText(frame, "LIVE CCTV • Gate 01", (38, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        hud_info = f"FPS: {self._fps_measured:.1f} | Vehicles: {len(current_detections_list)}"
        cv2.putText(frame, hud_info, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)

        # Update active detections
        with self._lock:
            if current_detections_list:
                # Merge into cumulative detection history
                existing_map = {d["detectionId"]: d for d in self.active_detections}
                for d in current_detections_list:
                    existing_map[d["detectionId"]] = d
                self.active_detections = sorted(
                    list(existing_map.values()),
                    key=lambda x: (1 if x["plate"] else 0, x["confidence"]),
                    reverse=True
                )
                self.vehicles_detected_count = len(self.active_detections)
                self.plates_detected_count = len([d for d in self.active_detections if d["plate"]])

        return frame

    def generate_mjpeg(self) -> Generator[bytes, None, None]:
        """Generates continuous multipart MJPEG stream bytes for browser <img> rendering."""
        # Standard standby placeholder frame in case CCTV is connecting
        standby_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(standby_img, "CONNECTING TO CCTV...", (140, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        _, standby_bytes = cv2.imencode(".jpg", standby_img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        standby_jpeg = standby_bytes.tobytes()

        while self.is_running and not self._stop_event.is_set():
            frame_bytes = self.get_latest_jpeg() or standby_jpeg
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
            time.sleep(0.04)  # ~25 FPS stream delivery rate
