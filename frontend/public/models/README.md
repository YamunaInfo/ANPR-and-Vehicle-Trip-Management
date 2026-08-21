# GateSense ANPR — ONNX Model Directory

This directory stores the object detection models used by the Edge AI ANPR pipeline in the browser via `onnxruntime-web`.

---

## Required Model Files

### 1. `vehicle_detector.onnx`
- **Location:** `frontend/public/models/vehicle_detector.onnx`
- **Purpose:** Primary vehicle detection stage (detects car, motorcycle, bus, truck).
- **Architecture:** YOLOv8 / YOLO11 / YOLOv5 / YOLOv7 / YOLOv9 object detection model.
- **Input Tensor:** `[1, 3, 640, 640]` or `[1, 3, H, W]` float32 normalized image (0.0 to 1.0), RGB format.
- **Status:** Integrated & verified operational.

### 2. `plate_detector.onnx`
- **Location:** `frontend/public/models/plate_detector.onnx`
- **Purpose:** Second-stage License Plate Localizer (detects exact license plate bounding box inside vehicle crops).
- **Status:** Module implemented; expecting trained ONNX file at this exact location.

---

## `plate_detector.onnx` Specifications & Expected Formats

### Supported Output Architectures (Auto-Detected)
The pipeline dynamically inspects `inputMetadata` and `outputMetadata` at runtime to auto-detect and decode any standard YOLO output format:

1. **YOLOv8 / YOLO11 Format (`[1, 4+C, N]`)**
   - Output shape: `[1, 5, 8400]` (for single-class license plate detector with 640x640 input)
   - Channels: `[center_x, center_y, width, height, plate_confidence]`
   - Anchors: `8400` grid cells

2. **YOLOv5 / YOLOv7 Format (`[1, N, 5+C]`)**
   - Output shape: `[1, 25200, 6]` (for single-class license plate detector)
   - Columns: `[center_x, center_y, width, height, objectness_confidence, class_0_confidence]`

3. **YOLOv9 / RT-DETR / Exported End-to-End Format (`[1, N, 6]`)**
   - Output shape: `[1, 300, 6]`
   - Columns: `[x1, y1, x2, y2, confidence, class_id]`

### Input Requirements
- **Shape:** `[1, 3, 640, 640]` or `[1, 3, 320, 320]` or `[1, 3, 416, 416]` (Auto-detected from model ONNX input shape)
- **Data Type:** `Float32Array` (float32)
- **Color Format:** RGB, normalized to `[0.0, 1.0]` (pixel / 255.0)
- **Padding:** Standard YOLO letterbox padding (grey fill `114, 114, 114`)

---

## How to Place a New Model

1. Train or export your license plate detection YOLO model to ONNX:
   ```bash
   yolo export model=best_plate.pt format=onnx imgsz=640 dynamic=False
   ```
2. Rename the exported `.onnx` file to `plate_detector.onnx`.
3. Copy it into this directory:
   ```
   frontend/public/models/plate_detector.onnx
   ```
4. Open the GateSense Web Interface (`/detect` page).
5. Click **"Run Plate Model Test"** in the AI Model Diagnostics Panel to verify:
   - `Plate Model: LOADED`
   - `Inference: SUCCESS`
   - Real license plate bounding box coordinates mapped to CCTV frame.
