/**
 * YoloPlateDetector — Real ONNX-based license plate detector.
 *
 * Supports YOLO v5, v8, v9, v10 ONNX output formats.
 * Auto-detects input size and output layout from model metadata.
 * NO fallback detections. NO heuristics. Real model only.
 *
 * Model must be placed at: /models/plate_detector.onnx
 */
import { InferenceSession, Tensor, env } from 'onnxruntime-web';
import { IPlateDetector, DetectionResult, BoundingBox } from '../core/interfaces';

// ─── Public diagnostic types ────────────────────────────────────────────────

export interface PlateDiagnosticResult {
  modelLoaded: boolean;
  inferenceExecuted: boolean;
  httpStatus: number;
  fileSizeBytes: number;
  inputNames: string[];
  outputNames: string[];
  inputShape: number[];
  outputShape: number[];
  inputDataType: string;
  outputDataType: string;
  architecture: string;
  detectedModelInputSize: number;
  rawDetectionCount: number;
  postNmsDetectionCount: number;
  highestConfidence: number;
  plateBoundingBox: BoundingBox | null;
  rawDetections: DetectionResult[];
  postNmsDetections: DetectionResult[];
  error?: string;
}

// ─── Internal helpers ────────────────────────────────────────────────────────

interface PreprocessResult {
  inputTensor: Tensor;
  scale: number;
  padX: number;
  padY: number;
  modelInputSize: number;
}

interface DecodeResult {
  rawDetections: DetectionResult[];
  postNmsDetections: DetectionResult[];
  architecture: string;
}

// ─── Detector class ──────────────────────────────────────────────────────────

export class YoloPlateDetector implements IPlateDetector {
  private session: InferenceSession | null = null;
  private modelInputSize = 640; // Actual value read from model input shape

  // Public state exposed to diagnostic panel
  public httpStatus: number = 0;
  public fileSizeBytes: number = 0;
  public modelMetadata: {
    inputNames: string[];
    outputNames: string[];
    inputShape: number[];
    outputShape: number[];
    inputDataType: string;
    outputDataType: string;
    architecture: string;
    detectedModelInputSize: number;
  } | null = null;

  // ── Load ──────────────────────────────────────────────────────────────────

  async load(): Promise<void> {
    if (this.session) return;

    // 1. Verify HTTP availability and check file size
    try {
      const res = await fetch('/models/plate_detector.onnx', { method: 'HEAD' });
      this.httpStatus = res.status;
      const cl = res.headers.get('content-length');
      this.fileSizeBytes = cl ? parseInt(cl, 10) : 0;

      if (res.status !== 200) {
        throw new Error(`PLATE MODEL MISSING (HTTP ${res.status}): /models/plate_detector.onnx not found.`);
      }
      if (this.fileSizeBytes === 0) {
        throw new Error('PLATE MODEL MISSING: /models/plate_detector.onnx is 0 bytes (empty file).');
      }
    } catch (err: any) {
      if (err.message?.startsWith('PLATE MODEL MISSING')) throw err;
      throw new Error(`PLATE MODEL MISSING: Cannot fetch /models/plate_detector.onnx (${err.message})`);
    }

    // 2. Create ONNX InferenceSession
    try {
      env.wasm.wasmPaths = '/';
      env.wasm.numThreads = 1;

      this.session = await InferenceSession.create('/models/plate_detector.onnx', {
        executionProviders: ['wasm'],
      });

      // 3. Inspect model input/output to determine shape and architecture
      const inputMeta = this.session.inputNames.map(n => (this.session!.inputMetadata as Record<string, any>)[n]);
      const outputMeta = this.session.outputNames.map(n => (this.session!.outputMetadata as Record<string, any>)[n]);


      const inputShape = (inputMeta[0]?.shape ?? []) as number[];
      const outputShape = (outputMeta[0]?.shape ?? []) as number[];

      // Detect input size from model (commonly 640, 416, 320, etc.)
      if (inputShape.length === 4 && typeof inputShape[2] === 'number' && inputShape[2] > 0) {
        this.modelInputSize = inputShape[2]; // NCHW → H dimension
      }

      const architecture = this._detectArchitecture(outputShape);

      this.modelMetadata = {
        inputNames: [...this.session.inputNames],
        outputNames: [...this.session.outputNames],
        inputShape: inputShape.map(d => (typeof d === 'number' ? d : -1)),
        outputShape: outputShape.map(d => (typeof d === 'number' ? d : -1)),
        inputDataType: inputMeta[0]?.type ?? 'unknown',
        outputDataType: outputMeta[0]?.type ?? 'unknown',
        architecture,
        detectedModelInputSize: this.modelInputSize,
      };

      console.log('[YoloPlateDetector] LOADED SUCCESSFULLY', this.modelMetadata);
    } catch (err: any) {
      console.error('[YoloPlateDetector] ONNX init error:', err);
      throw new Error(`PLATE MODEL MISSING: plate_detector.onnx failed to load (${err.message})`);
    }
  }

  // ── Detect ─────────────────────────────────────────────────────────────────
  // vehicleCropCanvas  – Canvas cropped to exactly the vehicle bounding box
  // vehicleBbox        – The vehicle's position in the original CCTV frame
  //                      (used to map plate coords back to frame coords)
  // Returns plate detections in FULL-FRAME coordinates.

  async detect(
    vehicleCropCanvas: HTMLCanvasElement,
    vehicleBbox?: BoundingBox,
    confThreshold = 0.35,
  ): Promise<DetectionResult[]> {
    if (!this.session) {
      // Return empty — caller handles missing model gracefully
      return [];
    }

    const { inputTensor, scale, padX, padY } = this.preprocess(vehicleCropCanvas);
    const feeds: Record<string, Tensor> = {};
    feeds[this.session.inputNames[0]] = inputTensor;

    const output = await this.session.run(feeds);
    const outputTensor = output[this.session.outputNames[0]];

    const { postNmsDetections } = this.decode(
      outputTensor,
      scale, padX, padY,
      vehicleCropCanvas.width,
      vehicleCropCanvas.height,
      confThreshold,
    );

    // Map from vehicle-crop coordinates to full-frame coordinates
    if (vehicleBbox) {
      return postNmsDetections.map(d => ({
        ...d,
        bbox: {
          x: vehicleBbox.x + d.bbox.x,
          y: vehicleBbox.y + d.bbox.y,
          w: d.bbox.w,
          h: d.bbox.h,
        },
      }));
    }
    return postNmsDetections;
  }

  // ── Diagnostic ─────────────────────────────────────────────────────────────
  // Runs inference on a real vehicle-crop canvas and returns full diagnostics.
  // vehicleBbox is optional — if supplied, plate coords are also returned in frame space.

  async diagnosePlate(
    vehicleCropCanvas: HTMLCanvasElement,
    vehicleBbox?: BoundingBox,
    confThreshold = 0.20,
  ): Promise<PlateDiagnosticResult> {
    const base: PlateDiagnosticResult = {
      modelLoaded: !!this.session,
      inferenceExecuted: false,
      httpStatus: this.httpStatus,
      fileSizeBytes: this.fileSizeBytes,
      inputNames: this.modelMetadata?.inputNames ?? [],
      outputNames: this.modelMetadata?.outputNames ?? [],
      inputShape: this.modelMetadata?.inputShape ?? [],
      outputShape: this.modelMetadata?.outputShape ?? [],
      inputDataType: this.modelMetadata?.inputDataType ?? 'unknown',
      outputDataType: this.modelMetadata?.outputDataType ?? 'unknown',
      architecture: this.modelMetadata?.architecture ?? 'UNKNOWN',
      detectedModelInputSize: this.modelMetadata?.detectedModelInputSize ?? this.modelInputSize,
      rawDetectionCount: 0,
      postNmsDetectionCount: 0,
      highestConfidence: 0,
      plateBoundingBox: null,
      rawDetections: [],
      postNmsDetections: [],
    };

    if (!this.session) {
      base.error = 'PLATE MODEL MISSING — session not loaded';
      return base;
    }

    try {
      const { inputTensor, scale, padX, padY } = this.preprocess(vehicleCropCanvas);

      const feeds: Record<string, Tensor> = {};
      feeds[this.session.inputNames[0]] = inputTensor;

      const output = await this.session.run(feeds);
      const outputTensor = output[this.session.outputNames[0]];

      // Update output shape from actual tensor (runtime dims may differ from metadata)
      const runtimeOutputShape = [...outputTensor.dims] as number[];

      const { rawDetections, postNmsDetections, architecture } = this.decode(
        outputTensor, scale, padX, padY,
        vehicleCropCanvas.width, vehicleCropCanvas.height,
        confThreshold,
      );

      const highestConf = rawDetections.reduce((m, d) => Math.max(m, d.confidence), 0);
      const bestPlate = postNmsDetections[0] ?? null;

      // Map to frame coords if vehicleBbox provided
      const mappedDetections = vehicleBbox
        ? postNmsDetections.map(d => ({
            ...d,
            bbox: {
              x: vehicleBbox.x + d.bbox.x,
              y: vehicleBbox.y + d.bbox.y,
              w: d.bbox.w,
              h: d.bbox.h,
            },
          }))
        : postNmsDetections;

      return {
        ...base,
        inferenceExecuted: true,
        outputShape: runtimeOutputShape,
        architecture,
        rawDetectionCount: rawDetections.length,
        postNmsDetectionCount: postNmsDetections.length,
        highestConfidence: highestConf,
        plateBoundingBox: bestPlate
          ? vehicleBbox
            ? { x: vehicleBbox.x + bestPlate.bbox.x, y: vehicleBbox.y + bestPlate.bbox.y, w: bestPlate.bbox.w, h: bestPlate.bbox.h }
            : bestPlate.bbox
          : null,
        rawDetections,
        postNmsDetections: mappedDetections,
      };
    } catch (err: any) {
      return {
        ...base,
        inferenceExecuted: false,
        error: err.message || String(err),
      };
    }
  }

  // ── Preprocessing ──────────────────────────────────────────────────────────

  private preprocess(canvas: HTMLCanvasElement): PreprocessResult {
    const sz = this.modelInputSize;
    const scale = Math.min(sz / canvas.width, sz / canvas.height);
    const newW = Math.round(canvas.width * scale);
    const newH = Math.round(canvas.height * scale);
    const padX = Math.floor((sz - newW) / 2);
    const padY = Math.floor((sz - newH) / 2);

    const tmp = document.createElement('canvas');
    tmp.width = sz;
    tmp.height = sz;
    const tctx = tmp.getContext('2d')!;

    // Standard YOLO letterbox padding (grey)
    tctx.fillStyle = 'rgb(114,114,114)';
    tctx.fillRect(0, 0, sz, sz);
    tctx.drawImage(canvas, padX, padY, newW, newH);

    const imgData = tctx.getImageData(0, 0, sz, sz).data;
    const tensorData = new Float32Array(3 * sz * sz);
    const plane = sz * sz;

    // HWC → CHW, normalize [0, 1], RGB order
    for (let i = 0, px = 0; i < imgData.length; i += 4, px++) {
      tensorData[px]           = imgData[i]     / 255.0; // R
      tensorData[px + plane]   = imgData[i + 1] / 255.0; // G
      tensorData[px + plane*2] = imgData[i + 2] / 255.0; // B
    }

    return {
      inputTensor: new Tensor('float32', tensorData, [1, 3, sz, sz]),
      scale,
      padX,
      padY,
      modelInputSize: sz,
    };
  }

  // ── Output decoding ────────────────────────────────────────────────────────

  private decode(
    tensor: Tensor,
    scale: number,
    padX: number,
    padY: number,
    origW: number,
    origH: number,
    confThreshold: number,
  ): DecodeResult {
    const data = tensor.data as Float32Array;
    const dims = tensor.dims as number[];
    let rawDetections: DetectionResult[] = [];
    let architecture = 'UNKNOWN';

    // ── Format A: YOLOv8 / YOLO11  [1, 4+C, N] where N > 4+C ──────────────
    // Each column i is an anchor: [xc, yc, w, h, cls0_conf, cls1_conf, ...]
    if (dims.length === 3 && dims[1] < dims[2]) {
      const numChannels = dims[1];   // 4 + numClasses
      const numAnchors  = dims[2];
      const numClasses  = numChannels - 4;
      architecture = numClasses <= 1
        ? 'YOLOv8/YOLO11 (1-class plate)'
        : `YOLOv8/YOLO11 (${numClasses} classes)`;

      for (let a = 0; a < numAnchors; a++) {
        let maxConf = -Infinity;
        let classIdx = 0;
        for (let c = 0; c < numClasses; c++) {
          const v = data[(4 + c) * numAnchors + a];
          if (v > maxConf) { maxConf = v; classIdx = c; }
        }
        if (maxConf < confThreshold) continue;

        const xc = data[0 * numAnchors + a];
        const yc = data[1 * numAnchors + a];
        const bw = data[2 * numAnchors + a];
        const bh = data[3 * numAnchors + a];

        const x1 = Math.max(0, (xc - bw / 2 - padX) / scale);
        const y1 = Math.max(0, (yc - bh / 2 - padY) / scale);
        const w  = Math.min(origW - x1, bw / scale);
        const h  = Math.min(origH - y1, bh / scale);

        if (w <= 0 || h <= 0) continue;

        rawDetections.push({
          bbox: { x: x1, y: y1, w, h },
          confidence: maxConf,
          class: `plate_${classIdx}`,
        });
      }
    }

    // ── Format B: YOLOv5 / YOLOv7  [1, N, 5+C] where N > 5+C ──────────────
    // Each row i is an anchor: [xc, yc, w, h, obj_conf, cls0, cls1, ...]
    else if (dims.length === 3 && dims[1] > dims[2]) {
      const numAnchors = dims[1];
      const numFields  = dims[2];
      const numClasses = numFields - 5;
      architecture = `YOLOv5/v7 (${numClasses} class${numClasses !== 1 ? 'es' : ''})`;

      for (let a = 0; a < numAnchors; a++) {
        const base = a * numFields;
        const objConf = data[base + 4];
        if (objConf < confThreshold) continue;

        let maxClsConf = 0;
        let classIdx = 0;
        for (let c = 0; c < numClasses; c++) {
          const v = data[base + 5 + c];
          if (v > maxClsConf) { maxClsConf = v; classIdx = c; }
        }

        const combined = objConf * maxClsConf;
        if (combined < confThreshold) continue;

        const xc = data[base + 0];
        const yc = data[base + 1];
        const bw = data[base + 2];
        const bh = data[base + 3];

        const x1 = Math.max(0, (xc - bw / 2 - padX) / scale);
        const y1 = Math.max(0, (yc - bh / 2 - padY) / scale);
        const w  = Math.min(origW - x1, bw / scale);
        const h  = Math.min(origH - y1, bh / scale);

        if (w <= 0 || h <= 0) continue;

        rawDetections.push({
          bbox: { x: x1, y: y1, w, h },
          confidence: combined,
          class: `plate_${classIdx}`,
        });
      }
    }

    // ── Format C: YOLOv9 / RT-DETR  [1, N, 6] (x1,y1,x2,y2,conf,cls) ──────
    else if (dims.length === 3 && dims[2] === 6) {
      const numAnchors = dims[1];
      architecture = 'YOLOv9/RT-DETR (xyxy format)';

      for (let a = 0; a < numAnchors; a++) {
        const base = a * 6;
        const conf = data[base + 4];
        if (conf < confThreshold) continue;

        const x1raw = data[base + 0];
        const y1raw = data[base + 1];
        const x2raw = data[base + 2];
        const y2raw = data[base + 3];
        const classIdx = Math.round(data[base + 5]);

        // Remove padding and inverse scale
        const x1 = Math.max(0, (x1raw - padX) / scale);
        const y1 = Math.max(0, (y1raw - padY) / scale);
        const x2 = Math.min(origW, (x2raw - padX) / scale);
        const y2 = Math.min(origH, (y2raw - padY) / scale);
        const w  = x2 - x1;
        const h  = y2 - y1;

        if (w <= 0 || h <= 0) continue;

        rawDetections.push({
          bbox: { x: x1, y: y1, w, h },
          confidence: conf,
          class: `plate_${classIdx}`,
        });
      }
    }

    // ── Format D: 2D output [1, N] or flat tensor — log and skip ─────────────
    else {
      architecture = `UNSUPPORTED dims=[${dims.join(',')}]`;
      console.warn('[YoloPlateDetector] Unsupported output format:', dims);
    }

    const postNmsDetections = this.nms(rawDetections, 0.45);

    return { rawDetections, postNmsDetections, architecture };
  }

  // ── NMS ───────────────────────────────────────────────────────────────────

  private nms(boxes: DetectionResult[], iouThreshold: number): DetectionResult[] {
    const sorted = [...boxes].sort((a, b) => b.confidence - a.confidence);
    const kept: DetectionResult[] = [];

    for (const box of sorted) {
      const overlap = kept.some(k => this.iou(box.bbox, k.bbox) > iouThreshold);
      if (!overlap) kept.push(box);
    }
    return kept;
  }

  private iou(a: BoundingBox, b: BoundingBox): number {
    const ix1 = Math.max(a.x, b.x);
    const iy1 = Math.max(a.y, b.y);
    const ix2 = Math.min(a.x + a.w, b.x + b.w);
    const iy2 = Math.min(a.y + a.h, b.y + b.h);

    const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
    if (inter === 0) return 0;
    return inter / (a.w * a.h + b.w * b.h - inter);
  }

  // ── Architecture detection helper ─────────────────────────────────────────

  private _detectArchitecture(dims: (number | bigint | string)[]): string {
    const d = dims.map(x => (typeof x === 'bigint' ? Number(x) : typeof x === 'string' ? parseInt(x) : x));
    if (d.length === 3) {
      if (d[1] < d[2]) return 'YOLOv8/YOLO11';
      if (d[1] > d[2]) return 'YOLOv5/v7';
      if (d[2] === 6)  return 'YOLOv9/RT-DETR';
    }
    return `UNKNOWN [${dims.join(',')}]`;
  }
}
