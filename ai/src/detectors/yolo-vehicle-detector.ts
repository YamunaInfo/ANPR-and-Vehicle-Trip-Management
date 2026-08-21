import { InferenceSession, Tensor, env } from 'onnxruntime-web';
import { IVehicleDetector, DetectionResult } from '../core/interfaces';

export interface DiagnosticResult {
  modelLoaded: boolean;
  inferenceExecuted: boolean;
  httpStatus?: number;
  fileSizeBytes?: number;
  rawOutputShape: string;
  architecture: string;
  rawDetectionCount: number;
  postNmsDetectionCount: number;
  detectedClassIds: number[];
  detectedClassNames: string[];
  highestConfidence: number;
  rawDetections: DetectionResult[];
  postNmsDetections: DetectionResult[];
  error?: string;
}

export class YoloVehicleDetector implements IVehicleDetector {
  private session: InferenceSession | null = null;
  private targetSize = 640; // Default YOLOv8 input size

  public modelMetadata: any = null;
  public httpStatus: number = 0;
  public fileSizeBytes: number = 0;

  async load(): Promise<void> {
    if (this.session) return;

    // 1. Verify HTTP availability and file size
    try {
      const res = await fetch('/models/vehicle_detector.onnx', { method: 'HEAD' });
      this.httpStatus = res.status;
      const contentLength = res.headers.get('content-length');
      this.fileSizeBytes = contentLength ? parseInt(contentLength, 10) : 0;
      
      if (res.status !== 200) {
        console.error(`[YoloVehicleDetector] HTTP fetch failed: ${res.status}`);
        throw new Error(`MODEL MISSING (HTTP ${res.status}): /models/vehicle_detector.onnx not found.`);
      }
    } catch (err: any) {
      if (err.message && err.message.startsWith('MODEL MISSING')) throw err;
      console.error('[YoloVehicleDetector] Fetch check error:', err);
      throw new Error(`MODEL MISSING: Cannot fetch /models/vehicle_detector.onnx (${err.message})`);
    }

    // 2. Load ONNX Session
    try {
      // Configure WASM paths - must be set before creating InferenceSession.
      // Points to where the ort-wasm-*.mjs / *.wasm files are served (public root).
      env.wasm.wasmPaths = '/';
      // Disable multi-threading to avoid SharedArrayBuffer requirement in dev
      env.wasm.numThreads = 1;

      this.session = await InferenceSession.create('/models/vehicle_detector.onnx', {
        executionProviders: ['wasm']
      });

      this.modelMetadata = {
        inputNames: this.session.inputNames,
        outputNames: this.session.outputNames,
        httpStatus: this.httpStatus,
        fileSizeBytes: this.fileSizeBytes
      };
      console.log('[YoloVehicleDetector] LOADED SUCCESSFULLY', this.modelMetadata);
    } catch (err: any) {
      console.error('[YoloVehicleDetector] ONNX init error:', err);
      throw new Error(`MODEL MISSING: vehicle_detector.onnx failed to load (${err.message})`);
    }
  }

  async detect(canvas: HTMLCanvasElement): Promise<DetectionResult[]> {
    if (!this.session) {
      throw new Error('Vehicle model inference failed (MODEL MISSING).');
    }

    const { inputTensor, padX, padY, scale } = this.preprocess(canvas);
    
    const feeds: Record<string, Tensor> = {};
    feeds[this.session.inputNames[0]] = inputTensor;
    
    const output = await this.session.run(feeds);
    const outputTensor = output[this.session.outputNames[0]];
    
    const { postNmsDetections } = this.decodeOutput(outputTensor, scale, padX, padY, canvas.width, canvas.height, 0.30);
    return postNmsDetections;
  }

  /**
   * Diagnostic execution on ONE frame from video with confThreshold = 0.20
   */
  async diagnoseFrame(canvas: HTMLCanvasElement, confThreshold: number = 0.20): Promise<DiagnosticResult> {
    if (!this.session) {
      return {
        modelLoaded: false,
        inferenceExecuted: false,
        httpStatus: this.httpStatus,
        fileSizeBytes: this.fileSizeBytes,
        rawOutputShape: 'N/A',
        architecture: 'UNKNOWN',
        rawDetectionCount: 0,
        postNmsDetectionCount: 0,
        detectedClassIds: [],
        detectedClassNames: [],
        highestConfidence: 0,
        rawDetections: [],
        postNmsDetections: [],
        error: 'MODEL MISSING'
      };
    }

    try {
      const { inputTensor, padX, padY, scale } = this.preprocess(canvas);
      const feeds: Record<string, Tensor> = {};
      feeds[this.session.inputNames[0]] = inputTensor;

      const output = await this.session.run(feeds);
      const outputTensor = output[this.session.outputNames[0]];

      const decoded = this.decodeOutput(outputTensor, scale, padX, padY, canvas.width, canvas.height, confThreshold);

      const classIdsSet = new Set<number>();
      const classNamesSet = new Set<string>();
      let maxConf = 0;

      for (const d of decoded.rawDetections) {
        if (d.confidence > maxConf) maxConf = d.confidence;
        if (d.class) classNamesSet.add(d.class);
      }

      return {
        modelLoaded: true,
        inferenceExecuted: true,
        httpStatus: this.httpStatus,
        fileSizeBytes: this.fileSizeBytes,
        rawOutputShape: JSON.stringify(outputTensor.dims),
        architecture: decoded.architecture,
        rawDetectionCount: decoded.rawDetections.length,
        postNmsDetectionCount: decoded.postNmsDetections.length,
        detectedClassIds: Array.from(classIdsSet),
        detectedClassNames: Array.from(classNamesSet),
        highestConfidence: maxConf,
        rawDetections: decoded.rawDetections,
        postNmsDetections: decoded.postNmsDetections
      };
    } catch (err: any) {
      return {
        modelLoaded: true,
        inferenceExecuted: false,
        rawOutputShape: 'N/A',
        architecture: 'UNKNOWN',
        rawDetectionCount: 0,
        postNmsDetectionCount: 0,
        detectedClassIds: [],
        detectedClassNames: [],
        highestConfidence: 0,
        rawDetections: [],
        postNmsDetections: [],
        error: err.message || String(err)
      };
    }
  }

  private preprocess(canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext('2d')!;
    const scale = Math.min(this.targetSize / canvas.width, this.targetSize / canvas.height);
    const newW = Math.round(canvas.width * scale);
    const newH = Math.round(canvas.height * scale);
    const padX = Math.floor((this.targetSize - newW) / 2);
    const padY = Math.floor((this.targetSize - newH) / 2);

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = this.targetSize;
    tempCanvas.height = this.targetSize;
    const tempCtx = tempCanvas.getContext('2d')!;
    
    // Gray padding (standard YOLO)
    tempCtx.fillStyle = 'rgb(114, 114, 114)';
    tempCtx.fillRect(0, 0, this.targetSize, this.targetSize);
    
    // Scale and draw image
    tempCtx.drawImage(canvas, padX, padY, newW, newH);
    
    const imgData = tempCtx.getImageData(0, 0, this.targetSize, this.targetSize).data;
    const tensorData = new Float32Array(3 * this.targetSize * this.targetSize);

    // HWC to CHW and normalize to [0, 1] (RGB)
    for (let i = 0, j = 0; i < imgData.length; i += 4, j++) {
      tensorData[j] = imgData[i] / 255.0; // R
      tensorData[j + this.targetSize * this.targetSize] = imgData[i + 1] / 255.0; // G
      tensorData[j + 2 * this.targetSize * this.targetSize] = imgData[i + 2] / 255.0; // B
    }

    return {
      inputTensor: new Tensor('float32', tensorData, [1, 3, this.targetSize, this.targetSize]),
      scale,
      padX,
      padY
    };
  }

  private decodeOutput(
    tensor: Tensor, 
    scale: number, 
    padX: number, 
    padY: number, 
    origW: number, 
    origH: number,
    confThreshold: number
  ): { rawDetections: DetectionResult[]; postNmsDetections: DetectionResult[]; architecture: string } {
    const data = tensor.data as Float32Array;
    const dims = tensor.dims;
    
    const rawDetections: DetectionResult[] = [];
    let architecture = 'UNKNOWN';

    // 1. YOLOv8 / YOLO11 Shape: [1, 4 + numClasses, numAnchors] (e.g. [1, 84, 8400] for COCO)
    if (dims.length === 3 && dims[1] < dims[2]) {
      architecture = dims[1] === 84 ? 'YOLOv8 / YOLO11 (COCO)' : `YOLOv8 / YOLO11 (${dims[1] - 4} classes)`;
      const numChannels = dims[1];
      const numAnchors = dims[2];
      const numClasses = numChannels - 4;

      for (let i = 0; i < numAnchors; i++) {
        let maxClassConf = 0;
        let classIdx = 0;

        for (let c = 0; c < numClasses; c++) {
          const conf = data[(4 + c) * numAnchors + i];
          if (conf > maxClassConf) {
            maxClassConf = conf;
            classIdx = c;
          }
        }

        if (maxClassConf > confThreshold) {
          const xc = data[0 * numAnchors + i];
          const yc = data[1 * numAnchors + i];
          const w = data[2 * numAnchors + i];
          const h = data[3 * numAnchors + i];

          let x1 = (xc - w / 2 - padX) / scale;
          let y1 = (yc - h / 2 - padY) / scale;
          const wOriginal = w / scale;
          const hOriginal = h / scale;

          let className = 'vehicle';
          if (numClasses > 10) {
            // Filter COCO vehicle classes: 2=car, 3=motorcycle, 5=bus, 7=truck
            if (![2, 3, 5, 7].includes(classIdx)) continue;
            const names: Record<number, string> = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'};
            className = names[classIdx] || 'vehicle';
          } else {
            const vehicleNames = ['car', 'truck', 'bus', 'motorcycle', 'auto'];
            className = vehicleNames[classIdx] || `class_${classIdx}`;
          }

          rawDetections.push({
            bbox: {
              x: Math.max(0, x1),
              y: Math.max(0, y1),
              w: Math.min(origW - x1, wOriginal),
              h: Math.min(origH - y1, hOriginal)
            },
            confidence: maxClassConf,
            class: className
          });
        }
      }
    } 
    // 2. YOLOv5 Shape: [1, numAnchors, 5 + numClasses] (e.g. [1, 25200, 85])
    else if (dims.length === 3 && dims[1] > dims[2]) {
      architecture = 'YOLOv5';
      const numAnchors = dims[1];
      const numFields = dims[2];
      const numClasses = numFields - 5;

      for (let i = 0; i < numAnchors; i++) {
        const objConf = data[i * numFields + 4];
        if (objConf <= confThreshold) continue;

        let maxClassConf = 0;
        let classIdx = 0;
        for (let c = 0; c < numClasses; c++) {
          const conf = data[i * numFields + 5 + c];
          if (conf > maxClassConf) {
            maxClassConf = conf;
            classIdx = c;
          }
        }

        const totalConf = objConf * maxClassConf;
        if (totalConf > confThreshold) {
          const xc = data[i * numFields + 0];
          const yc = data[i * numFields + 1];
          const w = data[i * numFields + 2];
          const h = data[i * numFields + 3];

          let x1 = (xc - w / 2 - padX) / scale;
          let y1 = (yc - h / 2 - padY) / scale;
          const wOriginal = w / scale;
          const hOriginal = h / scale;

          let className = 'vehicle';
          if (numClasses > 10) {
            if (![2, 3, 5, 7].includes(classIdx)) continue;
            const names: Record<number, string> = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'};
            className = names[classIdx] || 'vehicle';
          }

          rawDetections.push({
            bbox: {
              x: Math.max(0, x1),
              y: Math.max(0, y1),
              w: Math.min(origW - x1, wOriginal),
              h: Math.min(origH - y1, hOriginal)
            },
            confidence: totalConf,
            class: className
          });
        }
      }
    }

    // 3. Non-Maximum Suppression (NMS)
    const sorted = [...rawDetections].sort((a, b) => b.confidence - a.confidence);
    const postNmsDetections: DetectionResult[] = [];
    const iouThreshold = 0.45;

    for (const box of sorted) {
      let keep = true;
      for (const keptBox of postNmsDetections) {
        if (this.calculateIoU(box.bbox, keptBox.bbox) > iouThreshold) {
          keep = false;
          break;
        }
      }
      if (keep) {
        postNmsDetections.push(box);
      }
    }

    return { rawDetections, postNmsDetections, architecture };
  }

  private calculateIoU(boxA: any, boxB: any): number {
    const xA = Math.max(boxA.x, boxB.x);
    const yA = Math.max(boxA.y, boxB.y);
    const xB = Math.min(boxA.x + boxA.w, boxB.x + boxB.w);
    const yB = Math.min(boxA.y + boxA.h, boxB.y + boxB.h);

    const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
    if (interArea === 0) return 0;

    const boxAArea = boxA.w * boxA.h;
    const boxBArea = boxB.w * boxB.h;

    return interArea / (boxAArea + boxBArea - interArea);
  }
}

