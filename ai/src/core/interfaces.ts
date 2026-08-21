export interface BoundingBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface DetectionResult {
  bbox: BoundingBox;
  confidence: number;
  class?: string; // e.g. "car", "plate"
}

export interface TrackedObject extends DetectionResult {
  trackId: number;
  history: BoundingBox[];
  timeSinceUpdate: number;
  hits: number;
  hitStreak: number;
}

export interface OcrResult {
  text: string;
  confidence: number;
}

export interface ValidationResult {
  isValid: boolean;
  normalizedPlate: string;
  validationStatus: 'VALID' | 'INVALID' | 'LOW_CONFIDENCE';
  validationConfidence: number;
  patternMatch?: string;
  correctionsApplied?: number;
}

export interface PlateQualityMetrics {
  width: number;
  height: number;
  area: number;
  aspectRatio: number;
  blurScore: number; // Laplacian variance (higher = sharper)
  brightness: number; // 0-255
  contrast: number; // 0-255
  isUsable: boolean; // true if not completely unusable
  qualityReason?: string;
}

export interface PerspectiveConfig {
  rotationAngleDeg: number;
  isAngled: boolean;
  maxCorrectionAngle: number;
  cornerPoints?: { x: number; y: number }[]; // 4 corners [TL, TR, BR, BL]
}

export interface PreprocessVariant {
  id: 'A' | 'B' | 'C' | 'D' | 'E' | 'F' | 'G' | 'H' | 'I';
  name: string;
  canvas: HTMLCanvasElement;
}

export interface DebugPreprocessingInfo {
  originalCrop: string; // base64 data URL
  rectifiedCrop?: string; // after perspective correction
  superResOutput?: string; // after SR if applied
  variantSnapshots: { variantId: string; name: string; dataUrl: string; }[];
  qualityMetrics: PlateQualityMetrics;
  perspectiveInfo?: PerspectiveConfig;
  variantOcrResults: { variantId: string; rawText: string; confidence: number; validated: boolean; }[];
}

export interface PreprocessOutput {
  isValid: boolean;
  cropWidth: number;
  cropHeight: number;
  aspectRatio: number;
  qualityMetrics: PlateQualityMetrics;
  variants: PreprocessVariant[];
  rejectionReason?: string;
  rectifiedCanvas?: HTMLCanvasElement; // perspective-corrected plate
  superResCanvas?: HTMLCanvasElement; // super-resolution output if applied
}

export interface VariantOcrResult {
  variantId: 'A' | 'B' | 'C' | 'D' | 'E' | 'F' | 'G' | 'H' | 'I';
  variantName: string;
  rawText: string;
  cleanedText: string;
  ocrConfidence: number;
  validation: ValidationResult;
  candidateScore: number;
  canvasDataUrl?: string;
}

export interface StructuredOcrOutput {
  plateText: string | null;
  detectionConfidence: number;
  ocrConfidence: number;
  validationConfidence: number;
  validationStatus: 'VALID' | 'INVALID' | 'LOW_CONFIDENCE';
  trackId?: number;
  cameraId?: string;
  frameCount?: number;
  variantAgreement?: number;
  multiFrameAgreement?: number;
  cropWidth?: number;
  cropHeight?: number;
  aspectRatio?: number;
  variantResults?: VariantOcrResult[];
  originalCropDataUrl?: string;
  rectifiedCropDataUrl?: string;
  superResCropDataUrl?: string;
  debug?: DebugPreprocessingInfo;
}

export interface ProcessedPlateResult {
  text: string;
  ocrConfidence: number;
  detectionConfidence: number;
  isValid: boolean;
  timestamp: number;
  frameNumber: number;
  validationStatus?: 'VALID' | 'INVALID' | 'LOW_CONFIDENCE';
  validationConfidence?: number;
  variantResults?: VariantOcrResult[];
  originalCropDataUrl?: string;
  rectifiedCropDataUrl?: string;
  superResCropDataUrl?: string;
  cropWidth?: number;
  cropHeight?: number;
  aspectRatio?: number;
}

export interface TrackPlateObservation {
  frameNumber: number;
  timestamp: number;
  plateText: string;
  ocrConfidence: number;
  validationConfidence: number;
  isValid: boolean;
}

export interface FinalVehicleEvent {
  trackId: number;
  cameraId?: string;
  bestPlate: string | null;
  confidence: number; // fused confidence
  ocrConfidence: number;
  validationConfidence: number;
  detectionConfidence: number;
  validationStatus: 'VALID' | 'INVALID' | 'LOW_CONFIDENCE';
  vehicleClass: string;
  timestampStart: number;
  timestampEnd: number;
  frameCount: number;
  frameAgreements: number;
  multiFrameAgreement: number;
  variantAgreement: number;
  allPlates: string[];
  frameObservations: TrackPlateObservation[];
  isDuplicate: boolean;
  debugInfo?: StructuredOcrOutput;
}

export interface FrameRenderData {
  completedEvents: FinalVehicleEvent[];
  vehicles: { id: number; bbox: BoundingBox; class: string; confidence: number }[];
  plates: {
    bbox: BoundingBox;
    text?: string | null;
    confidence: number;
    ocrConfidence?: number;
    isValid?: boolean;
    validationStatus?: 'VALID' | 'INVALID' | 'LOW_CONFIDENCE';
    structuredResult?: StructuredOcrOutput;
  }[];
  fps?: number;
  totalFrames?: number;
  activeTracksCount?: number;
  ocrAttemptsCount?: number;
}

export interface IStreamInput {
  start(): Promise<void>;
  stop(): void;
  getNextFrame(): HTMLVideoElement | HTMLCanvasElement | null;
  get type(): 'video' | 'rtsp';
}

export interface IVehicleDetector {
  load(): Promise<void>;
  detect(imageSource: HTMLVideoElement | HTMLCanvasElement | HTMLImageElement): Promise<DetectionResult[]>;
}

export interface IPlateDetector {
  load(): Promise<void>;
  detect(imageSource: HTMLCanvasElement | HTMLVideoElement, vehicleBbox?: BoundingBox, confThreshold?: number): Promise<DetectionResult[]>;
  readonly modelMetadata: any | null;
  readonly httpStatus: number;
  readonly fileSizeBytes: number;
}

export interface ITracker {
  update(detections: DetectionResult[]): TrackedObject[];
}

export interface IPreprocessor {
  preprocess(plateCanvas: HTMLCanvasElement): HTMLCanvasElement[];
  preprocessDetailed(plateCanvas: HTMLCanvasElement): PreprocessOutput;
}

export interface IOcrEngine {
  load(): Promise<void>;
  recognize(canvas: HTMLCanvasElement): Promise<OcrResult>;
}

export interface IPlateValidator {
  isValidFormat(plate: string): boolean;
  normalize(plate: string): string;
  validate(plate: string): ValidationResult;
}

export interface IFusionEngine {
  addObservation(trackId: number, observation: ProcessedPlateResult, vehicleClass?: string, cameraId?: string): void;
  getBestResult(trackId: number, cameraId?: string): FinalVehicleEvent | null;
  clearTrack(trackId: number): void;
  clearOldTracks(maxAgeMs: number): void;
}
