import { 
  IVehicleDetector, 
  IPlateDetector, 
  ITracker, 
  IPreprocessor, 
  IOcrEngine, 
  IPlateValidator, 
  IFusionEngine,
  FinalVehicleEvent,
  FrameRenderData,
  BoundingBox,
  VariantOcrResult,
  StructuredOcrOutput
} from '../core/interfaces';
import { FusionEngine } from './fusion-engine';

export class ANPREngine {
  private isDetecting = false;  // Guard for fast detection path (~30ms)
  private isOcring = false;     // Guard for slow OCR path (~200-800ms); detection still runs while OCR is active
  private totalFrames = 0;
  private lastProcessTime = 0;
  private totalOcrAttempts = 0;
  private lastOcrTimePerTrack: Map<number, number> = new Map();
  private cameraId: string = 'unknown';
  private readonly debugLogEntries: any[] = [];

  // STEP 13: Performance optimization - OCR throttling
  private readonly OCR_INTERVAL_MS = 250; // Run OCR every 250-300ms per track

  constructor(
    private vehicleDetector: IVehicleDetector,
    private plateDetector: IPlateDetector,
    private tracker: ITracker,
    private preprocessor: IPreprocessor,
    private ocrEngine: IOcrEngine,
    private plateValidator: IPlateValidator,
    private fusionEngine: IFusionEngine
  ) {}

  // Set camera ID for tracking and duplicate filtering
  setCameraId(cameraId: string): void {
    this.cameraId = cameraId;
  }

  async loadAll(): Promise<void> {
    await this.vehicleDetector.load();

    try {
      await this.plateDetector.load();
    } catch (e: any) {
      console.warn('[ANPREngine] Plate detector failed to load:', e.message);
    }

    try {
      await this.ocrEngine.load();
      console.log('[ANPREngine] ✓ OCR Engine loaded successfully');
      console.log('[ANPREngine] For high accuracy, ensure Python PaddleOCR service is running on http://localhost:5001');
      console.log('[ANPREngine] Start service: cd backend && npm run ocr-service (or ./start_ocr_service.bat)');
    } catch (e: any) {
      console.warn('[ANPREngine] OCR engine failed to load:', e.message);
    }
  }

  private recordDebugEvent(event: Record<string, any>): void {
    this.debugLogEntries.push({
      timestamp: Date.now(),
      ...event
    });
    if (typeof console !== 'undefined') {
      console.log('[OCR_DEBUG]', JSON.stringify(event));
    }
  }

  private saveDebugBlob(filename: string, content: BlobPart, mimeType: string): void {
    try {
      const blob = new Blob([content], { type: mimeType });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 500);
    } catch (err) {
      console.warn('[OCR_DEBUG] Unable to save debug artifact:', filename, err);
    }
  }

  private savePlateCrop(frameNumber: number, trackId: number, canvas: HTMLCanvasElement): void {
    const filename = `debug/plate_crops/frame_${frameNumber}_track_${trackId}.png`;
    canvas.toBlob((blob) => {
      if (!blob) return;
      this.saveDebugBlob(filename, blob, 'image/png');
    }, 'image/png');
  }

  private savePreprocessedVariant(frameNumber: number, trackId: number, variantId: string, canvas: HTMLCanvasElement): void {
    const filename = `debug/preprocessed/frame_${frameNumber}_track_${trackId}_variant_${variantId}.png`;
    canvas.toBlob((blob) => {
      if (!blob) return;
      this.saveDebugBlob(filename, blob, 'image/png');
    }, 'image/png');
  }

  private writeDebugLogFile(frameNumber: number, payload: Record<string, any>): void {
    const filename = `debug/ocr_logs/frame_${frameNumber}_trace.json`;
    this.saveDebugBlob(filename, JSON.stringify(payload, null, 2), 'application/json');
  }

  private logLoss(stage: string, fileName: string, lineNumber: number, reason: string, value: unknown): void {
    const message = `[OCR_DEBUG_LOSS] stage=${stage} file=${fileName} line=${lineNumber} reason=${reason} value=${JSON.stringify(value)}`;
    console.error(message);
    this.recordDebugEvent({ stage, fileName, lineNumber, reason, value });
  }

  async processFrame(
    video: HTMLVideoElement,
    timestamp: number,
    onLog?: (msg: string) => void,
    forceOcr: boolean = true
  ): Promise<FrameRenderData> {
    // Fast path guard: skip only if detection itself is already running (~30ms)
    // OCR running (isOcring) does NOT block detection — tracker must be fed every frame
    if (this.isDetecting) return { completedEvents: [], vehicles: [], plates: [] };
    this.isDetecting = true;

    const log = (msg: string) => onLog && onLog(msg);
    const W = video.videoWidth || 640;
    const H = video.videoHeight || 480;
    const completedEvents: FinalVehicleEvent[] = [];
    const outVehicles: { id: number; bbox: BoundingBox; class: string; confidence: number }[] = [];
    const outPlates: {
      bbox: BoundingBox;
      text?: string | null;
      confidence: number;
      ocrConfidence?: number;
      isValid?: boolean;
      validationStatus?: 'VALID' | 'INVALID' | 'LOW_CONFIDENCE';
      structuredResult?: StructuredOcrOutput;
    }[] = [];

    try {
      this.totalFrames++;
      const now = performance.now();
      this.lastProcessTime = now;

      // Render frame to full canvas for cropping
      const fc = document.createElement('canvas');
      fc.width = W; fc.height = H;
      fc.getContext('2d')!.drawImage(video, 0, 0, W, H);

      // STEP 1 & 2: Detect Vehicles and update tracker
      const vehicleDetections = await this.vehicleDetector.detect(fc);
      const trackedVehicles = this.tracker.update(vehicleDetections);
      log(`Vehicle detected | Active Tracks: ${trackedVehicles.length}`);

      // Process each active tracked vehicle
      for (const vehicle of trackedVehicles) {
        const frameNumber = this.totalFrames;
        log(`--------------------------------------------------`);
        log(`Vehicle Class: ${vehicle.class ?? 'vehicle'} | Track ID: ${vehicle.trackId} | Confidence: ${(vehicle.confidence * 100).toFixed(1)}%`);
        log(`==============================`);
        log(`Frame: ${frameNumber}`);
        log(`Track ID: ${vehicle.trackId}`);
        log(`Vehicle Detection:`);
        log(`- Confidence: ${(vehicle.confidence * 100).toFixed(1)}%`);
        log(`- Bounding Box: ${JSON.stringify(vehicle.bbox)}`);
        this.recordDebugEvent({
          frameNumber,
          trackId: vehicle.trackId,
          stage: 'vehicle_detection',
          confidence: vehicle.confidence,
          bbox: vehicle.bbox,
          vehicleClass: vehicle.class ?? 'vehicle'
        });

        outVehicles.push({
          id: vehicle.trackId,
          bbox: vehicle.bbox,
          class: vehicle.class ?? 'vehicle',
          confidence: vehicle.confidence
        });

        // Crop vehicle region from full frame
        const vCanvas = this.cropCanvas(fc, vehicle.bbox.x, vehicle.bbox.y, vehicle.bbox.w, vehicle.bbox.h);
        
        // Detect License Plate inside vehicle crop
        const plateDetections = await this.plateDetector.detect(vCanvas, vehicle.bbox);

        // STEP 13: OCR Throttling - only run OCR every 250-300ms per track
        const lastOcrTime = this.lastOcrTimePerTrack.get(vehicle.trackId) || 0;
        const shouldRunOcr = forceOcr || (now - lastOcrTime >= this.OCR_INTERVAL_MS);

        for (const p of plateDetections) {
          log(`--------------------------------------------------`);
          log(`Plate detected | Confidence: ${(p.confidence * 100).toFixed(1)}%`);
          log(`Plate Detection:`);
          log(`- Confidence: ${(p.confidence * 100).toFixed(1)}%`);
          log(`- Bounding Box: ${JSON.stringify(p.bbox)}`);

          // Crop plate from full frame
          const cropX = Math.max(0, Math.min(W - 1, Math.round(p.bbox.x)));
          const cropY = Math.max(0, Math.min(H - 1, Math.round(p.bbox.y)));
          const cropW = Math.max(1, Math.min(W - cropX, Math.round(p.bbox.w)));
          const cropH = Math.max(1, Math.min(H - cropY, Math.round(p.bbox.h)));
          const pCanvas = this.cropCanvas(fc, cropX, cropY, cropW, cropH);
          const originalCropDataUrl = pCanvas.toDataURL('image/png');
          this.savePlateCrop(this.totalFrames, vehicle.trackId, pCanvas);
          log(`- Crop Width: ${cropW}`);
          log(`- Crop Height: ${cropH}`);
          this.recordDebugEvent({
            frameNumber: this.totalFrames,
            trackId: vehicle.trackId,
            stage: 'plate_crop',
            detectionConfidence: p.confidence,
            crop: { x: cropX, y: cropY, width: cropW, height: cropH },
            imageSize: { width: pCanvas.width, height: pCanvas.height }
          });

          let structuredResult: StructuredOcrOutput | undefined;

          // Only run OCR if not already running another OCR (OCR is slow: 200-800ms)
          // Detection above already ran — tracker is up to date regardless
          const canRunOcr = shouldRunOcr && !this.isOcring;
          if (canRunOcr) {
            this.isOcring = true;
            this.lastOcrTimePerTrack.set(vehicle.trackId, now);

            // STEP 1: PLATE QUALITY CHECK + STEP 2: PERSPECTIVE RECTIFICATION + STEP 3: SUPER RESOLUTION + STEP 4: PREPROCESSING
            const prepOutput = this.preprocessor.preprocessDetailed(pCanvas);
            
            log(`═══════════════════════════════════════════════════════════`);
            log(`PLATE CROP ANALYSIS - Track #${vehicle.trackId}`);
            log(`═══════════════════════════════════════════════════════════`);
            log(`Detector Confidence: ${(p.confidence * 100).toFixed(1)}%`);
            log(`Crop Dimensions: ${prepOutput.cropWidth}×${prepOutput.cropHeight}px (aspect ratio: ${prepOutput.aspectRatio.toFixed(2)})`);
            log(`Quality Metrics:`);
            log(`  • Blur Score: ${prepOutput.qualityMetrics.blurScore.toFixed(2)} (lower=sharper, >20=rejected)`);
            log(`  • Brightness: ${prepOutput.qualityMetrics.brightness}/255`);
            log(`  • Contrast: ${prepOutput.qualityMetrics.contrast}/255`);
            log(`  • Status: ${prepOutput.qualityMetrics.isUsable ? 'USABLE' : 'REJECTED'}`);
            log(`- Blur Score: ${prepOutput.qualityMetrics.blurScore.toFixed(2)}`);
            log(`- Brightness: ${prepOutput.qualityMetrics.brightness}`);
            log(`- Contrast: ${prepOutput.qualityMetrics.contrast}`);
            this.recordDebugEvent({
              frameNumber: this.totalFrames,
              trackId: vehicle.trackId,
              stage: 'preprocessing_quality',
              cropWidth: prepOutput.cropWidth,
              cropHeight: prepOutput.cropHeight,
              blurScore: prepOutput.qualityMetrics.blurScore,
              brightness: prepOutput.qualityMetrics.brightness,
              contrast: prepOutput.qualityMetrics.contrast,
              isUsable: prepOutput.qualityMetrics.isUsable,
              rejectionReason: prepOutput.rejectionReason
            });
            
            if (!prepOutput.isValid) {
              log(`❌ PLATE REJECTED: ${prepOutput.rejectionReason}`);
              structuredResult = {
                plateText: null,
                detectionConfidence: p.confidence,
                ocrConfidence: 0.0,
                validationConfidence: 0.0,
                validationStatus: 'LOW_CONFIDENCE',
                variantAgreement: 0,
                cropWidth: prepOutput.cropWidth,
                cropHeight: prepOutput.cropHeight,
                aspectRatio: prepOutput.aspectRatio,
                variantResults: [],
                originalCropDataUrl
              };
            } else {
              log(`✓ Plate quality acceptable. Processing 8 preprocessing variants...`);
              log(`Perspective Correction: ${prepOutput.rectifiedCanvas ? 'Applied' : 'None'}`);
              log(`Super Resolution: ${prepOutput.superResCanvas ? 'Applied (upscaled)' : 'Not needed'}`);
              log(``);

              // STEP 5 & 6: OCR on all 8 preprocessing variants
              const variantResults: VariantOcrResult[] = [];
              let bestVariantConfidence = 0;
              let bestVariantId = 'A';

              for (const variant of prepOutput.variants) {
                this.totalOcrAttempts++;
                const t0 = performance.now();
                this.savePreprocessedVariant(this.totalFrames, vehicle.trackId, variant.id, variant.canvas);
                log(`Preprocessing: Variant ${variant.id} (${variant.name})`);
                this.recordDebugEvent({
                  frameNumber: this.totalFrames,
                  trackId: vehicle.trackId,
                  stage: 'preprocessed_variant',
                  variant: variant.id,
                  variantName: variant.name,
                  variantSize: { width: variant.canvas.width, height: variant.canvas.height }
                });
                
                const ocrRes = await this.ocrEngine.recognize(variant.canvas);
                const tElapsed = Math.round(performance.now() - t0);

                // STEP 7 & 8: Indian Plate Validation + Position-Aware Correction
                const valRes = (this.plateValidator as any).validate ? 
                  (this.plateValidator as any).validate(ocrRes.text) : 
                  {
                    isValid: this.plateValidator.isValidFormat(ocrRes.text),
                    normalizedPlate: this.plateValidator.normalize(ocrRes.text),
                    validationStatus: this.plateValidator.isValidFormat(ocrRes.text) ? 'VALID' : 'INVALID',
                    validationConfidence: this.plateValidator.isValidFormat(ocrRes.text) ? 1.0 : 0.2
                  };

                const candidateScore = (valRes.validationConfidence * 0.40) + (ocrRes.confidence * 0.40);
                
                if (ocrRes.confidence > bestVariantConfidence) {
                  bestVariantConfidence = ocrRes.confidence;
                  bestVariantId = variant.id;
                }

                log(`Variant ${variant.id} - ${variant.name}`);
                log(`  Raw: "${ocrRes.text}" (OCR conf: ${(ocrRes.confidence * 100).toFixed(0)}%, time: ${tElapsed}ms)`);
                log(`  Validated: "${valRes.normalizedPlate}" (status: ${valRes.validationStatus}, conf: ${(valRes.validationConfidence * 100).toFixed(0)}%)`);
                log(`  Score: ${candidateScore.toFixed(3)}`);
                log(`Python PaddleOCR Request:`);
                log(`- Request sent: true`);
                log(`- Image size: ${variant.canvas.width}x${variant.canvas.height}`);
                log(`Frontend Parser:`);
                log(`- Parsed text: "${ocrRes.text}"`);
                log(`- Parsed confidence: ${(ocrRes.confidence * 100).toFixed(0)}%`);
                log(`Validation:`);
                log(`- Original text: "${ocrRes.text}"`);
                log(`- Corrected text: "${valRes.normalizedPlate}"`);
                log(`- Validation status: ${valRes.validationStatus}`);
                this.recordDebugEvent({
                  frameNumber: this.totalFrames,
                  trackId: vehicle.trackId,
                  stage: 'ocr_attempt',
                  variant: variant.id,
                  variantName: variant.name,
                  cropSize: { width: cropW, height: cropH },
                  detectionConfidence: p.confidence,
                  blurScore: prepOutput.qualityMetrics.blurScore,
                  brightness: prepOutput.qualityMetrics.brightness,
                  contrast: prepOutput.qualityMetrics.contrast,
                  rawOcrText: ocrRes.text,
                  rawOcrConfidence: ocrRes.confidence,
                  validationResult: valRes,
                  elapsedMs: tElapsed,
                  candidateScore
                });

                if (!ocrRes.text && String(ocrRes.text).length === 0) {
                  this.logLoss('ocr_result_empty', 'anpr-engine.ts', 214, 'OCR result is empty before validation', { ocrRes, variant: variant.id });
                }

                variantResults.push({
                  variantId: variant.id,
                  variantName: variant.name,
                  rawText: ocrRes.text,
                  cleanedText: valRes.normalizedPlate,
                  ocrConfidence: ocrRes.confidence,
                  validation: valRes,
                  candidateScore,
                  canvasDataUrl: variant.canvas.toDataURL('image/png')
                });
              }

              log(``);
              log(`✓ FUSION: Selecting highest-confidence valid plate...`);

              // STEP 9: Multi-Variant Fusion
              structuredResult = FusionEngine.fuseVariants(
                variantResults,
                p.confidence,
                prepOutput.cropWidth,
                prepOutput.cropHeight,
                prepOutput.aspectRatio,
                originalCropDataUrl,
                prepOutput.rectifiedCanvas?.toDataURL('image/png'),
                prepOutput.superResCanvas?.toDataURL('image/png')
              );

              log(`📋 RESULT: "${structuredResult.plateText || 'UNREAD'}" (OCR: ${(structuredResult.ocrConfidence * 100).toFixed(0)}%, Validation: ${(structuredResult.validationConfidence * 100).toFixed(0)}%, Status: ${structuredResult.validationStatus})`);
              log(`═══════════════════════════════════════════════════════════`);
            }

            this.isOcring = false;
          }

          if (!structuredResult) {
            structuredResult = {
              plateText: null,
              detectionConfidence: p.confidence,
              ocrConfidence: 0.0,
              validationConfidence: 0.0,
              validationStatus: 'LOW_CONFIDENCE',
              variantAgreement: 0,
              cropWidth: cropW,
              cropHeight: cropH,
              aspectRatio: cropW > 0 && cropH > 0 ? Number((cropW / cropH).toFixed(2)) : 0,
              variantResults: [],
              originalCropDataUrl
            };
          }

          // STEP 10: Add observation to Fusion Engine for multi-frame consensus
          const fusionObservation = {
            text: structuredResult.plateText || 'UNREAD',
            ocrConfidence: structuredResult.ocrConfidence,
            detectionConfidence: p.confidence,
            isValid: structuredResult.validationStatus === 'VALID',
            validationStatus: structuredResult.validationStatus,
            variantResults: structuredResult.variantResults,
            originalCropDataUrl: structuredResult.originalCropDataUrl,
            rectifiedCropDataUrl: structuredResult.rectifiedCropDataUrl,
            superResCropDataUrl: structuredResult.superResCropDataUrl,
            cropWidth: structuredResult.cropWidth,
            cropHeight: structuredResult.cropHeight,
            aspectRatio: structuredResult.aspectRatio,
            timestamp,
            frameNumber: this.totalFrames
          };

          const fusionObservationAdded = Boolean(structuredResult?.plateText || structuredResult?.validationStatus);
          log(`Fusion:`);
          log(`- Observation added: ${fusionObservationAdded ? 'YES' : 'NO'}`);
          log(`- Reason if skipped: ${fusionObservationAdded ? 'N/A' : 'No structured OCR result available'}`);
          log(`- Final plate: "${fusionObservation.text}"`);
          this.recordDebugEvent({
            frameNumber: this.totalFrames,
            trackId: vehicle.trackId,
            stage: 'fusion_input',
            observationAdded: fusionObservationAdded,
            observationText: fusionObservation.text,
            observationStatus: fusionObservation.validationStatus,
            observationConfidence: fusionObservation.ocrConfidence,
            detectionConfidence: fusionObservation.detectionConfidence
          });
          this.fusionEngine.addObservation(
            vehicle.trackId,
            fusionObservation,
            vehicle.class,
            this.cameraId
          );

          outPlates.push({ 
            bbox: p.bbox,
            text: structuredResult?.plateText || null,
            confidence: p.confidence,
            ocrConfidence: structuredResult?.ocrConfidence || 0,
            isValid: structuredResult?.validationStatus === 'VALID',
            validationStatus: structuredResult?.validationStatus || 'LOW_CONFIDENCE',
            structuredResult
          });
        }

        // STEP 10 & 11: Get Multi-frame consensus event for this track with duplicate filtering
        const currentBest = this.fusionEngine.getBestResult(vehicle.trackId, this.cameraId);
        if (currentBest && !currentBest.isDuplicate) {
          log(`Track #${vehicle.trackId}: ${currentBest.bestPlate} (${currentBest.validationStatus})`);
          log(`Final fused plate: "${currentBest.bestPlate ?? 'NULL'}"`);
          log(`Fusion current observations for track: ${currentBest.frameObservations.length}`);
          this.recordDebugEvent({
            frameNumber: this.totalFrames,
            trackId: vehicle.trackId,
            stage: 'fusion_result',
            finalPlate: currentBest.bestPlate,
            validationStatus: currentBest.validationStatus,
            ocrConfidence: currentBest.ocrConfidence,
            frameObservations: currentBest.frameObservations,
            allPlates: currentBest.allPlates
          });
          completedEvents.push(currentBest);
        }
      }

      // Cleanup stale tracks and events
      this.fusionEngine.clearOldTracks(10000);
      (this.fusionEngine as any).clearOldDuplicates?.(30000);

      log(`Frame ${this.totalFrames} completed.`);

    } catch (err: any) {
      log(`ERROR in ANPREngine.processFrame: ${err.message || err}`);
      this.isOcring = false; // ensure OCR lock is released on error
    } finally {
      this.isDetecting = false;
    }

    return { 
      completedEvents, 
      vehicles: outVehicles, 
      plates: outPlates,
      fps: this.lastProcessTime > 0 ? 1000 / (performance.now() - this.lastProcessTime) : 0,
      totalFrames: this.totalFrames,
      activeTracksCount: outVehicles.length,
      ocrAttemptsCount: this.totalOcrAttempts
    };
  }

  private cropCanvas(src: HTMLCanvasElement, sx: number, sy: number, sw: number, sh: number): HTMLCanvasElement {
    const ow = Math.max(1, Math.round(sw));
    const oh = Math.max(1, Math.round(sh));
    const c = document.createElement('canvas');
    c.width = ow;
    c.height = oh;
    const ctx = c.getContext('2d')!;
    ctx.drawImage(src, Math.max(0, sx), Math.max(0, sy), sw, sh, 0, 0, ow, oh);
    return c;
  }
}

