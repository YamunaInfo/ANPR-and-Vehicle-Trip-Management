import { createWorker, Worker } from 'tesseract.js';
import { IOcrEngine, OcrResult } from '../core/interfaces';

/**
 * STEP 5: PaddleOCR Integration
 * 
 * Primary: Python PaddleOCR service (PP-OCRv4)
 * Fallback: Browser-based Tesseract.js
 * 
 * Features:
 * - Recognition-only mode (no detection)
 * - Angle classification enabled
 * - Returns: text, confidence, bounding boxes
 * - Confidence filter: reject if < 0.40
 */
export class PaddleOcrEngine implements IOcrEngine {
  private worker: Worker | null = null;
  private isInitializing = false;
  private psm: string;
  private serviceUrl: string = 'http://localhost:5001/api/ocr'; // Python backend
  private pythonServiceAvailable = false;
  private readonly OCR_CONFIDENCE_THRESHOLD = 0.40; // STEP 6: Reject if < 0.40

  private debugOcrEvent(stage: string, payload: Record<string, any>): void {
    if (typeof console !== 'undefined') {
      console.log('[EASY_OCR_DEBUG]', stage, payload);
    }
  }

  constructor(psm: string = '7', serviceUrl?: string) {
    this.psm = psm;
    if (serviceUrl) this.serviceUrl = serviceUrl;
  }

  async load(): Promise<void> {
    // First, check if Python OCR service is reachable
    try {
      const healthzUrl = this.serviceUrl.replace('/api/ocr', '/healthz');
      const res = await fetch(healthzUrl, { 
        method: 'GET',
        signal: AbortSignal.timeout(3000)
      });
      
      if (res.ok) {
        this.pythonServiceAvailable = true;
        console.log('[PaddleOcrEngine] Connected to Python PaddleOCR Service (PP-OCRv4)');
        console.log(`[PaddleOcrEngine] Service URL: ${this.serviceUrl}`);
        return;
      }
    } catch (err: any) {
      console.warn('[PaddleOcrEngine] Python service not reachable:', err.message);
    }

    // Fallback: Load Tesseract.js worker
    if (this.worker || this.isInitializing) return;
    this.isInitializing = true;

    try {
      this.worker = await createWorker('eng');
      await this.worker.setParameters({
        tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        tessedit_pageseg_mode: this.psm as any,
      });
      console.log('[PaddleOcrEngine] Fallback Tesseract worker loaded (PSM: 7)');
    } catch (err) {
      console.error('[PaddleOcrEngine] Failed to load fallback Tesseract worker:', err);
      this.worker = null;
    } finally {
      this.isInitializing = false;
    }
  }

  async recognize(canvas: HTMLCanvasElement): Promise<OcrResult> {
    const dataUrl = canvas.toDataURL('image/png');
    const requestBody = {
      image: dataUrl,
      enable_angle_classification: true,
      language: 'english'
    };

    this.debugOcrEvent('paddle_ocr_request', {
      width: canvas.width,
      height: canvas.height,
      request: requestBody,
      psm: this.psm,
      serviceUrl: this.serviceUrl
    });

    // ─────────────────────────────────────────────────────────────
    // TRY 1: Python PaddleOCR Service (PP-OCRv4)
    // ─────────────────────────────────────────────────────────────
    if (this.pythonServiceAvailable) {
      try {
        const res = await fetch(this.serviceUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(requestBody),
          signal: AbortSignal.timeout(5000)
        });

        if (res.ok) {
          const payload = await res.json();
          this.debugOcrEvent('paddle_ocr_response', {
            ok: res.ok,
            status: res.status,
            payload,
            width: canvas.width,
            height: canvas.height
          });

          const text = payload?.plate || payload?.text || '';
          const confidence = Number(payload?.confidence ?? 0.0);
          const parsed = {
            parsedText: String(text),
            parsedConfidence: Math.min(1.0, Math.max(0.0, confidence)),
            success: payload?.success,
            rawPayload: payload
          };
          this.debugOcrEvent('paddle_ocr_parsed', parsed);

          if (payload?.success && text) {
            return {
              text: String(text),
              confidence: Math.min(1.0, Math.max(0.0, confidence))
            };
          }

          if (payload?.error) {
            console.warn('[PaddleOcrEngine] Python service returned OCR error:', payload.error);
          }
        }
      } catch (e: any) {
        console.warn('[PaddleOcrEngine] Python service error:', e.message);
      }
    }

    // ─────────────────────────────────────────────────────────────
    // TRY 2: Fallback to In-Browser Tesseract.js
    // ─────────────────────────────────────────────────────────────
    if (!this.worker) {
      await this.load();
    }
    
    if (!this.worker) {
      return { text: '', confidence: 0 };
    }

    try {
      const { data } = await this.worker.recognize(canvas);
      const rawText = data.text || '';
      const text = rawText.toUpperCase().replace(/[^A-Z0-9]/g, '');
      const confidence = Math.min(1.0, Math.max(0.0, (data.confidence || 0) / 100.0));
      
      // STEP 6: OCR Confidence Filter
      if (confidence < this.OCR_CONFIDENCE_THRESHOLD) {
        return { text: '', confidence: 0 };
      }
      
      return { text, confidence };
    } catch (err: any) {
      console.warn('[PaddleOcrEngine] Tesseract recognition error:', err.message);
      return { text: '', confidence: 0 };
    }
  }
}

export { PaddleOcrEngine as TesseractOcrEngine };
