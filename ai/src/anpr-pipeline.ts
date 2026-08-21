/**
 * ANPR Detection Pipeline — GateSense
 *
 * Full pipeline:
 *   Video Frame
 *     → COCO-SSD Vehicle Detection  (TensorFlow.js)
 *     → Plate Region Localization   (within vehicle bbox)
 *     → Multi-variant Preprocessing (contrast, Otsu, inverted)
 *     → Multi-PSM Tesseract OCR     (PSM 7, 8)
 *     → Frame-level candidate selection
 *     → Multi-frame Fusion          (edit-distance grouping + vote scoring)
 *     → Confidence Scoring          (vehicle + plate + OCR + agreement)
 *     → Final Plate / PLATE_UNCERTAIN
 */

import { createWorker } from 'tesseract.js';

// ─── Public Types ─────────────────────────────────────────────────────────────

export interface VehicleBox {
  bbox: [number, number, number, number]; // x, y, w, h (COCO-SSD format)
  score: number;
  class: string;
}

export interface OcrAttempt {
  text: string;
  confidence: number; // 0–1
  variant: string;
  psm: string;
  regionLabel: string;
}

export interface FrameResult {
  timestamp: number;
  vehicles: VehicleBox[];
  /** Plate bounding box (x,y,w,h) inside the full frame, or null if none found */
  plateBbox: [number, number, number, number] | null;
  ocrAttempts: OcrAttempt[];
  bestText: string | null;
  /** Composite quality 0–1: vehicle_detected + plate_found + ocr_confidence */
  quality: number;
}

export interface PipelineOutput {
  plate: string;
  confidence: number;       // 0–1
  frameAgreement: number;   // fraction of frames that agreed on this plate
  isPlateCertain: boolean;
  vehicleType: string | null;
  frames: FrameResult[];
  debugLog: string[];
}

// ─── Constants ────────────────────────────────────────────────────────────────

const VEHICLE_CLASSES = new Set(['car', 'truck', 'bus', 'motorcycle', 'motorbike', 'van']);

/** Strict Indian registration plate format */
const STRICT_RE = /^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{3,4}$/;

/** Loose pattern — tolerates common OCR character confusions */
const LOOSE_RE = /[A-Z0-9]{2}[0-9]{1,2}[A-Z0-9]{1,3}[0-9]{3,4}/;

// ─── Singletons ───────────────────────────────────────────────────────────────

let _detector: any = null;

/**
 * Load the COCO-SSD vehicle detector once and reuse.
 * Uses `lite_mobilenet_v2` for a smaller download.
 */
export async function loadDetector(): Promise<any> {
  if (!_detector) {
    // Dynamic imports so TF.js doesn't block the initial app load
    const [cocoSsdMod] = await Promise.all([
      import('@tensorflow-models/coco-ssd'),
      import('@tensorflow/tfjs-backend-webgl'),
      import('@tensorflow/tfjs'),
    ]);
    _detector = await cocoSsdMod.load({ base: 'lite_mobilenet_v2' });
  }
  return _detector;
}

// Single persistent Tesseract worker — creates once, reuses across all OCR calls
let _tessWorker: Awaited<ReturnType<typeof createWorker>> | null = null;

async function getTessWorker() {
  if (!_tessWorker) {
    _tessWorker = await createWorker('eng');
    await _tessWorker.setParameters({
      tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
    });
  }
  return _tessWorker;
}

// ─── Image Utilities ──────────────────────────────────────────────────────────

/** Draw a source rect into a new canvas at the given scale factor */
function cropAndScale(
  src: HTMLCanvasElement | HTMLVideoElement,
  sx: number, sy: number, sw: number, sh: number,
  scale: number
): HTMLCanvasElement {
  const ow = Math.max(1, Math.round(sw * scale));
  const oh = Math.max(1, Math.round(sh * scale));
  const c = document.createElement('canvas');
  c.width = ow; c.height = oh;
  c.getContext('2d')!.drawImage(src as any, sx, sy, sw, sh, 0, 0, ow, oh);
  return c;
}

/** Extract per-pixel luminance values */
function toLuminance(data: Uint8ClampedArray): Float32Array {
  const out = new Float32Array(data.length >> 2);
  for (let i = 0, gi = 0; i < data.length; i += 4, gi++) {
    out[gi] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  }
  return out;
}

/** Adaptive contrast stretch — maps [actual_min, actual_max] → [0, 255] */
function contrastStretch(canvas: HTMLCanvasElement): HTMLCanvasElement {
  const ctx = canvas.getContext('2d')!;
  const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const grays = toLuminance(img.data);
  let lo = 255, hi = 0;
  grays.forEach(g => { if (g < lo) lo = g; if (g > hi) hi = g; });
  const rng = hi - lo || 1;
  for (let i = 0, gi = 0; i < img.data.length; i += 4, gi++) {
    const v = Math.round(((grays[gi] - lo) / rng) * 255);
    img.data[i] = img.data[i + 1] = img.data[i + 2] = v;
    img.data[i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return canvas;
}

/** Otsu's global binarization */
function otsuThreshold(canvas: HTMLCanvasElement): HTMLCanvasElement {
  const ctx = canvas.getContext('2d')!;
  const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const grays = toLuminance(img.data);
  const hist = new Int32Array(256);
  grays.forEach(g => hist[Math.round(g)]++);
  const total = grays.length;
  let sum = 0;
  for (let i = 0; i < 256; i++) sum += i * hist[i];
  let sumB = 0, wB = 0, maxV = 0, thresh = 128;
  for (let t = 0; t < 256; t++) {
    wB += hist[t];
    if (!wB) continue;
    const wF = total - wB;
    if (!wF) break;
    sumB += t * hist[t];
    const mB = sumB / wB, mF = (sum - sumB) / wF;
    const v = wB * wF * (mB - mF) ** 2;
    if (v > maxV) { maxV = v; thresh = t; }
  }
  for (let i = 0, gi = 0; i < img.data.length; i += 4, gi++) {
    const v = grays[gi] >= thresh ? 255 : 0;
    img.data[i] = img.data[i + 1] = img.data[i + 2] = v;
    img.data[i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return canvas;
}

/** Contrast-stretch then invert (black chars on light plate → light on dark) */
function invertedContrast(canvas: HTMLCanvasElement): HTMLCanvasElement {
  const ctx = canvas.getContext('2d')!;
  const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const grays = toLuminance(img.data);
  let lo = 255, hi = 0;
  grays.forEach(g => { if (g < lo) lo = g; if (g > hi) hi = g; });
  const rng = hi - lo || 1;
  for (let i = 0, gi = 0; i < img.data.length; i += 4, gi++) {
    const v = 255 - Math.round(((grays[gi] - lo) / rng) * 255);
    img.data[i] = img.data[i + 1] = img.data[i + 2] = v;
    img.data[i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return canvas;
}

/** Simple grayscale mapping without forced thresholding/stretching */
function grayscale(canvas: HTMLCanvasElement): HTMLCanvasElement {
  const ctx = canvas.getContext('2d')!;
  const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const grays = toLuminance(img.data);
  for (let i = 0, gi = 0; i < img.data.length; i += 4, gi++) {
    img.data[i] = img.data[i + 1] = img.data[i + 2] = grays[gi];
    img.data[i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return canvas;
}

/** Variance of pixel luminance — used as sharpness/content proxy */
function pixelVariance(canvas: HTMLCanvasElement): number {
  const ctx = canvas.getContext('2d')!;
  const grays = toLuminance(ctx.getImageData(0, 0, canvas.width, canvas.height).data);
  const mean = grays.reduce((a, b) => a + b, 0) / grays.length;
  return grays.reduce((a, b) => a + (b - mean) ** 2, 0) / grays.length;
}

// ─── Plate Region Localization ────────────────────────────────────────────────

interface PlateRegion {
  sx: number; sy: number; sw: number; sh: number;
  label: string;
}

/**
 * Given a vehicle bounding box (from COCO-SSD) return several candidate plate
 * sub-regions to try.  We cover the four most common positions:
 *   • Front / bottom-centre (cars, sedans, SUVs)
 *   • Front / bottom-wide   (wider search for the above)
 *   • Top-mounted           (trucks, vans with number on front face)
 *   • Mid-centre            (some hatchbacks + rear-mounted plates)
 */
function getPlateRegions(
  vbbox: [number, number, number, number],
  frameW: number,
  frameH: number
): PlateRegion[] {
  const [vx, vy, vw, vh] = vbbox;
  const cl = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

  const raw: PlateRegion[] = [
    { sx: vx + vw * 0.10, sy: vy + vh * 0.62, sw: vw * 0.80, sh: vh * 0.30, label: 'front-bottom-center' },
    { sx: vx + vw * 0.02, sy: vy + vh * 0.55, sw: vw * 0.96, sh: vh * 0.42, label: 'front-bottom-wide' },
    { sx: vx + vw * 0.15, sy: vy + vh * 0.08, sw: vw * 0.70, sh: vh * 0.25, label: 'top-center' },
    { sx: vx + vw * 0.10, sy: vy + vh * 0.38, sw: vw * 0.80, sh: vh * 0.28, label: 'mid-center' },
  ];

  return raw
    .map(r => {
      const sx = cl(r.sx, 0, frameW - 1);
      const sy = cl(r.sy, 0, frameH - 1);
      return {
        sx, sy,
        sw: cl(r.sw, 1, frameW - sx),
        sh: cl(r.sh, 1, frameH - sy),
        label: r.label,
      };
    })
    .filter(r => r.sw >= 25 && r.sh >= 7);
}

/** Reject plate candidates that are clearly too small, wrong aspect ratio, or blank */
function isUsableRegion(sw: number, sh: number, variance: number): boolean {
  if (sw < 30 || sh < 8) return false;
  const ar = sw / sh;
  if (ar < 1.2 || ar > 12) return false;
  if (variance < 55) return false;   // nearly uniform → likely blank sky/ground
  return true;
}

// ─── Single-region OCR ────────────────────────────────────────────────────────

type Variant = 'grayscale' | 'contrast' | 'otsu' | 'inverted';
type PSM = '6' | '7' | '8' | '11';

const APPLY: Record<Variant, (c: HTMLCanvasElement) => HTMLCanvasElement> = {
  grayscale: grayscale,
  contrast: contrastStretch,
  otsu:     otsuThreshold,
  inverted: invertedContrast,
};

async function runOcr(
  scaledCanvas: HTMLCanvasElement,
  variant: Variant,
  psm: PSM,
  regionLabel: string
): Promise<OcrAttempt | null> {
  // Make a copy so we can apply the variant without mutating the shared scaled canvas
  const copy = cropAndScale(scaledCanvas, 0, 0, scaledCanvas.width, scaledCanvas.height, 1);
  APPLY[variant](copy);

  const worker = await getTessWorker();
  await worker.setParameters({ tessedit_pageseg_mode: psm as any });
  const { data } = await worker.recognize(copy);

  const text = data.text.trim().replace(/[^A-Z0-9]/g, '');
  if (text.length < 4) return null;

  return {
    text,
    confidence: data.confidence / 100,
    variant,
    psm,
    regionLabel,
  };
}

// ─── Per-frame Analysis ───────────────────────────────────────────────────────

/**
 * Analyse one video frame.
 *
 * Steps:
 *  1.  Draw frame to canvas
 *  2.  Run COCO-SSD on the video element → vehicle bounding boxes
 *  3.  For each vehicle, try 4 plate-region candidates
 *  4.  Reject regions that fail quality gate
 *  5.  For accepted regions: 3 preprocessing variants × 2 PSMs → up to 6 OCR calls
 *  6.  Stop early for this vehicle if a high-confidence STRICT plate is found
 *  7.  If zero vehicles detected, fall back to 3×3 grid scan (full frame)
 *  8.  Return frame result with all attempts + best candidate + composite quality
 */
export async function analyzeFrame(
  detector: any,
  video: HTMLVideoElement,
  timestamp: number,
  onLog: (msg: string) => void
): Promise<FrameResult> {
  const W = video.videoWidth  || 640;
  const H = video.videoHeight || 480;

  // Render current frame to an offscreen canvas (needed for cropping)
  const fc = document.createElement('canvas');
  fc.width = W; fc.height = H;
  fc.getContext('2d')!.drawImage(video, 0, 0, W, H);

  // ── Stage 1: Vehicle Detection ──────────────────────────────────────────────
  const rawPreds: any[] = await detector.detect(video);
  const vehicles: VehicleBox[] = rawPreds
    .filter((p: any) => VEHICLE_CLASSES.has(p.class) && p.score >= 0.25)
    .sort((a: any, b: any) => b.score - a.score);   // highest-confidence first

  onLog(`Frame @${timestamp.toFixed(1)}s — ${vehicles.length} vehicle(s): ${
    vehicles.map(v => `${v.class}@${(v.score * 100).toFixed(0)}%`).join(', ') || 'none'
  }`);

  const allAttempts: OcrAttempt[] = [];
  let bestPlateBbox: [number, number, number, number] | null = null;

  const VARIANTS: Variant[] = ['grayscale', 'contrast', 'otsu', 'inverted'];
  const PSMS: PSM[]          = ['11', '6', '7', '8'];

  // ── Stage 2–4: Plate Localization + Preprocessing + OCR ────────────────────
  for (const vehicle of vehicles) {
    const regions = getPlateRegions(vehicle.bbox, W, H);
    let vehiclePlateFound = false;

    for (const region of regions) {
      // Quality gate on raw (unscaled) crop
      const probe   = cropAndScale(fc, region.sx, region.sy, region.sw, region.sh, 1);
      const variance = pixelVariance(probe);

      if (!isUsableRegion(region.sw, region.sh, variance)) {
        onLog(`  [${region.label}] ${Math.round(region.sw)}×${Math.round(region.sh)} — rejected (var=${variance.toFixed(0)})`);
        continue;
      }

      onLog(`  [${region.label}] ${Math.round(region.sw)}×${Math.round(region.sh)} px (var=${variance.toFixed(0)})`);

      // Dynamic scale: target ~100px height for optimal Tesseract performance (usually chars ~30px)
      const scale = Math.max(1, Math.min(4, 100 / region.sh));
      const scaled = cropAndScale(fc, region.sx, region.sy, region.sw, region.sh, scale);

      for (const variant of VARIANTS) {
        if (vehiclePlateFound) break;
        for (const psm of PSMS) {
          const attempt = await runOcr(scaled, variant, psm, region.label);
          if (!attempt) continue;

          onLog(`    [${variant}/PSM${psm}] → "${attempt.text}" (${(attempt.confidence * 100).toFixed(0)}%)`);
          allAttempts.push(attempt);

          if (STRICT_RE.test(attempt.text) && attempt.confidence >= 0.38) {
            bestPlateBbox = [region.sx, region.sy, region.sw, region.sh];
            vehiclePlateFound = true;
            break;
          }
        }
      }

      if (vehiclePlateFound) break;
    }
  }

  // ── Stage 2 fallback: No vehicles detected ──────────────────────────────────
  // Scan a fine 3×3 grid instead of arbitrary region heuristics.
  if (vehicles.length === 0) {
    onLog('  No vehicles — fallback: 3×3 grid scan');
    for (let row = 0; row < 3; row++) {
      for (let col = 0; col < 3; col++) {
        const sx = Math.floor(W * col / 3);
        const sy = Math.floor(H * row / 3);
        const sw = Math.floor(W / 3);
        const sh = Math.floor(H / 3);
        const scaled = cropAndScale(fc, sx, sy, sw, sh, 2);
        // Use PSM 11 (sparse text) for fallback — more forgiving
        const worker = await getTessWorker();
        await worker.setParameters({
          tessedit_pageseg_mode: '11' as any,
          tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        });
        contrastStretch(scaled);
        const { data } = await worker.recognize(scaled);
        const text = data.text.trim().replace(/[^A-Z0-9]/g, '');
        if (text.length >= 6 && LOOSE_RE.test(text)) {
          allAttempts.push({ text, confidence: data.confidence / 100, variant: 'contrast', psm: '11', regionLabel: `grid-${row}-${col}` });
          onLog(`  Grid [${row},${col}] → "${text}"`);
        }
      }
    }
  }

  // ── Select best candidate from this frame ────────────────────────────────────
  const strictHits = allAttempts.filter(a => STRICT_RE.test(a.text));
  const looseHits  = allAttempts.filter(a => LOOSE_RE.test(a.text));
  const pool       = strictHits.length > 0 ? strictHits : looseHits;
  const best       = pool.sort((a, b) => b.confidence - a.confidence)[0] ?? null;

  // Composite quality score 0–1
  const quality =
    (vehicles.length > 0 ? 0.35 : 0.05) +
    (bestPlateBbox    !== null ? 0.30 : 0.00) +
    (best !== null ? Math.min(best.confidence, 1) * 0.35 : 0);

  return {
    timestamp,
    vehicles,
    plateBbox:   bestPlateBbox,
    ocrAttempts: allAttempts,
    bestText:    best?.text ?? null,
    quality,
  };
}

// ─── Multi-Frame Fusion ───────────────────────────────────────────────────────

function editDistance(a: string, b: string): number {
  const m = a.length, n = b.length;
  const dp = Array.from({ length: m + 1 }, (_, i) =>
    Array.from({ length: n + 1 }, (_, j) => (i === 0 ? j : j === 0 ? i : 0))
  );
  for (let i = 1; i <= m; i++)
    for (let j = 1; j <= n; j++)
      dp[i][j] =
        a[i - 1] === b[j - 1]
          ? dp[i - 1][j - 1]
          : 1 + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
  return dp[m][n];
}

/**
 * Multi-frame fusion with character-level vote scoring.
 *
 * Algorithm:
 *  1.  Collect all per-frame bestText values
 *  2.  Group by edit-distance ≤ 2 (handles 1–2 character OCR errors)
 *  3.  Score each group: vote_count × average_quality
 *  4.  Pick the highest-scoring group
 *  5.  Determine certainty: strict-RE match + (agreement ≥ 30% OR avg quality ≥ 55%)
 *  6.  Return PLATE_UNCERTAIN if confidence is insufficient
 */
export function fuseFrameResults(frames: FrameResult[]): PipelineOutput {
  const log: string[] = [];

  const candidates = frames.flatMap(f =>
    f.bestText ? [{ text: f.bestText, quality: f.quality }] : []
  );

  log.push(`Frames: ${frames.length} total, ${candidates.length} with a plate candidate`);

  if (candidates.length === 0) {
    log.push('No candidates — returning PLATE_UNCERTAIN');
    return { plate: 'PLATE_UNCERTAIN', confidence: 0, frameAgreement: 0, isPlateCertain: false, vehicleType: null, frames, debugLog: log };
  }

  // Group similar strings together
  type Group = { plate: string; totalQuality: number; count: number };
  const groups: Group[] = [];

  for (const { text, quality } of candidates) {
    const existing = groups.find(g => editDistance(g.plate, text) <= 2);
    if (existing) {
      existing.count++;
      existing.totalQuality += quality;
      // Keep the highest-quality individual result as the group's representative plate
      if (quality > existing.totalQuality / existing.count) {
        existing.plate = text;
      }
    } else {
      groups.push({ plate: text, totalQuality: quality, count: 1 });
    }
  }

  // Sort: most votes first, break ties by average quality
  groups.sort((a, b) =>
    (b.count * (b.totalQuality / b.count)) - (a.count * (a.totalQuality / a.count))
  );

  const best      = groups[0];
  const avgQual   = best.totalQuality / best.count;
  const agreement = best.count / Math.max(frames.length, 1);

  const vehicleCounts = new Map<string, number>();
  for (const frame of frames) {
    if (frame.vehicles.length > 0) {
       const vClass = frame.vehicles[0].class;
       vehicleCounts.set(vClass, (vehicleCounts.get(vClass) || 0) + 1);
    }
  }
  let dominantVehicleType: string | null = null;
  let maxCount = 0;
  for (const [vClass, count] of vehicleCounts.entries()) {
     if (count > maxCount) {
        dominantVehicleType = vClass;
        maxCount = count;
     }
  }

  log.push(`Top group: "${best.plate}" — ${best.count}/${frames.length} frames, avg quality ${(avgQual * 100).toFixed(0)}%`);
  log.push(`Frame agreement: ${(agreement * 100).toFixed(0)}%`);
  log.push(`All groups: ${groups.map(g => `"${g.plate}"×${g.count}`).join(', ')}`);

  const isPlateCertain =
    STRICT_RE.test(best.plate) && (agreement >= 0.30 || avgQual >= 0.55);

  if (!isPlateCertain) {
    log.push(`Certainty FAILED — strict=${STRICT_RE.test(best.plate)}, agreement=${(agreement*100).toFixed(0)}%, avgQual=${(avgQual*100).toFixed(0)}%`);
  }

  return {
    plate:         best.plate,
    confidence:    avgQual,
    frameAgreement: agreement,
    isPlateCertain,
    vehicleType:   dominantVehicleType,
    frames,
    debugLog: log,
  };
}
