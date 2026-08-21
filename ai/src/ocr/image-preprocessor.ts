import { 
  IPreprocessor, 
  PreprocessOutput, 
  PreprocessVariant,
  PlateQualityMetrics,
  PerspectiveConfig
} from '../core/interfaces';

export class ImagePreprocessor implements IPreprocessor {
  // Step 13: Performance optimization - adjust these for your target performance
  private readonly MAX_UPSCALE = 5.0;
  private readonly MIN_UPSCALE = 2.0;
  private readonly TARGET_HEIGHT = 100; // optimal for OCR
  private readonly SR_THRESHOLD_PX = 40; // if plate height < 40px, apply super-resolution
  private readonly SR_TARGET_HEIGHT = 120; // super-resolution target

  preprocess(plateCanvas: HTMLCanvasElement): HTMLCanvasElement[] {
    const detailed = this.preprocessDetailed(plateCanvas);
    return detailed.variants.map(v => v.canvas);
  }

  preprocessDetailed(plateCanvas: HTMLCanvasElement): PreprocessOutput {
    const w = plateCanvas.width;
    const h = plateCanvas.height;
    const aspectRatio = h > 0 ? parseFloat((w / h).toFixed(2)) : 0;

    // ─────────────────────────────────────────────────────────────
    // STEP 1: PLATE QUALITY CHECK
    // ─────────────────────────────────────────────────────────────
    const qualityMetrics = this.computePlateQualityMetrics(plateCanvas);

    // Reject only completely unusable crops
    if (!qualityMetrics.isUsable) {
      return {
        isValid: false,
        cropWidth: w,
        cropHeight: h,
        aspectRatio,
        qualityMetrics,
        variants: [],
        rejectionReason: qualityMetrics.qualityReason
      };
    }

    // ─────────────────────────────────────────────────────────────
    // STEP 2: PERSPECTIVE RECTIFICATION
    // ─────────────────────────────────────────────────────────────
    const perspectiveInfo = this.estimatePerspectiveCorrection(plateCanvas);
    let rectifiedCanvas = plateCanvas;

    if (perspectiveInfo.isAngled && Math.abs(perspectiveInfo.rotationAngleDeg) <= perspectiveInfo.maxCorrectionAngle) {
      rectifiedCanvas = this.applyPerspectiveCorrection(plateCanvas, perspectiveInfo);
    }

    // ─────────────────────────────────────────────────────────────
    // STEP 3: SUPER RESOLUTION (if needed)
    // ─────────────────────────────────────────────────────────────
    let srCanvas: HTMLCanvasElement | undefined;
    let baseCanvas = rectifiedCanvas;

    if (h < this.SR_THRESHOLD_PX) {
      srCanvas = this.applySuperResolution(rectifiedCanvas);
      baseCanvas = srCanvas;
    }

    // ─────────────────────────────────────────────────────────────
    // Scale base canvas to target height for optimal OCR
    // ─────────────────────────────────────────────────────────────
    const scale = Math.max(this.MIN_UPSCALE, Math.min(this.MAX_UPSCALE, this.TARGET_HEIGHT / baseCanvas.height));
    const scaledCanvas = this.scaleCanvas(baseCanvas, scale);

    // ─────────────────────────────────────────────────────────────
    // STEP 4: GENERATE 8 PREPROCESSING VARIANTS (A-H)
    // ─────────────────────────────────────────────────────────────
    const varA = this.variantA(this.cloneCanvas(scaledCanvas));
    const varB = this.variantB(this.cloneCanvas(scaledCanvas));
    const varC = this.variantC(this.cloneCanvas(scaledCanvas));
    const varD = this.variantD(this.cloneCanvas(scaledCanvas));
    const varE = this.variantE(this.cloneCanvas(scaledCanvas));
    const varF = this.variantF(this.cloneCanvas(scaledCanvas)); // NEW
    const varG = this.variantG(this.cloneCanvas(scaledCanvas)); // NEW
    const varH = this.variantH(this.cloneCanvas(scaledCanvas)); // NEW
    // Variant I: motion-deblur — aggressive unsharp mask for moving vehicle plates
    const varI = this.variantI(this.cloneCanvas(scaledCanvas));

    const variants: PreprocessVariant[] = [
      { id: 'A', name: 'Grayscale + Contrast + Denoise', canvas: varA },
      { id: 'B', name: 'CLAHE + Sharpened', canvas: varB },
      { id: 'C', name: 'Otsu Binarization', canvas: varC },
      { id: 'D', name: 'Adaptive Local Threshold', canvas: varD },
      { id: 'E', name: 'Sharpen + Inverted Binarization', canvas: varE },
      { id: 'F', name: 'Bilateral Filter + Morphology Open', canvas: varF },
      { id: 'G', name: 'Morphology Close + Sharpen', canvas: varG },
      { id: 'H', name: 'Edge-Preserving Denoise', canvas: varH },
      { id: 'I', name: 'Motion-Deblur Unsharp Mask', canvas: varI },
    ];

    // Debug: Async send all variants to backend to save and test PaddleOCR
    setTimeout(() => {
      variants.forEach(async (v) => {
        try {
          const dataUrl = v.canvas.toDataURL('image/png');
          await fetch(`http://localhost:5001/api/debug/save_variant?variant_id=${v.id}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: dataUrl })
          });
        } catch (e) {
          console.warn('[DEBUG] Failed to save variant', v.id, e);
        }
      });
    }, 0);

    return {
      isValid: true,
      cropWidth: w,
      cropHeight: h,
      aspectRatio,
      qualityMetrics,
      variants,
      rectifiedCanvas,
      superResCanvas: srCanvas
    };
  }

  // ──────────────────────────────────────────────────────────────
  // STEP 1: PLATE QUALITY METRICS
  // ──────────────────────────────────────────────────────────────

  private computePlateQualityMetrics(canvas: HTMLCanvasElement): PlateQualityMetrics {
    const w = canvas.width;
    const h = canvas.height;
    const area = w * h;
    const aspectRatio = h > 0 ? w / h : 0;

    const ctx = canvas.getContext('2d')!;
    const img = ctx.getImageData(0, 0, w, h);
    const data = img.data;

    // Calculate brightness
    let sumBrightness = 0;
    const pixelCount = w * h;
    for (let i = 0; i < data.length; i += 4) {
      sumBrightness += 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    }
    const brightness = Math.round(sumBrightness / pixelCount);

    // Calculate contrast (standard deviation of luminance)
    let sumSq = 0;
    for (let i = 0; i < data.length; i += 4) {
      const lum = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
      sumSq += (lum - brightness) ** 2;
    }
    const contrast = Math.round(Math.sqrt(sumSq / pixelCount));

    // Calculate blur score using Laplacian variance
    const blurScore = this.calculateBlurScore(canvas);

    // Determine if usable
    let isUsable = true;
    let qualityReason = '';

    if (w < 30 || h < 12 || area < 360) {
      isUsable = false;
      qualityReason = `Crop too small (${w}x${h}, area=${area}px) — min 30x12px required`;
    } else if (aspectRatio < 1.1 || aspectRatio > 7.5) {
      isUsable = false;
      qualityReason = `Invalid aspect ratio (${aspectRatio})`;
    } else if (blurScore < 5) {
      // Near-totally blank — almost certainly sky/ground, not a plate
      // NOTE: motion-blurred plates typically score 5–15; we still attempt OCR on those
      isUsable = false;
      qualityReason = `Image too blurry (blur score: ${blurScore.toFixed(2)})`;
    } else if (brightness < 30 || brightness > 220) {
      // Extreme brightness, but continue processing
      qualityReason = `Extreme brightness: ${brightness}`;
    }

    return {
      width: w,
      height: h,
      area,
      aspectRatio: parseFloat(aspectRatio.toFixed(2)),
      blurScore: parseFloat(blurScore.toFixed(2)),
      brightness,
      contrast,
      isUsable,
      qualityReason: qualityReason || undefined
    };
  }

  private calculateBlurScore(canvas: HTMLCanvasElement): number {
    const w = canvas.width;
    const h = canvas.height;
    const ctx = canvas.getContext('2d')!;
    const img = ctx.getImageData(0, 0, w, h);
    const data = img.data;

    // Convert to grayscale
    const gray = new Uint8Array(w * h);
    for (let i = 0, j = 0; i < data.length; i += 4, j++) {
      gray[j] = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
    }

    // Apply Laplacian kernel and compute variance
    let sumVar = 0, count = 0;
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const laplacian =
          -1 * gray[(y - 1) * w + (x - 1)] + -1 * gray[(y - 1) * w + x] + -1 * gray[(y - 1) * w + (x + 1)] +
          -1 * gray[y * w + (x - 1)] + 8 * gray[y * w + x] + -1 * gray[y * w + (x + 1)] +
          -1 * gray[(y + 1) * w + (x - 1)] + -1 * gray[(y + 1) * w + x] + -1 * gray[(y + 1) * w + (x + 1)];
        sumVar += laplacian ** 2;
        count++;
      }
    }

    return count > 0 ? sumVar / count : 0;
  }

  // ──────────────────────────────────────────────────────────────
  // STEP 2: PERSPECTIVE RECTIFICATION (±20°)
  // ──────────────────────────────────────────────────────────────

  private estimatePerspectiveCorrection(canvas: HTMLCanvasElement): PerspectiveConfig {
    const w = canvas.width;
    const h = canvas.height;
    const ctx = canvas.getContext('2d')!;
    const img = ctx.getImageData(0, 0, w, h);
    const data = img.data;

    // Detect plate orientation using horizontal gradients
    let sumX = 0, sumY = 0, sumXX = 0, sumYY = 0, sumXY = 0, totalWeight = 0;

    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const idx = (y * w + x) * 4;
        const gx = (data[idx + 4] || 0) - (data[idx - 4] || 0);
        const gy = (data[idx + w * 4] || 0) - (data[idx - w * 4] || 0);
        const weight = Math.sqrt(gx * gx + gy * gy);

        if (weight > 30) {
          sumX += x * weight;
          sumY += y * weight;
          sumXX += x * x * weight;
          sumYY += y * y * weight;
          sumXY += x * y * weight;
          totalWeight += weight;
        }
      }
    }

    let rotationAngleDeg = 0;
    let isAngled = false;

    if (totalWeight > 100) {
      const meanX = sumX / totalWeight;
      const meanY = sumY / totalWeight;
      const covXY = (sumXY / totalWeight) - (meanX * meanY);
      const varX = (sumXX / totalWeight) - (meanX * meanX);
      const varY = (sumYY / totalWeight) - (meanY * meanY);

      rotationAngleDeg = 0.5 * Math.atan2(2 * covXY, varX - varY) * (180 / Math.PI);
      isAngled = Math.abs(rotationAngleDeg) > 1.5;
    }

    return {
      rotationAngleDeg,
      isAngled,
      maxCorrectionAngle: 20 // Max ±20°
    };
  }

  private applyPerspectiveCorrection(canvas: HTMLCanvasElement, config: PerspectiveConfig): HTMLCanvasElement {
    const w = canvas.width;
    const h = canvas.height;
    const rad = config.rotationAngleDeg * Math.PI / 180;

    const out = document.createElement('canvas');
    out.width = w;
    out.height = h;
    const octx = out.getContext('2d')!;

    octx.translate(w / 2, h / 2);
    octx.rotate(-rad);
    octx.drawImage(canvas, -w / 2, -h / 2);

    return out;
  }

  // ──────────────────────────────────────────────────────────────
  // STEP 3: SUPER RESOLUTION (if height < 40px)
  // ──────────────────────────────────────────────────────────────

  private applySuperResolution(canvas: HTMLCanvasElement): HTMLCanvasElement {
    // Use bicubic interpolation (fallback approach in browser)
    // In production, you would call a Python service or use Real-ESRGAN
    const targetHeight = this.SR_TARGET_HEIGHT;
    const scale = targetHeight / canvas.height;
    return this.scaleCanvas(canvas, Math.min(scale, 3.0)); // Cap at 3x to avoid artifacts
  }

  // ──────────────────────────────────────────────────────────────
  // STEP 4: 8 PREPROCESSING VARIANTS (A-H)
  // ──────────────────────────────────────────────────────────────

  // Variant A: Grayscale + Contrast Stretch + Mild Denoise
  private variantA(canvas: HTMLCanvasElement): HTMLCanvasElement {
    const ctx = canvas.getContext('2d')!;
    const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = img.data;

    let minL = 255, maxL = 0;
    const lum = new Float32Array(data.length / 4);

    for (let i = 0, j = 0; i < data.length; i += 4, j++) {
      const l = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
      lum[j] = l;
      if (l < minL) minL = l;
      if (l > maxL) maxL = l;
    }

    const range = (maxL - minL) || 1;
    const w = canvas.width;
    const h = canvas.height;
    const smooth = new Float32Array(lum.length);

    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const idx = y * w + x;
        const val = (
          lum[idx - w - 1] + 2 * lum[idx - w] + lum[idx - w + 1] +
          2 * lum[idx - 1] + 4 * lum[idx] + 2 * lum[idx + 1] +
          lum[idx + w - 1] + 2 * lum[idx + w] + lum[idx + w + 1]
        ) / 16.0;
        smooth[idx] = val;
      }
    }

    for (let i = 0, j = 0; i < data.length; i += 4, j++) {
      const val = Math.min(255, Math.max(0, Math.round(((smooth[j] || lum[j]) - minL) * 255 / range)));
      data[i] = data[i + 1] = data[i + 2] = val;
      data[i + 3] = 255;
    }

    ctx.putImageData(img, 0, 0);
    return canvas;
  }

  // Variant B: CLAHE + Sharpen
  private variantB(canvas: HTMLCanvasElement): HTMLCanvasElement {
    const ctx = canvas.getContext('2d')!;
    const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = img.data;
    const w = canvas.width;
    const h = canvas.height;

    const lum = new Uint8Array(w * h);
    for (let i = 0, j = 0; i < data.length; i += 4, j++) {
      lum[j] = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
    }

    const gridX = 4, gridY = 4;
    const tileW = Math.ceil(w / gridX);
    const tileH = Math.ceil(h / gridY);
    const clipLimit = 3.0;
    const equalized = new Uint8Array(w * h);

    for (let ty = 0; ty < gridY; ty++) {
      for (let tx = 0; tx < gridX; tx++) {
        const x0 = tx * tileW, x1 = Math.min(w, (tx + 1) * tileW);
        const y0 = ty * tileH, y1 = Math.min(h, (ty + 1) * tileH);
        const hist = new Int32Array(256);
        let count = 0;

        for (let y = y0; y < y1; y++) {
          for (let x = x0; x < x1; x++) {
            hist[lum[y * w + x]]++;
            count++;
          }
        }

        const limit = Math.max(1, Math.round(clipLimit * count / 256));
        let excess = 0;
        for (let i = 0; i < 256; i++) {
          if (hist[i] > limit) {
            excess += hist[i] - limit;
            hist[i] = limit;
          }
        }
        const bonus = Math.floor(excess / 256);
        for (let i = 0; i < 256; i++) hist[i] += bonus;

        const cdf = new Float32Array(256);
        let sum = 0;
        for (let i = 0; i < 256; i++) {
          sum += hist[i];
          cdf[i] = (sum / count) * 255;
        }

        for (let y = y0; y < y1; y++) {
          for (let x = x0; x < x1; x++) {
            equalized[y * w + x] = Math.min(255, Math.max(0, Math.round(cdf[lum[y * w + x]])));
          }
        }
      }
    }

    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const idx = y * w + x;
        const sharp = 5 * equalized[idx] - equalized[idx - w] - equalized[idx + w] - equalized[idx - 1] - equalized[idx + 1];
        const val = Math.min(255, Math.max(0, sharp));
        const px = idx * 4;
        data[px] = data[px + 1] = data[px + 2] = val;
        data[px + 3] = 255;
      }
    }

    ctx.putImageData(img, 0, 0);
    return canvas;
  }

  // Variant C: Grayscale + Otsu Binarization
  private variantC(canvas: HTMLCanvasElement): HTMLCanvasElement {
    const ctx = canvas.getContext('2d')!;
    const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = img.data;

    const lum = new Uint8Array(data.length / 4);
    const hist = new Int32Array(256);

    for (let i = 0, j = 0; i < data.length; i += 4, j++) {
      const v = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
      lum[j] = v;
      hist[v]++;
    }

    const total = lum.length;
    let sum = 0;
    for (let t = 0; t < 256; t++) sum += t * hist[t];

    let sumB = 0, wB = 0, maxVar = 0, otsuThresh = 128;
    for (let t = 0; t < 256; t++) {
      wB += hist[t];
      if (!wB) continue;
      const wF = total - wB;
      if (!wF) break;
      sumB += t * hist[t];
      const mB = sumB / wB;
      const mF = (sum - sumB) / wF;
      const varBetween = wB * wF * (mB - mF) ** 2;
      if (varBetween > maxVar) {
        maxVar = varBetween;
        otsuThresh = t;
      }
    }

    for (let i = 0, j = 0; i < data.length; i += 4, j++) {
      const val = lum[j] >= otsuThresh ? 255 : 0;
      data[i] = data[i + 1] = data[i + 2] = val;
      data[i + 3] = 255;
    }

    ctx.putImageData(img, 0, 0);
    return canvas;
  }

  // Variant D: Adaptive Local Threshold
  private variantD(canvas: HTMLCanvasElement): HTMLCanvasElement {
    const ctx = canvas.getContext('2d')!;
    const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = img.data;
    const w = canvas.width;
    const h = canvas.height;

    const lum = new Uint8Array(w * h);
    for (let i = 0, j = 0; i < data.length; i += 4, j++) {
      lum[j] = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
    }

    const win = 7;
    const C = 7;

    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let sum = 0, cnt = 0;
        for (let dy = -win; dy <= win; dy++) {
          const py = y + dy;
          if (py < 0 || py >= h) continue;
          for (let dx = -win; dx <= win; dx++) {
            const px = x + dx;
            if (px < 0 || px >= w) continue;
            sum += lum[py * w + px];
            cnt++;
          }
        }
        const mean = sum / cnt;
        const val = lum[y * w + x] >= (mean - C) ? 255 : 0;
        const i = (y * w + x) * 4;
        data[i] = data[i + 1] = data[i + 2] = val;
        data[i + 3] = 255;
      }
    }

    ctx.putImageData(img, 0, 0);
    return canvas;
  }

  // Variant E: Sharpen + Inverted Binarization
  private variantE(canvas: HTMLCanvasElement): HTMLCanvasElement {
    const ctx = canvas.getContext('2d')!;
    const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = img.data;

    let minL = 255, maxL = 0;
    const lum = new Uint8Array(data.length / 4);

    for (let i = 0, j = 0; i < data.length; i += 4, j++) {
      const v = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
      lum[j] = v;
      if (v < minL) minL = v;
      if (v > maxL) maxL = v;
    }

    const range = (maxL - minL) || 1;
    for (let i = 0, j = 0; i < data.length; i += 4, j++) {
      const norm = ((lum[j] - minL) / range) * 255;
      const inv = 255 - norm;
      const val = inv >= 120 ? 255 : 0;
      data[i] = data[i + 1] = data[i + 2] = val;
      data[i + 3] = 255;
    }

    ctx.putImageData(img, 0, 0);
    return canvas;
  }

  // Variant F: Bilateral-like Filter + Morphology Open
  private variantF(canvas: HTMLCanvasElement): HTMLCanvasElement {
    const ctx = canvas.getContext('2d')!;
    const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = img.data;
    const w = canvas.width;
    const h = canvas.height;

    // Convert to grayscale
    const lum = new Uint8Array(w * h);
    for (let i = 0, j = 0; i < data.length; i += 4, j++) {
      lum[j] = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
    }

    // Simple bilateral filter (edge-preserving smoothing)
    const sigmaSpatial = 3.0, sigmaRange = 50.0;
    const bilateral = new Uint8Array(w * h);

    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let weightSum = 0, valueSum = 0;
        const centerVal = lum[y * w + x];

        for (let dy = -2; dy <= 2; dy++) {
          for (let dx = -2; dx <= 2; dx++) {
            const py = Math.min(h - 1, Math.max(0, y + dy));
            const px = Math.min(w - 1, Math.max(0, x + dx));
            const neighborVal = lum[py * w + px];

            const spatialDist = Math.sqrt(dx * dx + dy * dy);
            const rangeDist = Math.abs(neighborVal - centerVal);

            const wSpatial = Math.exp(-(spatialDist ** 2) / (2 * sigmaSpatial ** 2));
            const wRange = Math.exp(-(rangeDist ** 2) / (2 * sigmaRange ** 2));
            const weight = wSpatial * wRange;

            weightSum += weight;
            valueSum += neighborVal * weight;
          }
        }

        bilateral[y * w + x] = Math.round(valueSum / weightSum);
      }
    }

    // Morphology: Open (erosion followed by dilation)
    const kernel = [[0, 1, 0], [1, 1, 1], [0, 1, 0]];
    const opened = this.applyMorphologyOpen(bilateral, w, h, kernel);

    // Apply Otsu threshold and write back
    const hist = new Int32Array(256);
    for (let i = 0; i < opened.length; i++) hist[opened[i]]++;

    const total = opened.length;
    let sum = 0;
    for (let t = 0; t < 256; t++) sum += t * hist[t];

    let sumB = 0, wB = 0, maxVar = 0, thresh = 128;
    for (let t = 0; t < 256; t++) {
      wB += hist[t];
      if (!wB) continue;
      const wF = total - wB;
      if (!wF) break;
      sumB += t * hist[t];
      const mB = sumB / wB;
      const mF = (sum - sumB) / wF;
      const varBetween = wB * wF * (mB - mF) ** 2;
      if (varBetween > maxVar) {
        maxVar = varBetween;
        thresh = t;
      }
    }

    for (let i = 0, j = 0; i < data.length; i += 4, j++) {
      const val = opened[j] >= thresh ? 255 : 0;
      data[i] = data[i + 1] = data[i + 2] = val;
      data[i + 3] = 255;
    }

    ctx.putImageData(img, 0, 0);
    return canvas;
  }

  // Variant G: Morphology Close + Sharpen
  private variantG(canvas: HTMLCanvasElement): HTMLCanvasElement {
    const ctx = canvas.getContext('2d')!;
    const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = img.data;
    const w = canvas.width;
    const h = canvas.height;

    // Convert to grayscale and binarize
    const lum = new Uint8Array(w * h);
    for (let i = 0, j = 0; i < data.length; i += 4, j++) {
      lum[j] = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
    }

    // Otsu threshold
    const hist = new Int32Array(256);
    for (let i = 0; i < lum.length; i++) hist[lum[i]]++;

    const total = lum.length;
    let sum = 0;
    for (let t = 0; t < 256; t++) sum += t * hist[t];

    let sumB = 0, wB = 0, maxVar = 0, otsuThresh = 128;
    for (let t = 0; t < 256; t++) {
      wB += hist[t];
      if (!wB) continue;
      const wF = total - wB;
      if (!wF) break;
      sumB += t * hist[t];
      const mB = sumB / wB;
      const mF = (sum - sumB) / wF;
      const varBetween = wB * wF * (mB - mF) ** 2;
      if (varBetween > maxVar) {
        maxVar = varBetween;
        otsuThresh = t;
      }
    }

    const bin = new Uint8Array(w * h);
    for (let i = 0; i < lum.length; i++) {
      bin[i] = lum[i] >= otsuThresh ? 255 : 0;
    }

    // Morphology: Close (dilation followed by erosion)
    const kernel = [[1, 1, 1], [1, 1, 1], [1, 1, 1]];
    const closed = this.applyMorphologyClose(bin, w, h, kernel);

    // Sharpen
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const idx = y * w + x;
        const sharp = 5 * closed[idx] - closed[idx - w] - closed[idx + w] - closed[idx - 1] - closed[idx + 1];
        const val = Math.min(255, Math.max(0, sharp));
        const px = idx * 4;
        data[px] = data[px + 1] = data[px + 2] = val;
        data[px + 3] = 255;
      }
    }

    ctx.putImageData(img, 0, 0);
    return canvas;
  }

  // Variant H: Edge-Preserving Denoise
  private variantH(canvas: HTMLCanvasElement): HTMLCanvasElement {
    const ctx = canvas.getContext('2d')!;
    const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = img.data;
    const w = canvas.width;
    const h = canvas.height;

    // Convert to grayscale
    const lum = new Uint8Array(w * h);
    for (let i = 0, j = 0; i < data.length; i += 4, j++) {
      lum[j] = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
    }

    // Median filter (edge-preserving)
    const radius = 1;
    const denoised = new Uint8Array(w * h);

    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const values = [];
        for (let dy = -radius; dy <= radius; dy++) {
          for (let dx = -radius; dx <= radius; dx++) {
            const py = Math.min(h - 1, Math.max(0, y + dy));
            const px = Math.min(w - 1, Math.max(0, x + dx));
            values.push(lum[py * w + px]);
          }
        }
        values.sort((a, b) => a - b);
        denoised[y * w + x] = values[Math.floor(values.length / 2)];
      }
    }

    // Contrast stretching
    let minL = 255, maxL = 0;
    for (let i = 0; i < denoised.length; i++) {
      if (denoised[i] < minL) minL = denoised[i];
      if (denoised[i] > maxL) maxL = denoised[i];
    }

    const range = (maxL - minL) || 1;
    for (let i = 0, j = 0; i < data.length; i += 4, j++) {
      const val = Math.min(255, Math.max(0, Math.round(((denoised[j] - minL) * 255) / range)));
      data[i] = data[i + 1] = data[i + 2] = val;
      data[i + 3] = 255;
    }

    ctx.putImageData(img, 0, 0);
    return canvas;
  }

  // Variant I: Motion-Deblur — aggressive unsharp masking tuned for motion-blurred plates
  // Works by: grayscale → gaussian blur → subtract blur from original (unsharp mask with
  // a high amount factor) → contrast stretch → Otsu binarize.
  // This recovers character edges even when blur score is 5–15.
  private variantI(canvas: HTMLCanvasElement): HTMLCanvasElement {
    const ctx = canvas.getContext('2d')!;
    const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = img.data;
    const w = canvas.width;
    const h = canvas.height;

    // Step 1: Grayscale
    const lum = new Float32Array(w * h);
    for (let i = 0, j = 0; i < data.length; i += 4, j++) {
      lum[j] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    }

    // Step 2: Gaussian blur (radius 2, sigma 1.2) to form the "blurred" reference
    const blurred = new Float32Array(w * h);
    const kernel = [0.0625, 0.25, 0.375, 0.25, 0.0625]; // 1D Gaussian
    // Horizontal pass
    const tmp = new Float32Array(w * h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let sum = 0, wSum = 0;
        for (let k = -2; k <= 2; k++) {
          const px = Math.min(w - 1, Math.max(0, x + k));
          sum += lum[y * w + px] * kernel[k + 2];
          wSum += kernel[k + 2];
        }
        tmp[y * w + x] = sum / wSum;
      }
    }
    // Vertical pass
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let sum = 0, wSum = 0;
        for (let k = -2; k <= 2; k++) {
          const py = Math.min(h - 1, Math.max(0, y + k));
          sum += tmp[py * w + x] * kernel[k + 2];
          wSum += kernel[k + 2];
        }
        blurred[y * w + x] = sum / wSum;
      }
    }

    // Step 3: Unsharp mask — amount=3.0 (strong, recovers motion-blurred edges)
    const amount = 3.0;
    const unsharp = new Float32Array(w * h);
    for (let i = 0; i < w * h; i++) {
      unsharp[i] = lum[i] + amount * (lum[i] - blurred[i]);
    }

    // Step 4: Contrast stretch on the unsharp result
    let minU = Infinity, maxU = -Infinity;
    for (let i = 0; i < unsharp.length; i++) {
      if (unsharp[i] < minU) minU = unsharp[i];
      if (unsharp[i] > maxU) maxU = unsharp[i];
    }
    const rangeU = (maxU - minU) || 1;
    const stretched = new Uint8Array(w * h);
    for (let i = 0; i < unsharp.length; i++) {
      stretched[i] = Math.min(255, Math.max(0, Math.round(((unsharp[i] - minU) / rangeU) * 255)));
    }

    // Step 5: Otsu binarize the sharpened result
    const hist = new Int32Array(256);
    for (let i = 0; i < stretched.length; i++) hist[stretched[i]]++;
    const total = stretched.length;
    let sum = 0;
    for (let t = 0; t < 256; t++) sum += t * hist[t];
    let sumB = 0, wB = 0, maxVar = 0, thresh = 128;
    for (let t = 0; t < 256; t++) {
      wB += hist[t];
      if (!wB) continue;
      const wF = total - wB;
      if (!wF) break;
      sumB += t * hist[t];
      const mB = sumB / wB;
      const mF = (sum - sumB) / wF;
      const varBetween = wB * wF * (mB - mF) ** 2;
      if (varBetween > maxVar) { maxVar = varBetween; thresh = t; }
    }

    for (let i = 0, j = 0; i < data.length; i += 4, j++) {
      const val = stretched[j] >= thresh ? 255 : 0;
      data[i] = data[i + 1] = data[i + 2] = val;
      data[i + 3] = 255;
    }

    ctx.putImageData(img, 0, 0);
    return canvas;
  }

  // ──────────────────────────────────────────────────────────────
  // HELPER METHODS
  // ──────────────────────────────────────────────────────────────

  private applyMorphologyOpen(data: Uint8Array, w: number, h: number, kernel: number[][]): Uint8Array {
    // Erosion
    const eroded = new Uint8Array(w * h);
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        let minVal = 255;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            if (kernel[dy + 1][dx + 1]) {
              minVal = Math.min(minVal, data[(y + dy) * w + (x + dx)]);
            }
          }
        }
        eroded[y * w + x] = minVal;
      }
    }

    // Dilation
    const dilated = new Uint8Array(w * h);
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        let maxVal = 0;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            if (kernel[dy + 1][dx + 1]) {
              maxVal = Math.max(maxVal, eroded[(y + dy) * w + (x + dx)]);
            }
          }
        }
        dilated[y * w + x] = maxVal;
      }
    }

    return dilated;
  }

  private applyMorphologyClose(data: Uint8Array, w: number, h: number, kernel: number[][]): Uint8Array {
    // Dilation
    const dilated = new Uint8Array(w * h);
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        let maxVal = 0;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            if (kernel[dy + 1][dx + 1]) {
              maxVal = Math.max(maxVal, data[(y + dy) * w + (x + dx)]);
            }
          }
        }
        dilated[y * w + x] = maxVal;
      }
    }

    // Erosion
    const eroded = new Uint8Array(w * h);
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        let minVal = 255;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            if (kernel[dy + 1][dx + 1]) {
              minVal = Math.min(minVal, dilated[(y + dy) * w + (x + dx)]);
            }
          }
        }
        eroded[y * w + x] = minVal;
      }
    }

    return eroded;
  }

  private scaleCanvas(src: HTMLCanvasElement, scale: number): HTMLCanvasElement {
    const ow = Math.max(1, Math.round(src.width * scale));
    const oh = Math.max(1, Math.round(src.height * scale));
    const c = document.createElement('canvas');
    c.width = ow;
    c.height = oh;
    const ctx = c.getContext('2d')!;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(src, 0, 0, src.width, src.height, 0, 0, ow, oh);
    return c;
  }

  private cloneCanvas(source: HTMLCanvasElement): HTMLCanvasElement {
    const c = document.createElement('canvas');
    c.width = source.width;
    c.height = source.height;
    c.getContext('2d')!.drawImage(source, 0, 0);
    return c;
  }
}
