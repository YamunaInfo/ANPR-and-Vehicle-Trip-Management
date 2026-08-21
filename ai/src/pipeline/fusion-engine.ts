import { 
  IFusionEngine, 
  ProcessedPlateResult, 
  FinalVehicleEvent, 
  StructuredOcrOutput, 
  VariantOcrResult,
  TrackPlateObservation
} from '../core/interfaces';

interface TrackData {
  trackId: number;
  cameraId?: string;
  observations: ProcessedPlateResult[];
  frameObservations: TrackPlateObservation[];
  vehicleClass?: string;
  startTime: number;
  lastTime: number;
  processed: boolean;
}

const normalizeFusionPlateText = (value: string | null | undefined): string | null => {
  if (!value) return null;
  const normalized = value
    .toUpperCase()
    .replace(/\s+/g, '')
    .replace(/[^A-Z0-9]/g, '');

  return normalized.length >= 4 ? normalized : null;
};

const applyIndianPlateCorrections = (plate: string): string => {
  const mapping: Record<string, string> = {
    O: '0', Q: '0', I: '1', L: '1', Z: '2', S: '5', B: '8', G: '6',
    0: 'O', 1: 'I', 5: 'S', 6: 'G', 8: 'B'
  };

  return plate
    .split('')
    .map((char, idx) => {
      const next = mapping[char] ?? char;
      if (idx < 2 && ['0', '1', '8', '6'].includes(char)) return mapping[char] ?? char;
      if (idx >= 2 && idx <= 5 && ['O', 'Q', 'I', 'L', 'Z', 'S', 'B', 'G'].includes(char)) return mapping[char] ?? char;
      if (idx >= plate.length - 4 && ['O', 'Q', 'I', 'L', 'S', 'B', 'G'].includes(char)) return mapping[char] ?? char;
      return next;
    })
    .join('');
};

/**
 * STEP 9-11: FUSION ENGINE
 * - Multi-variant fusion (STEP 9)
 * - Multi-frame fusion with weighted voting (STEP 10)
 * - Duplicate filter per camera and trackId (STEP 11)
 */
export class FusionEngine implements IFusionEngine {
  private tracks: Map<number, TrackData> = new Map();
  
  // STEP 11: Duplicate filter - store processed events per camera
  private processedEvents: Map<string, { plate: string; trackId: number; timestamp: number }[]> = new Map();
  private readonly DUPLICATE_TIME_WINDOW_MS = 5000;
  private readonly PLATE_SIMILARITY_THRESHOLD = 0.85;

  // ─────────────────────────────────────────────────────────────
  // STEP 9: MULTI-VARIANT FUSION
  // ─────────────────────────────────────────────────────────────

  static fuseVariants(
    variantResults: VariantOcrResult[],
    detectionConfidence: number,
    cropWidth: number,
    cropHeight: number,
    aspectRatio: number,
    originalCropDataUrl?: string,
    rectifiedCropDataUrl?: string,
    superResCropDataUrl?: string
  ): StructuredOcrOutput {
    if (!variantResults || variantResults.length === 0) {
      return {
        plateText: null,
        detectionConfidence,
        ocrConfidence: 0.0,
        validationConfidence: 0.0,
        validationStatus: 'LOW_CONFIDENCE',
        variantAgreement: 0,
        cropWidth,
        cropHeight,
        aspectRatio,
        variantResults: [],
        originalCropDataUrl,
        rectifiedCropDataUrl,
        superResCropDataUrl
      };
    }

    // Filter candidates with minimum confidence threshold
    // Moving/blurry plates score 0.25–0.40; still plates score 0.60+.
    // Use 0.25 so motion-blurred reads enter the fusion pool instead of being silently dropped.
    const validCandidates = variantResults.filter(
      v => v.cleanedText && v.cleanedText.length >= 4 && v.ocrConfidence >= 0.25
    );

    if (validCandidates.length === 0) {
      const maxConf = variantResults.reduce((max, v) => Math.max(max, v.ocrConfidence), 0.0);
      const firstVariant = variantResults[0];
      return {
        plateText: firstVariant?.cleanedText || null,
        detectionConfidence,
        ocrConfidence: maxConf,
        validationConfidence: firstVariant?.validation.validationConfidence || 0.0,
        validationStatus: 'LOW_CONFIDENCE',
        variantAgreement: 0,
        cropWidth,
        cropHeight,
        aspectRatio,
        variantResults,
        originalCropDataUrl,
        rectifiedCropDataUrl,
        superResCropDataUrl
      };
    }

    // Frequency agreement map
    const frequencyMap = new Map<string, { 
      count: number; 
      totalScore: number; 
      totalValidConf: number;
      bestVariant: VariantOcrResult;
    }>();

    for (const v of validCandidates) {
      const txt = v.cleanedText;
      if (!frequencyMap.has(txt)) {
        frequencyMap.set(txt, { count: 0, totalScore: 0, totalValidConf: 0, bestVariant: v });
      }
      const item = frequencyMap.get(txt)!;
      item.count++;
      
      const score = (v.validation.validationConfidence * 0.40) + (v.ocrConfidence * 0.40);
      item.totalScore += score;
      item.totalValidConf += v.validation.validationConfidence;
      
      if (v.candidateScore > item.bestVariant.candidateScore) {
        item.bestVariant = v;
      }
    }

    let winnerText: string | null = null;
    let winnerScore = -1;
    let winnerVariant: VariantOcrResult = validCandidates[0];
    let winnerAgreement = 1;
    let winnerValidConf = 0;

    for (const [txt, data] of frequencyMap.entries()) {
      const avgScore = data.totalScore / data.count;
      const avgValidConf = data.totalValidConf / data.count;
      const combinedScore = avgScore + (data.count * 0.20);

      if (combinedScore > winnerScore) {
        winnerScore = combinedScore;
        winnerText = txt;
        winnerVariant = data.bestVariant;
        winnerAgreement = data.count;
        winnerValidConf = avgValidConf;
      }
    }

    return {
      plateText: winnerText,
      detectionConfidence,
      ocrConfidence: winnerVariant.ocrConfidence,
      validationConfidence: winnerValidConf,
      validationStatus: winnerVariant.validation.validationStatus,
      variantAgreement: winnerAgreement,
      cropWidth,
      cropHeight,
      aspectRatio,
      variantResults,
      originalCropDataUrl,
      rectifiedCropDataUrl,
      superResCropDataUrl
    };
  }

  // ─────────────────────────────────────────────────────────────
  // STEP 10: MULTI-FRAME FUSION (per trackId)
  // ─────────────────────────────────────────────────────────────

  addObservation(trackId: number, observation: ProcessedPlateResult, vehicleClass?: string, cameraId?: string): void {
    const normalizedText = normalizeFusionPlateText(observation.text);
    if (!normalizedText) {
      return;
    }

    const correctedPlate = applyIndianPlateCorrections(normalizedText);
    const normalizedObservation: ProcessedPlateResult = {
      ...observation,
      text: correctedPlate,
      ocrConfidence: Math.max(0, Math.min(1, observation.ocrConfidence || 0)),
      validationConfidence: observation.validationConfidence ?? (observation.validationStatus === 'VALID' ? 0.95 : observation.ocrConfidence * 0.8),
      validationStatus: observation.validationStatus ?? (observation.isValid ? 'VALID' : 'LOW_CONFIDENCE'),
      isValid: observation.isValid || observation.validationStatus === 'VALID'
    };

    if (!this.tracks.has(trackId)) {
      this.tracks.set(trackId, {
        trackId,
        cameraId,
        observations: [],
        frameObservations: [],
        vehicleClass,
        startTime: normalizedObservation.timestamp,
        lastTime: normalizedObservation.timestamp,
        processed: false
      });
    }

    const track = this.tracks.get(trackId)!;
    track.observations.push(normalizedObservation);
    
    // Store frame observation for tracking history
    track.frameObservations.push({
      frameNumber: normalizedObservation.frameNumber,
      timestamp: normalizedObservation.timestamp,
      plateText: normalizedObservation.text,
      ocrConfidence: normalizedObservation.ocrConfidence,
      validationConfidence: normalizedObservation.validationConfidence ?? normalizedObservation.ocrConfidence * 0.8,
      isValid: normalizedObservation.isValid
    });
    
    if (vehicleClass) track.vehicleClass = vehicleClass;
    if (cameraId) track.cameraId = cameraId;
    track.lastTime = normalizedObservation.timestamp;
  }

  /**
   * STEP 10: Multi-frame fusion using weighted voting
   */
  getBestResult(trackId: number, cameraId?: string): FinalVehicleEvent | null {
    const track = this.tracks.get(trackId);
    if (!track || track.observations.length === 0) return null;

    // Count plate text frequency and score using the normalized OCR values from every frame.
    const plateCounts = new Map<string, {
      count: number;
      totalOcrConf: number;
      totalValidConf: number;
      totalDetConf: number;
      lastObs: ProcessedPlateResult;
      firstFrame: number;
      lastFrame: number;
    }>();

    for (const obs of track.observations) {
      const normalizedText = normalizeFusionPlateText(obs.text);
      if (!normalizedText) {
        continue;
      }

      const text = applyIndianPlateCorrections(normalizedText);
      if (!plateCounts.has(text)) {
        plateCounts.set(text, {
          count: 0,
          totalOcrConf: 0,
          totalValidConf: 0,
          totalDetConf: 0,
          lastObs: obs,
          firstFrame: obs.frameNumber,
          lastFrame: obs.frameNumber
        });
      }

      const entry = plateCounts.get(text)!;
      entry.count++;
      entry.totalOcrConf += obs.ocrConfidence;
      entry.totalDetConf += obs.detectionConfidence;
      entry.totalValidConf += obs.validationConfidence ?? (obs.validationStatus === 'VALID' ? 0.95 : obs.ocrConfidence * 0.8);
      entry.lastFrame = Math.max(entry.lastFrame, obs.frameNumber);
      entry.lastObs = obs;
    }

    let bestPlate: string | null = null;
    let maxWeightedScore = -1;
    let bestAvgOcrConf = 0;
    let bestAvgValidConf = 0;
    let bestAvgDetConf = 0;
    let bestMultiFrameAgreement = 1;
    let bestObs: ProcessedPlateResult = track.observations[track.observations.length - 1];

    for (const [plate, data] of plateCounts.entries()) {
      const avgOcr = data.totalOcrConf / data.count;
      const avgValid = data.totalValidConf / data.count;
      const avgDet = data.totalDetConf / data.count;
      const frequencyBonus = data.count * 0.55;
      const confidenceBonus = (avgOcr * 0.35) + (avgValid * 0.25);
      const weightedScore = frequencyBonus + confidenceBonus + (avgDet * 0.15);

      if (weightedScore > maxWeightedScore) {
        maxWeightedScore = weightedScore;
        bestPlate = plate;
        bestAvgOcrConf = avgOcr;
        bestAvgValidConf = avgValid;
        bestAvgDetConf = avgDet;
        bestObs = data.lastObs;
        bestMultiFrameAgreement = data.count;
      }
    }

    // Requirement: if 3 or more frames agree on the same plate, prefer it over LOW_CONFIDENCE.
    const consensusPlate = Array.from(plateCounts.entries())
      .filter(([, data]) => data.count >= 3)
      .sort((a, b) => b[1].count - a[1].count || b[1].totalOcrConf - a[1].totalOcrConf)[0];

    if (consensusPlate) {
      bestPlate = consensusPlate[0];
      const data = consensusPlate[1];
      bestAvgOcrConf = data.totalOcrConf / data.count;
      bestAvgValidConf = data.totalValidConf / data.count;
      bestAvgDetConf = data.totalDetConf / data.count;
      bestMultiFrameAgreement = data.count;
      bestObs = data.lastObs;
    }

    const isDuplicate = this.isDuplicateEvent(trackId, bestPlate, cameraId || 'unknown');
    const validationStatus = bestPlate && bestMultiFrameAgreement >= 3
      ? 'VALID'
      : (bestAvgOcrConf >= 0.40 && bestAvgValidConf >= 0.40 ? 'VALID' : 'LOW_CONFIDENCE');

    if (!bestPlate || bestMultiFrameAgreement < 1) {
      return {
        trackId,
        cameraId: cameraId || track.cameraId || 'unknown',
        bestPlate: null,
        confidence: 0,
        ocrConfidence: 0,
        validationConfidence: 0,
        detectionConfidence: 0,
        validationStatus: 'LOW_CONFIDENCE',
        vehicleClass: track.vehicleClass || 'vehicle',
        timestampStart: track.startTime,
        timestampEnd: track.lastTime,
        frameCount: track.observations.length,
        frameAgreements: 0,
        multiFrameAgreement: 0,
        variantAgreement: bestObs.variantResults?.length || 0,
        allPlates: [],
        frameObservations: track.frameObservations,
        isDuplicate,
        debugInfo: {
          plateText: null,
          detectionConfidence: 0,
          ocrConfidence: 0,
          validationConfidence: 0,
          validationStatus: 'LOW_CONFIDENCE',
          trackId,
          cameraId: cameraId || track.cameraId || 'unknown',
          frameCount: track.observations.length,
          variantAgreement: bestObs.variantResults?.length || 0,
          multiFrameAgreement: 0,
          variantResults: bestObs.variantResults,
          originalCropDataUrl: bestObs.originalCropDataUrl,
          rectifiedCropDataUrl: bestObs.rectifiedCropDataUrl,
          superResCropDataUrl: bestObs.superResCropDataUrl,
          cropWidth: bestObs.cropWidth,
          cropHeight: bestObs.cropHeight,
          aspectRatio: bestObs.aspectRatio
        }
      };
    }

    return {
      trackId,
      cameraId: cameraId || track.cameraId || 'unknown',
      bestPlate,
      confidence: parseFloat(((bestAvgOcrConf + bestAvgValidConf + bestAvgDetConf) / 3).toFixed(3)),
      ocrConfidence: parseFloat(bestAvgOcrConf.toFixed(3)),
      validationConfidence: parseFloat(bestAvgValidConf.toFixed(3)),
      detectionConfidence: parseFloat(bestAvgDetConf.toFixed(3)),
      validationStatus,
      vehicleClass: track.vehicleClass || 'vehicle',
      timestampStart: track.startTime,
      timestampEnd: track.lastTime,
      frameCount: track.observations.length,
      frameAgreements: bestPlate ? (plateCounts.get(bestPlate)?.count || 1) : 1,
      multiFrameAgreement: bestMultiFrameAgreement,
      variantAgreement: bestObs.variantResults?.length || 0,
      allPlates: Array.from(new Set(
        track.observations
          .map(o => normalizeFusionPlateText(o.text) || '')
          .filter((t): t is string => Boolean(t))
      )),
      frameObservations: track.frameObservations,
      isDuplicate,
      debugInfo: {
        plateText: bestPlate,
        detectionConfidence: bestAvgDetConf,
        ocrConfidence: bestAvgOcrConf,
        validationConfidence: bestAvgValidConf,
        validationStatus,
        trackId,
        cameraId: cameraId || track.cameraId || 'unknown',
        frameCount: track.observations.length,
        variantAgreement: bestObs.variantResults?.length || 0,
        multiFrameAgreement: bestMultiFrameAgreement,
        variantResults: bestObs.variantResults,
        originalCropDataUrl: bestObs.originalCropDataUrl,
        rectifiedCropDataUrl: bestObs.rectifiedCropDataUrl,
        superResCropDataUrl: bestObs.superResCropDataUrl,
        cropWidth: bestObs.cropWidth,
        cropHeight: bestObs.cropHeight,
        aspectRatio: bestObs.aspectRatio
      }
    };
  }

  // ─────────────────────────────────────────────────────────────
  // STEP 11: DUPLICATE FILTER
  // ─────────────────────────────────────────────────────────────

  private isDuplicateEvent(trackId: number, plateText: string | null, cameraId: string): boolean {
    if (!plateText) return false;

    const key = `${cameraId}`;
    if (!this.processedEvents.has(key)) {
      this.processedEvents.set(key, []);
    }

    const events = this.processedEvents.get(key)!;
    const now = Date.now();

    // Clean up old events outside time window
    for (let i = events.length - 1; i >= 0; i--) {
      if (now - events[i].timestamp > this.DUPLICATE_TIME_WINDOW_MS) {
        events.splice(i, 1);
      }
    }

    // Check similarity with recent events
    for (const evt of events) {
      if (evt.trackId === trackId) continue;
      
      const similarity = this.calculateStringSimilarity(plateText, evt.plate);
      if (similarity >= this.PLATE_SIMILARITY_THRESHOLD) {
        return true;
      }
    }

    // Record this event
    events.push({ plate: plateText, trackId, timestamp: now });
    return false;
  }

  private calculateStringSimilarity(str1: string, str2: string): number {
    if (!str1 || !str2) return 0;
    if (str1 === str2) return 1.0;

    const maxLen = Math.max(str1.length, str2.length);
    let matches = 0;

    for (let i = 0; i < Math.min(str1.length, str2.length); i++) {
      if (str1[i] === str2[i]) matches++;
    }

    return matches / maxLen;
  }

  clearTrack(trackId: number): void {
    this.tracks.delete(trackId);
  }

  clearOldTracks(maxAgeMs: number): void {
    const now = Date.now();
    for (const [trackId, data] of this.tracks.entries()) {
      if (now - data.lastTime > maxAgeMs) {
        this.tracks.delete(trackId);
      }
    }
  }

  clearOldDuplicates(maxAgeMs: number): void {
    const now = Date.now();
    for (const [, events] of this.processedEvents.entries()) {
      for (let i = events.length - 1; i >= 0; i--) {
        if (now - events[i].timestamp > maxAgeMs) {
          events.splice(i, 1);
        }
      }
    }
  }
}
