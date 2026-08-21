import { ITracker, DetectionResult, TrackedObject, BoundingBox } from '../core/interfaces';

export class IoUTracker implements ITracker {
  private tracks: TrackedObject[] = [];
  private nextTrackId: number = 1;

  // Hyperparameters
  private maxTimeSinceUpdate: number = 10; // Frames to keep track alive if undetected
  private minHits: number = 1;  // Return vehicle on FIRST confirmed hit — critical for fast-moving vehicles
  private iouThreshold: number = 0.2; // Lowered: fast vehicles move far between frames, reducing IoU overlap

  update(detections: DetectionResult[]): TrackedObject[] {
    // 1. Predict next state using simple velocity model (linear motion)
    //    For each track, shift bbox by its last observed velocity (dx, dy).
    //    This dramatically improves association of fast-moving vehicles.
    this.tracks.forEach(t => {
      t.timeSinceUpdate++;
      if (t.history.length >= 2) {
        const last = t.history[t.history.length - 1];
        const prev = t.history[t.history.length - 2];
        const dx = last.x - prev.x;
        const dy = last.y - prev.y;
        // Apply velocity prediction to the predicted bbox (only used for matching, not stored)
        (t as any)._predictedBbox = { x: t.bbox.x + dx, y: t.bbox.y + dy, w: t.bbox.w, h: t.bbox.h };
      } else {
        (t as any)._predictedBbox = t.bbox;
      }
    });

    // 2. Associate detections to existing tracks
    const matchedDetections = new Set<number>();
    const matchedTracks = new Set<number>();

    // Greedy matching (a better approach uses Hungarian algorithm, but greedy is fast for Edge)
    for (let d = 0; d < detections.length; d++) {
      let bestTrackIdx = -1;
      let bestIou = this.iouThreshold;

      for (let t = 0; t < this.tracks.length; t++) {
        if (matchedTracks.has(t)) continue;

        // Use velocity-predicted bbox for matching if available
        const predictedBbox = (this.tracks[t] as any)._predictedBbox || this.tracks[t].bbox;
        const iou = this.calculateIoU(detections[d].bbox, predictedBbox);
        if (iou > bestIou) {
          bestIou = iou;
          bestTrackIdx = t;
        }
      }

      if (bestTrackIdx !== -1) {
        // Match found
        matchedDetections.add(d);
        matchedTracks.add(bestTrackIdx);
        
        const track = this.tracks[bestTrackIdx];
        track.bbox = detections[d].bbox;
        track.confidence = detections[d].confidence;
        track.class = detections[d].class;
        track.timeSinceUpdate = 0;
        track.hits++;
        track.hitStreak++;
        track.history.push(detections[d].bbox);
        if (track.history.length > 20) track.history.shift(); // Keep history bounded
      }
    }

    // 3. Create new tracks for unmatched detections
    for (let d = 0; d < detections.length; d++) {
      if (!matchedDetections.has(d)) {
        this.tracks.push({
          trackId: this.nextTrackId++,
          bbox: detections[d].bbox,
          confidence: detections[d].confidence,
          class: detections[d].class,
          timeSinceUpdate: 0,
          hits: 1,
          hitStreak: 1,
          history: [detections[d].bbox]
        });
      }
    }

    // 4. Remove dead tracks
    this.tracks = this.tracks.filter(t => {
      // If a track was lost for too long, delete it
      if (t.timeSinceUpdate > this.maxTimeSinceUpdate) return false;
      return true;
    });

    // 5. Reset hitStreak for lost tracks
    this.tracks.forEach(t => {
      if (t.timeSinceUpdate > 0) t.hitStreak = 0;
    });

    // 6. Return "Active" tracks (those with enough hits and recently updated)
    return this.tracks.filter(t => t.hits >= this.minHits && t.timeSinceUpdate === 0);
  }

  private calculateIoU(boxA: BoundingBox, boxB: BoundingBox): number {
    const xA = Math.max(boxA.x, boxB.x);
    const yA = Math.max(boxA.y, boxB.y);
    const xB = Math.min(boxA.x + boxA.w, boxB.x + boxB.w);
    const yB = Math.min(boxA.y + boxA.h, boxB.y + boxB.h);

    const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
    if (interArea === 0) return 0;

    const boxAArea = boxA.w * boxA.h;
    const boxBArea = boxB.w * boxB.h;

    const iou = interArea / (boxAArea + boxBArea - interArea);
    return iou;
  }
}
