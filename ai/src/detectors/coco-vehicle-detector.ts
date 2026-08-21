import { IVehicleDetector, DetectionResult } from '../core/interfaces';

const VEHICLE_CLASSES = new Set(['car', 'truck', 'bus', 'motorcycle', 'motorbike', 'van', 'bicycle']);

export class COCOVehicleDetector implements IVehicleDetector {
  private detector: any = null;

  async load(): Promise<void> {
    if (!this.detector) {
      const [cocoSsdMod] = await Promise.all([
        import('@tensorflow-models/coco-ssd'),
        import('@tensorflow/tfjs-backend-webgl'),
        import('@tensorflow/tfjs'),
      ]);
      this.detector = await cocoSsdMod.load({ base: 'lite_mobilenet_v2' });
    }
  }

  async detect(imageSource: HTMLVideoElement | HTMLCanvasElement | HTMLImageElement): Promise<DetectionResult[]> {
    if (!this.detector) throw new Error('Detector not loaded');

    const rawPreds: any[] = await this.detector.detect(imageSource);
    
    return rawPreds
      .filter((p: any) => VEHICLE_CLASSES.has(p.class) && p.score >= 0.20)
      .sort((a: any, b: any) => b.score - a.score)
      .map((p: any) => {
        const cls = p.class;
        const normalized = (cls === 'motorcycle' || cls === 'motorbike' || cls === 'bicycle') ? 'bike' : cls;
        return {
          bbox: {
            x: p.bbox[0],
            y: p.bbox[1],
            w: p.bbox[2],
            h: p.bbox[3]
          },
          confidence: p.score,
          class: normalized
        };
      });
  }
}
