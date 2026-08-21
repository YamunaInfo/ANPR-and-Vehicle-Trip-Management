import { IStreamInput } from '../core/interfaces';

export class UploadedVideoSource implements IStreamInput {
  private videoEl: HTMLVideoElement;

  constructor(videoElement: HTMLVideoElement) {
    this.videoEl = videoElement;
  }

  get type(): 'video' | 'rtsp' {
    return 'video';
  }

  async start(): Promise<void> {
    if (this.videoEl.paused) {
      await this.videoEl.play();
    }
  }

  stop(): void {
    this.videoEl.pause();
  }

  getNextFrame(): HTMLVideoElement | HTMLCanvasElement | null {
    if (this.videoEl.paused || this.videoEl.ended) return null;
    return this.videoEl;
  }
}

export class RTSPStreamSource implements IStreamInput {
  private url: string;

  constructor(rtspUrl: string) {
    this.url = rtspUrl;
  }

  get type(): 'video' | 'rtsp' {
    return 'rtsp';
  }

  async start(): Promise<void> {
    console.warn(`[RTSPStreamSource] Attempting to connect to ${this.url}`);
    console.warn('[RTSPStreamSource] Native browser RTSP is not supported. This requires a backend WebRTC/HLS proxy to function.');
    throw new Error('RTSP proxy not implemented in frontend. Please use backend service.');
  }

  stop(): void {
    // Stop WebSocket/WebRTC connection
  }

  getNextFrame(): HTMLVideoElement | HTMLCanvasElement | null {
    return null;
  }
}
