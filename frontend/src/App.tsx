import React, { type ReactNode, useEffect, useMemo, useRef, useState } from 'react';
import { 
  COCOVehicleDetector,
  YoloVehicleDetector,
  YoloPlateDetector, 
  PaddleOcrEngine, 
  IndianPlateValidator, 
} from '@workspace/ai';

import { QueryClient, QueryClientProvider, useQueryClient } from '@tanstack/react-query';
import { Link, Route, Switch, useLocation, Router as WouterRouter } from 'wouter';
import { 
  Activity, AlertCircle, AlertTriangle, ArrowDownToLine, ArrowLeftRight, ArrowUpRight, 
  BarChart3, Bell, Bike, Bus, Camera, Car, Check, CheckCircle2, ChevronDown, CircleDot, 
  Clock3, Cpu, Database, Eye, FileBarChart, FileSearch, Film, Gauge, Layers, LogOut, Maximize2, 
  Menu, Pause, Pencil, Play, Plus, Radio, RefreshCw, RotateCcw, Search, Settings, ShieldCheck, 
  Siren, SlidersHorizontal, Sparkles, Trash2, Truck, Upload, UserRound, Users, Video, Volume2, VolumeX, XCircle 
} from 'lucide-react';

import {
  getGetActiveTripsQueryKey, getGetAlertsQueryKey, getGetCamerasQueryKey, getGetDashboardSummaryQueryKey, 
  getGetEventsQueryKey, getGetReviewQueueQueryKey, getGetTripsQueryKey, getGetVehiclesQueryKey, 
  getGetDriversQueryKey, getGetMeQueryKey,
  useCreateCamera, useCreateDetection, useCreateDriver, useCreateTrip, useCreateVehicle, 
  useCorrectPlate, useDeleteVehicle, useGetActiveTrips, useGetAlerts, useGetCameras, 
  useGetDashboardActivity, useGetDashboardSummary, useGetDrivers, useGetEvents, useGetMe, 
  useGetReportsOverview, useGetReviewQueue, useGetTrips, useGetVehicles, useHealthCheck, 
  useLogin, useMarkAlertRead, useSimulateTraffic, useUpdateTripStatus, useUpdateVehicle,
  type ActivityItem, type Alert, type Camera as CameraRecord, type Driver, type GateEvent, type Trip, type Vehicle,
} from '@workspace/api-client-react';
import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  AreaChart,
  Area,
} from 'recharts';
import { ErrorBoundary } from '@/components/error-boundary';
import { Busy, Button, Card, DataTable, DetailLink, EmptyState, ErrorState, Field, IconButton, LoadingRows, Logo, Metric, Modal, Notice, PageHeader, SearchBox, SelectField, StatusPill, TinyBar } from '@/components/gatesense-ui';
import { AuthProvider, useAuth } from '@/lib/auth';
import LoginPage from '@/pages/login';
import ProfilePage from '@/pages/profile';
import NotFound from '@/pages/not-found';
import '@/index.css';

const queryClient = new QueryClient();

const nav = [
  { href: '/', label: 'Live Gate View', icon: SlidersHorizontal },
  { href: '/vehicles', label: 'Vehicle Master', icon: Truck },
  { href: '/drivers', label: 'Driver Master', icon: Users },
  { href: '/schedule', label: 'Trip Scheduling', icon: Clock3 },
  { href: '/events', label: 'Entry/Exit Register', icon: Database },
  { href: '/whitelist', label: 'Whitelist', icon: ShieldCheck },
  { href: '/watchlist', label: 'Watchlist', icon: AlertCircle },
  { href: '/review', label: 'Manual Review', icon: FileSearch },
  { href: '/trips', label: 'Trip Monitoring', icon: ArrowLeftRight },
  { href: '/alerts', label: 'Alerts', icon: Bell },
  { href: '/reports', label: 'Reports', icon: BarChart3 },
  { href: '/cameras', label: 'Camera Management', icon: Camera },
];

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <WouterRouter base={import.meta.env.BASE_URL.replace(/\/$/, '')}>
          <Router />
        </WouterRouter>
      </AuthProvider>
    </QueryClientProvider>
  );
}

function Router() {
  const [location] = useLocation();
  const { isAuthenticated } = useAuth();

  if (!isAuthenticated || location === '/login' || location === '/signup') {
    return (
      <ErrorBoundary resetKey={location}>
        <LoginPage initialMode={location === '/login' ? 'signin' : 'signup'} />
      </ErrorBoundary>
    );
  }

  return (
    <ErrorBoundary resetKey={location}>
      <Shell>
        <Switch>
          <Route path="/" component={LiveGateViewPage} />
          <Route path="/live" component={LiveGateViewPage} />
          <Route path="/detect" component={LiveGateViewPage} />
          <Route path="/trips" component={TripsPage} />
          <Route path="/schedule" component={SchedulePage} />
          <Route path="/events" component={EventsPage} />
          <Route path="/vehicles" component={VehiclesPage} />
          <Route path="/whitelist" component={VehiclesPage} />
          <Route path="/watchlist" component={AlertsPage} />
          <Route path="/drivers" component={DriversPage} />
          <Route path="/alerts" component={AlertsPage} />
          <Route path="/review" component={ReviewPage} />
          <Route path="/reports" component={ReportsPage} />
          <Route path="/cameras" component={CamerasPage} />
          <Route path="/profile" component={ProfilePage} />
          <Route path="/settings" component={ProfilePage} />
          <Route component={NotFound} />
        </Switch>
      </Shell>
    </ErrorBoundary>
  );
}

function Shell({ children }: { children: ReactNode }) {
  const [location, setLocation] = useLocation();
  const [mobileOpen, setMobileOpen] = useState(false);
  const health = useHealthCheck({ query: { refetchInterval: 30000, queryKey: ['/api/healthz'] } });
  const { user, logout } = useAuth();

  const operatorName = user?.name || 'Ravi Kumar';
  const operatorRole = user?.role || 'guard';
  const initials = operatorName
    .split(' ')
    .map((n) => n[0])
    .join('')
    .toUpperCase()
    .slice(0, 2) || 'RK';

  const handleSignOut = () => {
    logout();
    setLocation('/login');
  };

  return (
    <div className="min-h-[100dvh] bg-[#0B0F15] text-white">
      {/* Sidebar */}
      <aside className={`fixed inset-y-0 left-0 z-40 flex w-[250px] flex-col border-r border-[#1A222F] bg-[#0B0F15] transition-transform lg:translate-x-0 ${mobileOpen ? 'translate-x-0' : '-translate-x-full'}`}>
        {/* Top Logo */}
        <div className="flex h-[72px] items-center border-b border-[#1A222F] px-5">
          <Logo />
        </div>

        {/* Navigation List */}
        <div className="px-3 pt-5 flex-1 overflow-y-auto">
          <nav className="space-y-1">
            {nav.map(({ href, label, icon: Icon }) => {
              const isActive = location === href || (href === '/' && (location === '/live' || location === '/detect'));
              return (
                <Link
                  key={href}
                  href={href}
                  onClick={() => setMobileOpen(false)}
                  data-testid={`link-nav-${label.toLowerCase().replace(/[\s/]+/g, '-')}`}
                  className={`group flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition ${
                    isActive
                      ? 'bg-[#1A1811] text-[#F59E0B] font-semibold border border-[#F59E0B]/20'
                      : 'text-gray-400 hover:bg-[#141A24] hover:text-gray-200'
                  }`}
                >
                  <Icon className={`h-4 w-4 ${isActive ? 'text-[#F59E0B]' : 'text-gray-400 group-hover:text-gray-200'}`} />
                  <span className="flex-1 truncate">{label}</span>
                  {label === 'Alerts' && <AlertBadge />}
                </Link>
              );
            })}
          </nav>
        </div>

        {/* Footer Profile */}
        <div className="mt-auto border-t border-[#1A222F] p-3 space-y-2">
          <Link
            href="/profile"
            data-testid="link-settings"
            className="flex items-center gap-3 rounded-lg px-3 py-2 text-sm text-gray-400 hover:bg-[#141A24] hover:text-gray-200"
          >
            <UserRound className="h-4 w-4" />
            Profile
          </Link>
          <div className="flex items-center gap-3 rounded-xl border border-[#1E2638] bg-[#121822] p-2.5">
            <span className="grid h-8 w-8 place-items-center rounded-full bg-[#1C2432] text-xs font-bold text-[#F59E0B] border border-amber-500/20">
              {initials}
            </span>
            <div className="min-w-0 flex-1">
              <p className="truncate text-xs font-semibold text-white">{operatorName}</p>
            </div>
            <button
              data-testid="button-sidebar-logout"
              onClick={handleSignOut}
              title="Sign out / Switch account"
              className="p-1 rounded text-gray-400 hover:bg-[#1C2432] hover:text-red-400 transition"
            >
              <LogOut className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>
      </aside>

      {mobileOpen && (
        <button
          aria-label="Close navigation"
          data-testid="button-close-mobile-nav"
          onClick={() => setMobileOpen(false)}
          className="fixed inset-0 z-30 bg-black/70 lg:hidden"
        />
      )}

      {/* Main Content Area */}
      <main className="min-h-[100dvh] lg:pl-[250px]">
        {/* Top Header */}
        <header className="sticky top-0 z-20 flex h-[72px] items-center justify-between border-b border-[#1A222F] bg-[#0B0F15]/95 px-4 backdrop-blur-md md:px-8">
          <div className="flex items-center gap-3">
            <button
              data-testid="button-open-mobile-nav"
              aria-label="Open navigation"
              onClick={() => setMobileOpen(true)}
              className="rounded-md p-2 text-gray-400 hover:bg-[#161D27] lg:hidden"
            >
              <Menu className="h-5 w-5" />
            </button>
            <div className="flex items-center gap-2 text-xs text-gray-400">
              <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
              Plant 04 / North gate cluster
            </div>
          </div>

          <div className="flex items-center gap-4">
            <Link
              href="/alerts"
              data-testid="link-header-alerts"
              className="relative rounded-md p-2 text-gray-400 hover:bg-[#161D27] hover:text-white"
            >
              <Bell className="h-[18px] w-[18px]" />
              <AlertBadge isHeader />
            </Link>
            <div className="hidden h-5 w-px bg-[#1A222F] sm:block" />
            <span className="font-mono text-xs text-gray-300">
              {new Date().toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', hour12: false })} IST
            </span>
          </div>
        </header>

        {/* Page Content Container */}
        <div className="mx-auto max-w-[1550px] p-4 md:p-8">{children}</div>
      </main>
    </div>
  );
}

function AlertBadge({ isHeader }: { isHeader?: boolean }) {
  const alerts = useGetAlerts({ query: { refetchInterval: 3000, queryKey: getGetAlertsQueryKey() } });
  const unread = (alerts.data ?? []).filter(a => !a.isRead).length;
  if (unread <= 0) return null;
  return (
    <span
      data-testid="badge-unread-alerts"
      className={`${
        isHeader ? 'absolute right-1 top-1' : 'ml-auto'
      } grid h-4 min-w-4 place-items-center rounded-full bg-red-500 px-1 text-[9px] font-bold text-white shadow-md animate-pulse`}
    >
      {unread}
    </span>
  );
}

interface LiveDetectedVehicle {
  id: string;
  class: string;
  plate: string;
  displayPlate?: string;
  status?: string; // 'finalized' | 'pending' | 'manual_review'
  confidence: number;
  ocrConfidence?: number;
  frameCount?: number;
  evidenceCount?: number;
  agreementRatio?: number;
  timestamp: string;
  thumbnail: string;
  lastSeen: number;
  detectionId: number;
  vehicleBbox?: number[];
  plateBbox?: number[];
  framePredictions?: Array<{ frame_number: number; plate_number: string; confidence: number; raw_text?: string }>;
  supportingPredictions?: Array<{ plate: string; confidence: number; frame?: number }>;
}

interface AiHealthStatus {
  vehicle_model?: string;
  plate_model?: string;
  ocr?: string;
  opencv?: string;
}

function LiveGateViewPage() {
  const qc = useQueryClient();
  const [videoSource, setVideoSource] = useState<string | null>(null);
  const [rawVideoUrl, setRawVideoUrl] = useState<string | null>(null);
  const [annotatedVideoUrl, setAnnotatedVideoUrl] = useState<string | null>(null);
  const [videoMode, setVideoMode] = useState<'annotated' | 'original'>('annotated');
  const [isWebcam, setIsWebcam] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string>('');
  const [fps, setFps] = useState(0.0);
  const [frameNumber, setFrameNumber] = useState(0);
  const [vehiclesCount, setVehiclesCount] = useState(0);
  const [platesCount, setPlatesCount] = useState(0);
  const [ocrSuccessCount, setOcrSuccessCount] = useState(0);
  const [hasProcessedVideo, setHasProcessedVideo] = useState(false);
  const [aiHealth, setAiHealth] = useState<AiHealthStatus | null>(null);
  const [aiHealthError, setAiHealthError] = useState<string | null>(null);
  const [selectedTrackId, setSelectedTrackId] = useState<number | null>(null);

  // Camera Source & CCTV States
  const [cameraSource, setCameraSource] = useState<'cctv' | 'laptop'>('cctv');
  const [rtspUrl, setRtspUrl] = useState<string>('');
  const [cctvStatus, setCctvStatus] = useState<'disconnected' | 'connecting' | 'connected' | 'reconnecting' | 'error'>('disconnected');
  const [cctvStatusMessage, setCctvStatusMessage] = useState<string>('CCTV Disconnected');
  const [isCctvStreaming, setIsCctvStreaming] = useState<boolean>(false);
  const [isCctvConnecting, setIsCctvConnecting] = useState<boolean>(false);
  const cctvPollIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const prevPlatesCountRef = useRef<number>(0);

  // Video playback states
  const [isPlaying, setIsPlaying] = useState(true);
  const [isLooping, setIsLooping] = useState(true);
  const [isMuted, setIsMuted] = useState(true);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // REAL detections only - initialized to empty array (NO mock data)
  const [detectedVehicles, setDetectedVehicles] = useState<LiveDetectedVehicle[]>([]);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const videoContainerRef = useRef<HTMLDivElement | null>(null);
  const rawVideoUrlRef = useRef<string | null>(null);

  const animFrameIdRef = useRef<number | null>(null);
  const frameCountRef = useRef<number>(0);
  const lastFpsTimeRef = useRef<number>(performance.now());
  const framesSinceLastFpsRef = useRef<number>(0);

  // CCTV Live Polling
  const startCctvPolling = () => {
    stopCctvPolling();
    cctvPollIntervalRef.current = setInterval(async () => {
      try {
        const res = await fetch('http://localhost:5001/api/cctv/status');
        if (!res.ok) return;
        const data = await res.json();
        setCctvStatus(data.status);
        setCctvStatusMessage(data.message || (data.status === 'connected' ? 'CCTV Connected' : 'CCTV Disconnected'));
        setFps(data.fps || 0.0);
        setFrameNumber(data.frames_processed || 0);
        setVehiclesCount(data.vehicles_count || 0);
        setPlatesCount(data.plates_count || 0);
        setOcrSuccessCount(data.plates_count || 0);

        if (Array.isArray(data.detections) && data.detections.length > 0) {
          setDetectedVehicles(data.detections);
          if (!selectedTrackId) {
            setSelectedTrackId(data.detections[0].detectionId);
          }
        }

        const newPlates = Number(data.plates_count || 0);
        if (newPlates > prevPlatesCountRef.current) {
          prevPlatesCountRef.current = newPlates;
          qc.invalidateQueries({ queryKey: getGetEventsQueryKey() });
          qc.invalidateQueries({ queryKey: getGetActiveTripsQueryKey() });
          qc.invalidateQueries({ queryKey: getGetDashboardSummaryQueryKey() });
          qc.invalidateQueries({ queryKey: getGetAlertsQueryKey() });
        }
      } catch (e) {
        console.warn('[CCTV] Polling error:', e);
      }
    }, 600);
  };

  const stopCctvPolling = () => {
    if (cctvPollIntervalRef.current) {
      clearInterval(cctvPollIntervalRef.current);
      cctvPollIntervalRef.current = null;
    }
  };

  const handleConnectCctv = async (overrideUrl?: string) => {
    const urlToUse = (overrideUrl || rtspUrl).trim();
    if (!urlToUse) {
      setCctvStatus('error');
      setCctvStatusMessage('Invalid RTSP URL');
      return;
    }

    // Stop webcam if active
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
    setIsWebcam(false);
    setRawVideoUrl(null);
    setAnnotatedVideoUrl(null);
    setVideoSource(null);
    setDetectedVehicles([]);
    setVehiclesCount(0);
    setPlatesCount(0);
    setOcrSuccessCount(0);

    setIsCctvConnecting(true);
    setCctvStatus('connecting');
    setCctvStatusMessage('Connecting...');

    try {
      const res = await fetch('http://localhost:5001/api/cctv/connect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rtsp_url: urlToUse }),
      });
      const data = await res.json();
      if (data.success && data.status !== 'error') {
        setCctvStatus(data.status);
        setCctvStatusMessage(data.message || 'CCTV Connected');
        setIsCctvStreaming(true);
        setHasProcessedVideo(true);
        startCctvPolling();
      } else {
        setCctvStatus('error');
        setCctvStatusMessage(data.message || 'Unable to connect to camera');
        setIsCctvStreaming(false);
      }
    } catch (err: any) {
      setCctvStatus('error');
      setCctvStatusMessage(`Unable to connect to camera (${err.message})`);
      setIsCctvStreaming(false);
    } finally {
      setIsCctvConnecting(false);
    }
  };

  const handleStopCctv = async () => {
    stopCctvPolling();
    setIsCctvStreaming(false);
    setIsCctvConnecting(false);
    setCctvStatus('disconnected');
    setCctvStatusMessage('CCTV Disconnected');
    try {
      await fetch('http://localhost:5001/api/cctv/stop', { method: 'POST' });
    } catch (e) {}
  };

  // Check backend AI Health on mount
  useEffect(() => {
    fetch('http://localhost:5001/api/ai/health')
      .then(async res => {
        if (!res.ok) {
          throw new Error(`AI Service returned HTTP ${res.status}`);
        }
        const data = await res.json();
        setAiHealth(data);
        if (data.vehicle_model !== 'loaded' || data.plate_model !== 'loaded' || data.ocr !== 'loaded') {
          setAiHealthError(`Models not fully loaded: Vehicle=${data.vehicle_model}, Plate=${data.plate_model}, OCR=${data.ocr}`);
        } else {
          setAiHealthError(null);
        }
      })
      .catch(err => {
        console.warn('[LiveGateView] Python AI Health Check Error:', err);
        setAiHealthError(`Cannot connect to Python AI backend at http://localhost:5001 (${err.message})`);
      });

    return () => {
      if (animFrameIdRef.current) cancelAnimationFrame(animFrameIdRef.current);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop());
      }
    };
  }, []);

  // Frame processing loop for video playback & canvas overlay
  const processFrame = () => {
    if (!videoRef.current || !canvasRef.current) {
      animFrameIdRef.current = requestAnimationFrame(processFrame);
      return;
    }
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (video.paused || video.ended || video.videoWidth === 0) {
      animFrameIdRef.current = requestAnimationFrame(processFrame);
      return;
    }

    const now = performance.now();
    frameCountRef.current += 1;
    setFrameNumber(frameCountRef.current);

    if (now - lastFpsTimeRef.current >= 500) {
      const measured = (framesSinceLastFpsRef.current / (now - lastFpsTimeRef.current)) * 1000;
      if (measured > 0) setFps(parseFloat(measured.toFixed(1)));
      lastFpsTimeRef.current = now;
      framesSinceLastFpsRef.current = 0;
    }
    framesSinceLastFpsRef.current += 1;

    if (canvas.width !== video.clientWidth || canvas.height !== video.clientHeight) {
      canvas.width = video.clientWidth;
      canvas.height = video.clientHeight;
    }

    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // In Original Video mode, draw dynamic bounding boxes from detected vehicles
      if (videoMode === 'original' && detectedVehicles.length > 0 && video.videoWidth > 0 && video.videoHeight > 0) {
        const scaleX = canvas.width / video.videoWidth;
        const scaleY = canvas.height / video.videoHeight;

        detectedVehicles.forEach(vehicle => {
          const isSelected = selectedTrackId === vehicle.detectionId;

          // 1. Vehicle Bounding Box
          if (vehicle.vehicleBbox) {
            const [vx1, vy1, vx2, vy2] = vehicle.vehicleBbox;
            const x = vx1 * scaleX;
            const y = vy1 * scaleY;
            const w = (vx2 - vx1) * scaleX;
            const h = (vy2 - vy1) * scaleY;

            ctx.strokeStyle = isSelected ? '#F59E0B' : '#22C55E';
            ctx.lineWidth = isSelected ? 3 : 2;
            ctx.strokeRect(x, y, w, h);

            // Label tag on top of vehicle
            const label = `${vehicle.class} ${(vehicle.confidence * 100).toFixed(0)}%`;
            ctx.font = 'bold 11px monospace';
            const textWidth = ctx.measureText(label).width;
            ctx.fillStyle = isSelected ? '#F59E0B' : '#22C55E';
            ctx.fillRect(x, Math.max(0, y - 18), textWidth + 8, 18);
            ctx.fillStyle = '#000000';
            ctx.fillText(label, x + 4, Math.max(13, y - 5));
          }

          // 2. License Plate Box
          if (vehicle.plateBbox) {
            const [px1, py1, px2, py2] = vehicle.plateBbox;
            const px = px1 * scaleX;
            const py = py1 * scaleY;
            const pw = (px2 - px1) * scaleX;
            const ph = (py2 - py1) * scaleY;

            ctx.strokeStyle = '#3B82F6';
            ctx.lineWidth = 2;
            ctx.strokeRect(px, py, pw, ph);

            if (vehicle.plate || vehicle.displayPlate) {
              const pText = vehicle.plate || vehicle.displayPlate || '';
              ctx.font = 'bold 10px monospace';
              const pTextWidth = ctx.measureText(pText).width;
              ctx.fillStyle = '#3B82F6';
              ctx.fillRect(px, py + ph, pTextWidth + 6, 16);
              ctx.fillStyle = '#FFFFFF';
              ctx.fillText(pText, px + 3, py + ph + 12);
            }
          }
        });
      }
    }

    animFrameIdRef.current = requestAnimationFrame(processFrame);
  };

  const startLoop = () => {
    if (animFrameIdRef.current) cancelAnimationFrame(animFrameIdRef.current);
    animFrameIdRef.current = requestAnimationFrame(processFrame);
  };

  // Video error recovery: fallback to raw video so screen never stays blank
  const handleVideoError = () => {
    console.warn('[LiveGateView] Video load error for source:', videoSource);
    if (rawVideoUrlRef.current && videoSource !== rawVideoUrlRef.current) {
      console.log('[LiveGateView] Auto-falling back to original uploaded video:', rawVideoUrlRef.current);
      setVideoMode('original');
      setVideoSource(rawVideoUrlRef.current);
    }
  };

  // Real Upload Handler: Sends video directly to Python Backend POST /api/video/process
  const handleVideoUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
    setIsWebcam(false);
    
    // Store local object URL persistently so video never disappears
    const localPreviewUrl = URL.createObjectURL(file);
    rawVideoUrlRef.current = localPreviewUrl;
    setRawVideoUrl(localPreviewUrl);
    setAnnotatedVideoUrl(null);
    setVideoMode('original');
    setVideoSource(localPreviewUrl);
    setHasProcessedVideo(false);
    setDetectedVehicles([]);
    setVehiclesCount(0);
    setPlatesCount(0);
    setOcrSuccessCount(0);
    setFps(0.0);
    setFrameNumber(0);
    frameCountRef.current = 0;
    setCurrentTime(0);
    setIsPlaying(true);

    setIsUploading(true);
    setUploadStatus(`Uploading ${file.name} to Python AI backend (YOLOv8 + EasyOCR)...`);

    const formData = new FormData();
    formData.append('video', file);
    formData.append('conf_threshold', '0.30');
    formData.append('plate_conf_threshold', '0.25');
    formData.append('process_fps', '4');
    formData.append('vehicle_img_size', '384');
    formData.append('max_process_fps', '4');

    try {
      setUploadStatus(`Processing frames with YOLOv8 vehicle detector, license plate model, and EasyOCR...`);
      const response = await fetch('http://localhost:5001/api/video/process', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`AI Backend Error (${response.status}): ${errorText}`);
      }

      const result = await response.json();
      console.log('[LiveGateView] REAL AI Process Result:', result);

      setHasProcessedVideo(true);
      const vCount = Number(result.vehicles_detected ?? 0);
      const pCount = Number(result.plates_detected ?? 0);
      setVehiclesCount(vCount);
      setPlatesCount(pCount);
      setOcrSuccessCount(pCount);
      setFps(result.fps || 0.0);
      setFrameNumber(result.frames_processed || 0);

      // Convert REAL backend detections to UI items (no mock fallback)
      const realDetections: LiveDetectedVehicle[] = (result.detections || []).map((d: any, idx: number) => {
        const timeStr = new Date().toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
        return {
          id: `track-${d.track_id}-${idx}`,
          detectionId: d.track_id,
          class: (d.vehicle_class || 'Vehicle').charAt(0).toUpperCase() + (d.vehicle_class || 'Vehicle').slice(1),
          plate: d.plate_number || '',
          displayPlate: d.display_plate || (d.plate_number ? d.plate_number : (d.status === 'manual_review' ? 'Requires Manual Review' : 'Recognizing...')),
          status: d.status || (d.plate_number ? 'finalized' : 'pending'),
          confidence: Number(d.final_confidence ?? d.ocr_confidence ?? d.vehicle_confidence ?? 0.0),
          ocrConfidence: Number(d.ocr_confidence ?? 0.0),
          frameCount: Number(d.frame_count ?? 1),
          evidenceCount: Number(d.evidence_count ?? 1),
          agreementRatio: Number(d.agreement_ratio ?? 1.0),
          timestamp: timeStr,
          thumbnail: d.thumbnail || '',
          lastSeen: Date.now(),
          vehicleBbox: d.vehicle_bbox,
          plateBbox: d.plate_bbox,
          framePredictions: (d.frame_predictions && d.frame_predictions.length > 0)
            ? d.frame_predictions
            : (d.supporting_predictions || []).map((sp: any, sIdx: number) => ({
                frame_number: sp.frame || sIdx + 1,
                plate_number: sp.plate,
                confidence: sp.confidence,
              })),
          supportingPredictions: d.supporting_predictions || [],
        };
      });

      setDetectedVehicles(realDetections);
      if (realDetections.length > 0) {
        setSelectedTrackId(realDetections[0].detectionId);
      }

      // Activate annotated video if produced by AI pipeline
      if (result.video_url) {
        const processedVideoUrl = result.video_url.startsWith('http')
          ? result.video_url
          : `http://localhost:5001${result.video_url}`;

        setAnnotatedVideoUrl(processedVideoUrl);
        setVideoMode('annotated');
        setVideoSource(processedVideoUrl);
      }
      // Automatically sync and refresh Entry/Exit Register, Active Trips, and Dashboard Metrics
      qc.invalidateQueries({ queryKey: getGetEventsQueryKey() });
      qc.invalidateQueries({ queryKey: getGetActiveTripsQueryKey() });
      qc.invalidateQueries({ queryKey: getGetDashboardSummaryQueryKey() });
      qc.invalidateQueries({ queryKey: getGetReviewQueueQueryKey() });
      qc.invalidateQueries({ queryKey: getGetAlertsQueryKey() });
    } catch (err: any) {
      console.error('[LiveGateView] Video process failed:', err);
      // Original video stays visible even if AI service encountered an error!
      alert(`AI processing failed: ${err.message}\n\nPlease verify that the Python service is running on port 5001. Your uploaded video will continue playing.`);
    } finally {
      setIsUploading(false);
      setUploadStatus('');
      if (e.target) e.target.value = '';
    }
  };

  const handleSwitchMode = (mode: 'annotated' | 'original') => {
    setVideoMode(mode);
    if (mode === 'annotated' && annotatedVideoUrl) {
      setVideoSource(annotatedVideoUrl);
    } else if (rawVideoUrl) {
      setVideoSource(rawVideoUrl);
    }
  };

  const togglePlayPause = () => {
    if (!videoRef.current) return;
    if (videoRef.current.paused || videoRef.current.ended) {
      videoRef.current.play().then(() => setIsPlaying(true)).catch(() => {});
    } else {
      videoRef.current.pause();
      setIsPlaying(false);
    }
  };

  const handleReplay = () => {
    if (!videoRef.current) return;
    videoRef.current.currentTime = 0;
    setCurrentTime(0);
    videoRef.current.play().then(() => setIsPlaying(true)).catch(() => {});
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const targetTime = parseFloat(e.target.value);
    if (videoRef.current) {
      videoRef.current.currentTime = targetTime;
      setCurrentTime(targetTime);
    }
  };

  const toggleMute = () => {
    if (videoRef.current) {
      videoRef.current.muted = !isMuted;
      setIsMuted(!isMuted);
    }
  };

  const toggleLoop = () => {
    setIsLooping(!isLooping);
  };

  const toggleFullscreen = () => {
    if (!videoContainerRef.current) return;
    if (!document.fullscreenElement) {
      videoContainerRef.current.requestFullscreen().catch(() => {});
      setIsFullscreen(true);
    } else {
      document.exitFullscreen().catch(() => {});
      setIsFullscreen(false);
    }
  };

  const formatTime = (secs: number) => {
    if (isNaN(secs) || secs < 0) return '00:00';
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  };

  const startWebcam = async () => {
    try {
      if (isCctvStreaming) {
        handleStopCctv();
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'environment' }
      });
      streamRef.current = stream;
      rawVideoUrlRef.current = null;
      setIsWebcam(true);
      setHasProcessedVideo(false);
      setRawVideoUrl(null);
      setAnnotatedVideoUrl(null);
      setVideoMode('original');
      setVideoSource(null);
      setFrameNumber(0);
      frameCountRef.current = 0;
      setDetectedVehicles([]);
      setVehiclesCount(0);
      setPlatesCount(0);
      setOcrSuccessCount(0);

      if (videoRef.current) {
        videoRef.current.src = '';
        videoRef.current.srcObject = stream;
        videoRef.current.play().then(startLoop).catch(() => {});
      }
    } catch {
      alert('Unable to access webcam. Please verify browser camera permissions.');
    }
  };

  const stopWebcam = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsWebcam(false);
  };

  const activeVehicle = detectedVehicles.find(v => v.detectionId === selectedTrackId) || detectedVehicles[0];

  return (
    <>
      {/* Page Header */}
      <div className="mb-6 flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
        <div>
          <p className="mb-1 text-xs font-bold uppercase tracking-[.2em] text-[#F59E0B]">
            VISION / CAMERA &amp; VIDEO DETECTION
          </p>
          <h1 className="text-3xl font-bold tracking-tight text-white">Live Gate view</h1>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          {/* Upload recorded video */}
          <input type="file" ref={fileInputRef} accept="video/*,.mp4,.mov,.avi,.mkv" className="hidden" onChange={handleVideoUpload} />
          <button
            type="button"
            disabled={isUploading}
            onClick={() => fileInputRef.current?.click()}
            data-testid="button-upload-video"
            className="inline-flex items-center gap-2 rounded-lg border border-[#2B3240] bg-[#161B22] hover:bg-[#1E2530] text-gray-200 px-3.5 py-2 text-xs font-medium transition cursor-pointer shadow-sm disabled:opacity-60"
          >
            <Upload className="h-4 w-4 text-[#F59E0B]" />
            {isUploading ? 'Processing Video...' : 'Upload recorded video'}
          </button>

        </div>
      </div>

      {/* CCTV RTSP Connection Bar */}
      <div className="mb-6 rounded-2xl border border-[#1E2738] bg-[#101520] p-4 shadow-xl">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div className="flex-1 min-w-0">
            <div className="flex items-center justify-between mb-2">
              <label className="text-xs font-bold uppercase tracking-wider text-gray-300 flex items-center gap-2">
                <Radio className={`h-4 w-4 ${isCctvStreaming ? 'text-emerald-400 animate-pulse' : 'text-[#F59E0B]'}`} />
                CCTV RTSP / IP CAMERA FEED
              </label>

              {/* Helper options */}
              <div className="flex items-center gap-2 text-[11px] font-mono">
                <button
                  type="button"
                  onClick={() => setRtspUrl('0')}
                  className="text-xs text-emerald-400/90 hover:text-emerald-300 underline cursor-pointer"
                >
                  USB Camera (0)
                </button>
                <span className="text-gray-600">|</span>
                <button
                  type="button"
                  onClick={() => setRtspUrl('')}
                  className="text-xs text-gray-400 hover:text-gray-200 underline cursor-pointer"
                >
                  Clear URL
                </button>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <input
                type="text"
                value={rtspUrl}
                onChange={(e) => setRtspUrl(e.target.value)}
                placeholder="rtsp://username:password@192.168.1.100:554/stream"
                className="w-full rounded-lg border border-[#2B3547] bg-[#161C26] px-3.5 py-2 text-xs font-mono text-white placeholder-gray-500 focus:border-[#F59E0B] focus:outline-none focus:ring-1 focus:ring-[#F59E0B]"
              />
            </div>
          </div>

          <div className="flex items-center gap-3 shrink-0">
            {!isCctvStreaming ? (
              <button
                type="button"
                disabled={isCctvConnecting}
                onClick={() => handleConnectCctv()}
                data-testid="button-connect-cctv"
                className="inline-flex items-center gap-2 rounded-lg bg-emerald-500 hover:bg-emerald-600 text-black px-4 py-2 text-xs font-bold shadow-md transition cursor-pointer disabled:opacity-60"
              >
                {isCctvConnecting ? (
                  <>
                    <RefreshCw className="h-4 w-4 animate-spin" />
                    Connecting...
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4 fill-black" />
                    Connect CCTV
                  </>
                )}
              </button>
            ) : (
              <button
                type="button"
                onClick={handleStopCctv}
                data-testid="button-stop-cctv"
                className="inline-flex items-center gap-2 rounded-lg border border-red-500/40 bg-red-500/10 hover:bg-red-500/20 text-red-200 px-4 py-2 text-xs font-bold transition cursor-pointer"
              >
                <XCircle className="h-4 w-4" />
                Stop CCTV
              </button>
            )}

            {/* Status Badge */}
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-[#1E2738] bg-[#141A24] text-xs font-mono font-semibold">
              {cctvStatus === 'connected' ? (
                <span className="flex items-center gap-1.5 text-emerald-400">
                  <span className="h-2 w-2 rounded-full bg-emerald-400 animate-ping" />
                  CCTV Connected
                </span>
              ) : cctvStatus === 'connecting' ? (
                <span className="flex items-center gap-1.5 text-amber-400">
                  <RefreshCw className="h-3 w-3 animate-spin text-amber-400" />
                  Connecting...
                </span>
              ) : cctvStatus === 'reconnecting' ? (
                <span className="flex items-center gap-1.5 text-orange-400 animate-pulse">
                  <RotateCcw className="h-3 w-3 animate-spin" />
                  CCTV Connection Lost - Reconnecting...
                </span>
              ) : cctvStatus === 'error' ? (
                <span className="flex items-center gap-1.5 text-red-400" title={cctvStatusMessage}>
                  <AlertTriangle className="h-3.5 w-3.5" />
                  {cctvStatusMessage || 'Unable to connect to camera'}
                </span>
              ) : (
                <span className="flex items-center gap-1.5 text-gray-400">
                  <CircleDot className="h-3 w-3" />
                  CCTV Disconnected
                </span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* AI Health Error Banner (If backend models are missing) */}
      {aiHealthError && (
        <div className="mb-6 flex items-start gap-3 rounded-xl border border-red-500/40 bg-red-500/10 p-4 text-sm text-red-200 shadow-lg">
          <AlertTriangle className="h-5 w-5 text-red-400 shrink-0 mt-0.5" />
          <div className="min-w-0 flex-1">
            <p className="font-bold text-red-300">AI MODEL ERROR</p>
            <p className="text-xs mt-1 text-red-200/90">{aiHealthError}</p>
            <div className="mt-2 text-[11px] font-mono text-gray-300 bg-black/40 p-2 rounded">
              Vehicle Model: {aiHealth?.vehicle_model || 'NOT LOADED'} | Plate Model: {aiHealth?.plate_model || 'NOT LOADED'} | OCR: {aiHealth?.ocr || 'NOT LOADED'}
            </div>
          </div>
        </div>
      )}

      {/* Upload & Real Processing Status Banner */}
      {isUploading && (
        <div className="mb-6 flex items-center gap-3 rounded-xl border border-[#F59E0B]/30 bg-[#1A1811] p-4 text-sm text-[#F59E0B] shadow-lg animate-pulse">
          <RefreshCw className="h-5 w-5 animate-spin shrink-0" />
          <div className="min-w-0 flex-1">
            <p className="font-bold">Real AI Pipeline Executing (YOLOv8 + EasyOCR)</p>
            <p className="text-xs text-gray-400 mt-0.5">{uploadStatus}</p>
          </div>
        </div>
      )}

      {/* Main 2-Column Grid */}
      <div className="grid gap-6 xl:grid-cols-[1.5fr_.85fr] items-start">
        {/* Left Column: Video Player + Controls + HUD */}
        <div>
          <div 
            ref={videoContainerRef}
            className="group relative overflow-hidden rounded-2xl border border-[#1A222F] bg-black aspect-[16/10] flex items-center justify-center shadow-2xl"
          >
            {isCctvStreaming ? (
              <div className="relative w-full h-full flex items-center justify-center bg-black">
                <img
                  src="http://localhost:5001/api/cctv/stream"
                  alt="Live CCTV Stream"
                  className="h-full w-full object-contain"
                />

                {/* Top-left HUD badge */}
                <div className="absolute top-3 left-3 z-10 rounded-lg bg-black/85 backdrop-blur-md border border-red-500/30 px-3.5 py-1.5 text-white font-mono text-xs shadow-xl flex flex-col gap-0.5 pointer-events-none">
                  <div className="flex items-center gap-2">
                    <span className="h-2.5 w-2.5 rounded-full bg-red-500 animate-ping" />
                    <span className="font-bold text-xs text-red-400">
                      LIVE CCTV STREAM
                    </span>
                  </div>
                  <div className="text-[11px] text-gray-300">
                    FPS: {fps.toFixed(1)} | Frames: {frameNumber}
                  </div>
                </div>

                {/* Stop button overlay */}
                <div className="absolute top-3 right-3 z-10">
                  <button
                    type="button"
                    onClick={handleStopCctv}
                    className="flex items-center gap-1.5 rounded-lg bg-black/80 backdrop-blur-md border border-red-500/40 px-2.5 py-1 text-[11px] font-mono text-red-300 hover:bg-red-500/20 transition cursor-pointer shadow-xl"
                  >
                    <XCircle className="h-3.5 w-3.5" />
                    Stop Stream
                  </button>
                </div>
              </div>
            ) : videoSource || isWebcam ? (
              <>
                <video
                  ref={videoRef}
                  src={videoSource || undefined}
                  autoPlay
                  playsInline
                  muted={isMuted}
                  loop={isLooping && !isWebcam}
                  onPlay={() => {
                    setIsPlaying(true);
                    startLoop();
                  }}
                  onPause={() => setIsPlaying(false)}
                  onEnded={() => {
                    if (isLooping && !isWebcam && videoRef.current) {
                      videoRef.current.currentTime = 0;
                      videoRef.current.play().catch(() => {});
                    } else {
                      setIsPlaying(false);
                    }
                  }}
                  onTimeUpdate={() => {
                    if (videoRef.current) {
                      setCurrentTime(videoRef.current.currentTime);
                    }
                  }}
                  onLoadedMetadata={() => {
                    if (videoRef.current) {
                      setDuration(videoRef.current.duration || 0);
                      videoRef.current.play().then(() => setIsPlaying(true)).catch(() => {});
                    }
                  }}
                  onError={handleVideoError}
                  className="h-full w-full object-contain cursor-pointer"
                  onClick={togglePlayPause}
                />

                <canvas
                  ref={canvasRef}
                  className="absolute inset-0 h-full w-full pointer-events-none"
                />

                {/* Top-left HUD badge */}
                <div className="absolute top-3 left-3 z-10 rounded-lg bg-black/80 backdrop-blur-md border border-white/10 px-3.5 py-1.5 text-white font-mono text-xs shadow-xl flex flex-col gap-0.5 pointer-events-none">
                  <div className="flex items-center gap-2">
                    <span className={`h-2 w-2 rounded-full ${isUploading ? 'bg-[#F59E0B] animate-spin' : isPlaying ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`} />
                    <span className="font-semibold text-xs">
                      {isWebcam ? 'LIVE CAMERA' : videoMode === 'annotated' ? 'AI ANNOTATED' : 'ORIGINAL SOURCE'}
                    </span>
                  </div>
                  <div className="text-[11px] text-gray-300">
                    FPS: {fps.toFixed(1)} | Frame: {frameNumber}
                  </div>
                </div>

                {/* Top-right Dual Mode Switcher (Annotated vs Original) */}
                {annotatedVideoUrl && rawVideoUrl && !isWebcam && (
                  <div className="absolute top-3 right-3 z-10 flex items-center gap-1.5 rounded-lg bg-black/80 backdrop-blur-md border border-white/10 p-1 shadow-xl">
                    <button
                      type="button"
                      onClick={() => handleSwitchMode('annotated')}
                      className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-mono font-bold transition cursor-pointer ${
                        videoMode === 'annotated'
                          ? 'bg-[#3B82F6] text-white shadow-sm'
                          : 'text-gray-400 hover:text-white hover:bg-white/10'
                      }`}
                    >
                      <Sparkles className="h-3 w-3" />
                      AI Annotated
                    </button>
                    <button
                      type="button"
                      onClick={() => handleSwitchMode('original')}
                      className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-mono font-bold transition cursor-pointer ${
                        videoMode === 'original'
                          ? 'bg-[#F59E0B] text-black shadow-sm'
                          : 'text-gray-400 hover:text-white hover:bg-white/10'
                      }`}
                    >
                      <Eye className="h-3 w-3" />
                      Original Video
                    </button>
                  </div>
                )}

                {/* Zero detection notification */}
                {hasProcessedVideo && detectedVehicles.length === 0 && !isUploading && (
                  <div className="absolute top-14 inset-x-6 z-10 rounded-xl bg-black/85 backdrop-blur-md border border-yellow-500/30 p-2.5 text-center text-xs font-semibold text-yellow-300 pointer-events-none shadow-2xl">
                    No vehicle detected in current video
                  </div>
                )}

                {/* Floating Bottom Video Controls HUD (Visible on hover & during pause) */}
                {!isWebcam && (
                  <div className="absolute bottom-0 inset-x-0 z-20 bg-gradient-to-t from-black/90 via-black/60 to-transparent p-3 pt-6 transition-opacity duration-200 flex flex-col gap-2">
                    {/* Timeline Scrubber */}
                    <div className="flex items-center gap-2">
                      <input
                        type="range"
                        min={0}
                        max={duration || 1}
                        step={0.05}
                        value={currentTime}
                        onChange={handleSeek}
                        aria-label="Video scrubber"
                        className="w-full h-1.5 bg-gray-700/80 rounded-lg appearance-none cursor-pointer accent-[#F59E0B] hover:h-2 transition-all"
                      />
                    </div>

                    {/* Controls Row */}
                    <div className="flex items-center justify-between text-white text-xs font-mono">
                      <div className="flex items-center gap-2.5">
                        <button
                          type="button"
                          onClick={togglePlayPause}
                          className="p-1.5 rounded-lg bg-white/10 hover:bg-white/20 transition cursor-pointer text-white"
                          title={isPlaying ? 'Pause' : 'Play'}
                        >
                          {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4 fill-white" />}
                        </button>

                        <button
                          type="button"
                          onClick={handleReplay}
                          className="p-1.5 rounded-lg bg-white/10 hover:bg-white/20 transition cursor-pointer text-white"
                          title="Replay from start"
                        >
                          <RotateCcw className="h-3.5 w-3.5" />
                        </button>

                        <span className="text-gray-300 text-[11px] ml-1">
                          {formatTime(currentTime)} / {formatTime(duration)}
                        </span>
                      </div>

                      <div className="flex items-center gap-2">
                        <button
                          type="button"
                          onClick={toggleLoop}
                          className={`px-2 py-1 rounded-md text-[11px] font-mono font-semibold transition cursor-pointer flex items-center gap-1 ${
                            isLooping ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40' : 'bg-white/10 text-gray-400'
                          }`}
                          title="Loop Video"
                        >
                          <span>Loop {isLooping ? 'ON' : 'OFF'}</span>
                        </button>

                        <button
                          type="button"
                          onClick={toggleMute}
                          className="p-1.5 rounded-lg bg-white/10 hover:bg-white/20 transition cursor-pointer text-white"
                          title={isMuted ? 'Unmute' : 'Mute'}
                        >
                          {isMuted ? <VolumeX className="h-4 w-4 text-gray-400" /> : <Volume2 className="h-4 w-4 text-emerald-400" />}
                        </button>

                        <button
                          type="button"
                          onClick={toggleFullscreen}
                          className="p-1.5 rounded-lg bg-white/10 hover:bg-white/20 transition cursor-pointer text-white"
                          title="Fullscreen"
                        >
                          <Maximize2 className="h-4 w-4" />
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div className="flex flex-col items-center justify-center p-8 text-center">
                <div className="mb-4 grid h-16 w-16 place-items-center rounded-2xl border border-[#2B3547] bg-[#141A24] text-gray-400">
                  <Radio className="h-8 w-8 text-[#F59E0B]" />
                </div>
                <h3 className="text-base font-bold text-white mb-1">No CCTV stream or video active</h3>
                <p className="text-xs text-gray-400 max-w-sm mb-5">
                  Enter an RTSP URL above and click <strong>Connect CCTV</strong>, or upload a video file to begin ANPR processing.
                </p>
                <div className="flex items-center gap-3">
                  <button
                    type="button"
                    onClick={() => handleConnectCctv()}
                    className="inline-flex items-center gap-2 rounded-lg bg-emerald-500 hover:bg-emerald-600 text-black px-4 py-2 text-xs font-bold transition cursor-pointer"
                  >
                    <Play className="h-3.5 w-3.5 fill-black" />
                    Connect CCTV Stream
                  </button>
                  <button
                    type="button"
                    disabled={isUploading}
                    onClick={() => fileInputRef.current?.click()}
                    className="inline-flex items-center gap-2 rounded-lg border border-[#2B3547] bg-[#161C26] hover:bg-[#1E2530] text-gray-300 px-4 py-2 text-xs font-bold transition cursor-pointer"
                  >
                    <Upload className="h-3.5 w-3.5" />
                    Upload Video File
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Bottom Status bar */}
          <div className="mt-4 flex items-center justify-between gap-2.5 text-xs text-gray-300 font-mono">
            <div className="flex items-center gap-2">
              <span className={`h-2 w-2 rounded-full ${isUploading ? 'bg-[#F59E0B] animate-spin' : hasProcessedVideo ? 'bg-emerald-400 animate-pulse' : 'bg-gray-500'}`} />
              <span>
                {isUploading
                  ? `Processing video with YOLOv8 & EasyOCR...`
                  : hasProcessedVideo
                  ? `Processed: ${fps.toFixed(1)} FPS | Vehicles: ${vehiclesCount} | Plates: ${platesCount} | OCR Success: ${ocrSuccessCount}`
                  : `Ready | Vehicles: 0 | Plates: 0 | OCR Success: 0`}
              </span>
            </div>

            {hasProcessedVideo && rawVideoUrl && (
              <div className="text-[11px] text-gray-400 font-sans hidden sm:block">
                Mode: <span className="text-white font-semibold">{videoMode === 'annotated' ? 'AI Annotated Video' : 'Original Video'}</span>
              </div>
            )}
          </div>
        </div>

        {/* Right Column: DETECTION RESULTS */}
        <div className="rounded-2xl border border-[#1A222F] bg-[#10151E] p-5 shadow-2xl flex flex-col max-h-[640px]">
          <div className="flex items-center justify-between border-b border-[#1A222F] pb-3.5 mb-4">
            <span className="text-xs font-bold uppercase tracking-wider text-gray-300">DETECTION RESULTS</span>
            <span className="text-xs font-mono text-gray-400 font-semibold">{detectedVehicles.length} vehicle(s) detected</span>
          </div>

          <div className="space-y-3 overflow-y-auto flex-1 pr-1">
            {detectedVehicles.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-20 text-center text-gray-500">
                <Car className="h-10 w-10 opacity-30 mb-2" />
                <p className="text-xs font-medium text-gray-400">No vehicles detected in the processed frames.</p>
                <p className="text-[10px] mt-1 text-gray-500">Upload a CCTV video to begin frame-by-frame YOLO &amp; EasyOCR processing.</p>
              </div>
            ) : (
              detectedVehicles.map((item, idx) => {
                const isSelected = selectedTrackId === item.detectionId;
                return (
                  <div 
                    key={item.id + idx} 
                    onClick={() => setSelectedTrackId(item.detectionId)}
                    className={`flex items-center justify-between gap-3 p-3.5 rounded-xl border transition cursor-pointer ${
                      isSelected 
                        ? 'border-[#F59E0B] bg-[#1A1F2B] shadow-md ring-1 ring-[#F59E0B]/30' 
                        : 'border-[#1E2738] bg-[#141A25] hover:border-gray-600'
                    }`}
                  >
                    <div className="flex items-center gap-3 min-w-0">
                      <div className={`grid h-10 w-10 shrink-0 place-items-center rounded-lg border ${
                        isSelected 
                          ? 'bg-[#2A2312] border-[#F59E0B]/50 text-[#F59E0B]' 
                          : 'bg-[#13261E] border-[#1E3E2F] text-emerald-400'
                      }`}>
                        {item.class.toLowerCase() === 'truck' ? (
                          <Truck className="h-5 w-5" />
                        ) : item.class.toLowerCase() === 'bike' || item.class.toLowerCase() === 'motorcycle' ? (
                          <Bike className="h-5 w-5" />
                        ) : item.class.toLowerCase() === 'bus' ? (
                          <Bus className="h-5 w-5" />
                        ) : (
                          <Car className="h-5 w-5" />
                        )}
                      </div>
                      <div className="min-w-0">
                        <div className="text-sm font-bold text-white capitalize flex items-center gap-2">
                          <span>{item.class}</span>
                          {isSelected && (
                            <span className="text-[10px] bg-[#F59E0B]/20 text-[#F59E0B] px-1.5 py-0.2 rounded font-mono">SELECTED</span>
                          )}
                        </div>
                        {item.plate && item.plate !== 'Recognizing...' && item.plate !== 'Requires Manual Review' ? (
                          <div className="text-xs font-mono font-bold text-[#3B82F6] tracking-wide">{item.plate}</div>
                        ) : item.displayPlate && item.displayPlate !== 'Recognizing...' && item.displayPlate !== 'Requires Manual Review' ? (
                          <div className="text-xs font-mono font-bold text-[#3B82F6] tracking-wide">{item.displayPlate}</div>
                        ) : item.status === 'manual_review' || item.displayPlate === 'Requires Manual Review' ? (
                          <div className="text-xs font-mono font-bold text-amber-400 tracking-wide flex items-center gap-1">
                            <AlertTriangle className="h-3.5 w-3.5 text-amber-400 shrink-0 inline" />
                            <span>Requires Manual Review</span>
                          </div>
                        ) : (
                          <div className="text-xs font-mono font-semibold text-blue-300 italic flex items-center gap-1.5">
                            <span className="h-2 w-2 rounded-full bg-blue-400 animate-ping inline-block shrink-0" />
                            <span>Recognizing...</span>
                          </div>
                        )}
                        <div className="text-[10px] font-mono text-gray-400 flex items-center gap-1.5 flex-wrap">
                          <span>{item.timestamp} | Track ID: {item.detectionId}</span>
                          {item.evidenceCount && item.evidenceCount > 1 ? (
                            <span className="text-emerald-400 font-semibold">• Evidence: {item.evidenceCount} frames</span>
                          ) : null}
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center gap-3 shrink-0">
                      <span className="text-sm font-mono font-semibold text-gray-200">{item.confidence.toFixed(2)}</span>
                      {item.thumbnail ? (
                        <img src={item.thumbnail} alt={item.class} className="h-12 w-16 rounded-lg object-cover border border-[#263246] bg-black/60 shadow-sm" />
                      ) : (
                        <div className="h-12 w-16 rounded-lg bg-[#18202D] border border-[#263246] grid place-items-center text-[9px] text-gray-500">No Crop</div>
                      )}
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>
      </div>

      {/* ================================================== */}
      {/* MULTI-FRAME PLATE RECOGNITION SECTION              */}
      {/* ONLY TWO PARTS: 1. INPUT FRAMES | 2. FUSION RESULT */}
      {/* ================================================== */}
      <div className="mt-6 rounded-2xl border border-[#1A222F] bg-[#10151E] p-6 shadow-2xl space-y-6">
        <div className="flex items-center justify-between border-b border-[#1A222F] pb-4 flex-wrap gap-3">
          <div className="flex items-center gap-2.5">
            <span className="h-2.5 w-2.5 rounded-full bg-[#3B82F6] animate-pulse" />
            <h2 className="text-sm font-bold uppercase tracking-wider text-white">
              MULTI-FRAME PLATE RECOGNITION
            </h2>
          </div>
          {detectedVehicles.length > 1 && (
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-xs text-gray-400 font-mono">Vehicle Track:</span>
              {detectedVehicles.map((v) => (
                <button
                  key={v.id}
                  onClick={() => setSelectedTrackId(v.detectionId)}
                  className={`px-3 py-1 text-xs font-mono font-semibold rounded-lg border transition ${
                    (selectedTrackId === v.detectionId || (!selectedTrackId && detectedVehicles[0]?.detectionId === v.detectionId))
                      ? 'bg-[#3B82F6]/20 border-[#3B82F6] text-[#60A5FA]'
                      : 'bg-[#141A25] border-[#1E2738] text-gray-400 hover:text-white hover:border-gray-600'
                  }`}
                >
                  Track #{v.detectionId} ({v.class}{v.plate ? ` • ${v.plate}` : ''})
                </button>
              ))}
            </div>
          )}
        </div>

        {/* 1. INPUT FRAMES */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-bold uppercase tracking-wider text-gray-300">
              INPUT FRAMES
            </h3>
            {activeVehicle && (
              <span className="text-xs font-mono text-gray-400">
                {activeVehicle.framePredictions && activeVehicle.framePredictions.length > 0
                  ? `${activeVehicle.framePredictions.length} frame(s) observed`
                  : ''}
              </span>
            )}
          </div>

          <div className="rounded-xl border border-[#1E2738] bg-[#0B0F15] p-4 font-mono text-xs max-h-60 overflow-y-auto space-y-2">
            {!activeVehicle || !activeVehicle.framePredictions || activeVehicle.framePredictions.length === 0 ? (
              <div className="py-6 text-center text-gray-500 text-xs italic">
                {isUploading
                  ? 'Accumulating real-time OCR frame predictions from YOLO & EasyOCR...'
                  : 'No OCR plate frames detected yet. Upload a CCTV video above.'}
              </div>
            ) : (
              activeVehicle.framePredictions.map((fp, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between py-2 px-3.5 rounded-lg bg-[#141A25]/70 hover:bg-[#141A25] border border-transparent hover:border-[#1E2738] transition"
                >
                  <div className="flex items-center gap-6">
                    <span className="text-gray-400 font-semibold w-24">
                      Frame {String(fp.frame_number).padStart(2, '0')}
                    </span>
                    <span className="font-bold text-gray-100 tracking-wider">
                      {fp.plate_number || fp.raw_text}
                    </span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-[11px] text-gray-500">
                      confidence <span className="text-gray-300 font-semibold">{Number(fp.confidence).toFixed(2)}</span>
                    </span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* 2. FUSION RESULT */}
        <div className="space-y-3 pt-2 border-t border-[#1A222F]">
          <h3 className="text-xs font-bold uppercase tracking-wider text-gray-300">
            FUSION RESULT
          </h3>

          {!activeVehicle || !activeVehicle.framePredictions || activeVehicle.framePredictions.length === 0 ? (
            <div className="rounded-xl border border-[#1E2738] bg-[#0B0F15] p-6 text-center text-gray-500 text-xs italic font-mono">
              Waiting for multi-frame predictions to fuse...
            </div>
          ) : activeVehicle.plate && activeVehicle.plate !== 'Recognizing...' && activeVehicle.plate !== 'Requires Manual Review' ? (
            <div className="rounded-xl border border-[#1E2738] bg-[#0B0F15] p-5 font-mono space-y-3">
              <div className="flex items-baseline gap-3 flex-wrap">
                <span className="text-xs font-semibold text-gray-400">Final Plate:</span>
                <span className="text-xl font-extrabold text-[#3B82F6] tracking-widest bg-[#3B82F6]/10 px-3 py-1 rounded-lg border border-[#3B82F6]/30">
                  {activeVehicle.plate || activeVehicle.displayPlate}
                </span>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2 text-xs">
                <div className="flex items-center gap-2 text-gray-300 bg-[#141A25] p-3 rounded-lg border border-[#1E2738]">
                  <span className="text-gray-400">Confidence:</span>
                  <span className="font-bold text-emerald-400 text-sm">
                    {Math.round(activeVehicle.confidence * 100)}% ({activeVehicle.confidence.toFixed(2)})
                  </span>
                </div>
                <div className="flex items-center gap-2 text-gray-300 bg-[#141A25] p-3 rounded-lg border border-[#1E2738]">
                  <span className="text-gray-400">Multi-Frame Agreement:</span>
                  <span className="font-bold text-emerald-400 text-sm">
                    {activeVehicle.evidenceCount && activeVehicle.frameCount
                      ? `${activeVehicle.evidenceCount} / ${activeVehicle.frameCount} frames (${Math.round((activeVehicle.agreementRatio ?? 0) * 100)}%)`
                      : `${Math.round((activeVehicle.agreementRatio ?? 0) * 100)}%`}
                  </span>
                </div>
              </div>
            </div>
          ) : activeVehicle.status === 'manual_review' || activeVehicle.displayPlate === 'Requires Manual Review' ? (
            <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-5 font-mono space-y-2">
              <div className="flex items-center gap-2 text-amber-400 text-sm font-bold">
                <AlertTriangle className="h-5 w-5" />
                <span>Requires Manual Review</span>
              </div>
              <p className="text-xs text-amber-300/80">
                High OCR conflict across frames. Vehicle plate predictions require operator review.
              </p>
              <div className="text-xs text-gray-400 pt-1">
                Evidence: <span className="text-gray-200 font-semibold">{activeVehicle.evidenceCount || activeVehicle.framePredictions.length} frames</span>
              </div>
            </div>
          ) : (
            <div className="rounded-xl border border-blue-500/30 bg-blue-500/10 p-5 font-mono space-y-2">
              <div className="flex items-center gap-2 text-blue-400 text-sm font-bold">
                <span className="h-2.5 w-2.5 rounded-full bg-blue-400 animate-ping inline-block" />
                <span>Recognizing...</span>
              </div>
              <p className="text-xs text-blue-300/80">
                Accumulating multi-frame evidence...
              </p>
              <div className="text-xs text-gray-400 pt-1">
                Evidence: <span className="text-gray-200 font-semibold">{activeVehicle.framePredictions.length} frame(s)</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );
}

// ── Other Pages (Preserved & Styled) ───────────────────────────

function TripsPage() {
  const qc = useQueryClient();
  const active = useGetActiveTrips({ query: { refetchInterval: 20000, queryKey: getGetActiveTripsQueryKey() } });
  const update = useUpdateTripStatus();
  const [notice, setNotice] = useState('');

  const exit = (id: number) => {
    update.mutate(
      { id, data: { status: 'exited' } },
      {
        onSuccess: () => {
          setNotice('Trip marked exited and active count refreshed.');
          qc.invalidateQueries({ queryKey: getGetActiveTripsQueryKey() });
          qc.invalidateQueries({ queryKey: getGetDashboardSummaryQueryKey() });
        },
        onError: () => setNotice('Unable to update trip status.'),
      }
    );
  };

  return (
    <>
      <PageHeader eyebrow="Trips / monitoring" title="Trip Monitoring" description="Monitor active trips and reconcile open entries before they become overstays." />
      {notice && <Notice kind={notice.includes('Unable') ? 'bad' : 'good'}>{notice}</Notice>}
      <Card title="Active trip register" action={<span className="data-text text-xs text-muted-foreground">{active.data?.length ?? 0} open trips</span>}>
        {active.isLoading ? (
          <LoadingRows count={6} />
        ) : active.isError ? (
          <ErrorState retry={() => active.refetch()} />
        ) : !(active.data ?? []).length ? (
          <EmptyState title="No vehicles inside" detail="The facility is clear. New entries will appear here in real time." />
        ) : (
          <DataTable headers={['S.No', 'Plate', 'Driver / transporter', 'Gate', 'Vehicle', 'Dwell', 'Last event', 'Action']}>
            <tbody>
              {(active.data ?? []).map((t, idx) => (
                <tr key={t.id} className="border-b border-border/50 transition hover:bg-secondary/30">
                  <td className="data-text px-5 py-4 text-xs font-mono text-muted-foreground">{idx + 1}</td>
                  <td className="px-5 py-4">
                    <span className="plate-text text-sm font-bold">{t.plate}</span>
                    <p className="mt-1 text-[10px] text-muted-foreground">{t.purpose}</p>
                  </td>
                  <td className="px-5 py-4">
                    <p className="text-xs">{t.driver}</p>
                    <p className="text-[10px] text-muted-foreground">{t.transporter}</p>
                  </td>
                  <td className="px-5 py-4 text-xs">{t.gate}</td>
                  <td className="px-5 py-4 text-xs">{t.vehicleType}</td>
                  <td className="data-text px-5 py-4 text-xs text-primary">{t.dwellMinutes ?? 0} min</td>
                  <td className="data-text px-5 py-4 text-[10px] text-muted-foreground">{formatTime(t.lastEvent)}</td>
                  <td className="px-5 py-4">
                    <Button variant="secondary" onClick={() => exit(t.id)} disabled={update.isPending} testId={`button-exit-trip-${t.id}`}>
                      <ArrowUpRight className="h-3.5 w-3.5" />
                      Mark exit
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </DataTable>
        )}
      </Card>
    </>
  );
}

function SchedulePage() {
  const qc = useQueryClient();
  const trips = useGetTrips({ query: { queryKey: getGetTripsQueryKey() } });
  const create = useCreateTrip();
  const [open, setOpen] = useState(false);
  const [notice, setNotice] = useState('');
  const [form, setForm] = useState({ plate: '', driver: '', transporter: '', gate: 'North Gate', purpose: 'Delivery', expectedArrival: '', expectedDeparture: '' });
  const update = (k: keyof typeof form, v: string) => setForm({ ...form, [k]: v });

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    create.mutate(
      { data: form },
      {
        onSuccess: () => {
          setOpen(false);
          setForm({ plate: '', driver: '', transporter: '', gate: 'North Gate', purpose: 'Delivery', expectedArrival: '', expectedDeparture: '' });
          setNotice('Trip scheduled.');
          qc.invalidateQueries({ queryKey: getGetTripsQueryKey() });
        },
        onError: () => setNotice('Trip could not be scheduled.'),
      }
    );
  };

  return (
    <>
      <PageHeader eyebrow="Trips / planning" title="Trip Scheduling" description="Put expected movements on the board before the vehicle reaches the lane.">
        <Button onClick={() => setOpen(true)} testId="button-create-trip">
          <Plus className="h-4 w-4" />
          Schedule trip
        </Button>
      </PageHeader>
      {notice && <Notice kind={notice.includes('could') ? 'bad' : 'good'}>{notice}</Notice>}
      <Card title="Scheduled movements" action={<span className="data-text text-xs text-muted-foreground">{trips.data?.length ?? 0} records</span>}>
        {trips.isLoading ? (
          <LoadingRows count={6} />
        ) : trips.isError ? (
          <ErrorState retry={() => trips.refetch()} />
        ) : !(trips.data ?? []).length ? (
          <EmptyState title="Schedule is clear" detail="Create an expected movement to give the gate a head start." />
        ) : (
          <DataTable headers={['S.No', 'Plate', 'Driver', 'Purpose', 'Gate', 'Expected arrival', 'Expected departure', 'Status']}>
            <tbody>
              {(trips.data ?? []).map((t, idx) => (
                <tr key={t.id} className="border-b border-border/50 hover:bg-secondary/30">
                  <td className="data-text px-5 py-4 text-xs font-mono text-muted-foreground">{idx + 1}</td>
                  <td className="plate-text px-5 py-4 text-sm font-bold">{t.plate}</td>
                  <td className="px-5 py-4 text-xs">{t.driver}</td>
                  <td className="px-5 py-4 text-xs text-muted-foreground">{t.purpose}</td>
                  <td className="px-5 py-4 text-xs">{t.gate}</td>
                  <td className="data-text px-5 py-4 text-[10px]">{formatTime(t.expectedArrival)}</td>
                  <td className="data-text px-5 py-4 text-[10px]">{formatTime(t.expectedDeparture)}</td>
                  <td className="px-5 py-4"><StatusPill value={t.status} /></td>
                </tr>
              ))}
            </tbody>
          </DataTable>
        )}
      </Card>
      {open && (
        <Modal title="Schedule a trip" onClose={() => setOpen(false)}>
          <form className="space-y-4" onSubmit={submit}>
            <div className="grid gap-4 sm:grid-cols-2">
              <Field label="Plate" value={form.plate} onChange={v => update('plate', v.toUpperCase())} placeholder="MH12AB1234" required />
              <Field label="Driver" value={form.driver} onChange={v => update('driver', v)} required />
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <Field label="Transporter" value={form.transporter} onChange={v => update('transporter', v)} required />
              <SelectField label="Gate" value={form.gate} onChange={v => update('gate', v)} options={['North Gate', 'South Gate', 'Warehouse Gate']} />
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <SelectField label="Purpose" value={form.purpose} onChange={v => update('purpose', v)} options={['Delivery', 'Pickup', 'Contractor', 'Visitor']} />
              <Field label="Expected arrival" value={form.expectedArrival} onChange={v => update('expectedArrival', v)} type="datetime-local" required />
            </div>
            <Field label="Expected departure" value={form.expectedDeparture} onChange={v => update('expectedDeparture', v)} type="datetime-local" required />
            <div className="flex justify-end gap-2 pt-2">
              <Button variant="ghost" onClick={() => setOpen(false)}>Cancel</Button>
              <Button type="submit" disabled={create.isPending} testId="button-submit-trip">
                {create.isPending ? <Busy label="Scheduling" /> : 'Schedule trip'}
              </Button>
            </div>
          </form>
        </Modal>
      )}
    </>
  );
}

function EventsPage() {
  const [search, setSearch] = useState('');
  const [decision, setDecision] = useState('');
  const [eventType, setEventType] = useState('');
  const params = useMemo(() => ({ search: search || undefined, decision: decision || undefined, eventType: eventType || undefined }), [search, decision, eventType]);
  const events = useGetEvents(params, { query: { queryKey: getGetEventsQueryKey(params) } });

  return (
    <>
      <PageHeader eyebrow="Audit / events" title="Entry/Exit Register" description="Searchable entry and exit history with the read that made the decision." />
      <Card>
        <div className="flex flex-col gap-3 border-b border-border/70 p-5 md:flex-row">
          <SearchBox value={search} onChange={setSearch} placeholder="Plate, gate, transporter" />
          <SelectField label="" value={decision} onChange={setDecision} options={['', 'allowed', 'denied', 'manual review']} />
          <SelectField label="" value={eventType} onChange={setEventType} options={['', 'entry', 'exit']} />
        </div>
        {events.isLoading ? (
          <LoadingRows count={8} />
        ) : events.isError ? (
          <ErrorState retry={() => events.refetch()} />
        ) : !(events.data ?? []).length ? (
          <EmptyState title="No matching events" detail="Try a wider search or clear the decision filters." />
        ) : (
          <DataTable headers={['S.No', 'Timestamp', 'Plate', 'Movement', 'Gate / camera', 'Decision', 'Confidence', 'Vehicle']}>
            <tbody>
              {(events.data ?? []).map((e, idx) => (
                <tr key={e.id} className="border-b border-border/50 hover:bg-secondary/30">
                  <td className="data-text px-5 py-4 text-xs font-mono text-muted-foreground">{idx + 1}</td>
                  <td className="data-text px-5 py-4 text-[10px] text-muted-foreground">{formatTime(e.timestamp)}</td>
                  <td className="px-5 py-4">
                    <span className="plate-text text-sm font-bold">{e.plate}</span>
                    {e.isCorrected && <span className="ml-2 text-[9px] uppercase tracking-wider text-primary">corrected</span>}
                  </td>
                  <td className="px-5 py-4">
                    <span className={`inline-flex items-center gap-1 text-xs ${e.eventType.toLowerCase().includes('entry') ? 'text-accent' : 'text-primary'}`}>
                      {e.eventType.toLowerCase().includes('entry') ? <ArrowDownToLine className="h-3.5 w-3.5" /> : <ArrowUpRight className="h-3.5 w-3.5" />}
                      {e.eventType}
                    </span>
                  </td>
                  <td className="px-5 py-4">
                    <p className="text-xs">{e.gate}</p>
                    <p className="text-[10px] text-muted-foreground">{e.camera}</p>
                  </td>
                  <td className="px-5 py-4"><StatusPill value={e.decision} /></td>
                  <td className="px-5 py-4">
                    <ConfidenceBadge value={e.confidence} />
                  </td>
                  <td className="px-5 py-4 text-xs text-muted-foreground">{e.vehicleType}</td>
                </tr>
              ))}
            </tbody>
          </DataTable>
        )}
      </Card>
    </>
  );
}

function VehiclesPage() {
  const [location] = useLocation();
  const isWhitelist = location === '/whitelist';
  const qc = useQueryClient();
  const vehicles = useGetVehicles({ query: { queryKey: getGetVehiclesQueryKey() } });
  const create = useCreateVehicle();
  const updateVehicle = useUpdateVehicle();
  const remove = useDeleteVehicle();
  const [open, setOpen] = useState(false);
  const [editing, setEditing] = useState<Vehicle | null>(null);
  const [search, setSearch] = useState('');
  const [notice, setNotice] = useState('');
  const [form, setForm] = useState({ plate: '', type: 'Truck', owner: '', transporter: '', authorized: true });

  const filtered = (vehicles.data ?? [])
    .filter(v => (!isWhitelist || v.authorized))
    .filter(v => `${v.plate} ${v.owner} ${v.transporter}`.toLowerCase().includes(search.toLowerCase()));

  const start = (vehicle?: Vehicle) => {
    setEditing(vehicle ?? null);
    setForm(vehicle ? { plate: vehicle.plate, type: vehicle.type, owner: vehicle.owner, transporter: vehicle.transporter, authorized: vehicle.authorized } : { plate: '', type: 'Truck', owner: '', transporter: '', authorized: true });
    setOpen(true);
  };

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const done = () => {
      setOpen(false);
      setNotice(editing ? 'Vehicle updated.' : isWhitelist ? 'Vehicle added to whitelist.' : 'Vehicle added to master.');
      qc.invalidateQueries({ queryKey: getGetVehiclesQueryKey() });
    };
    editing ? updateVehicle.mutate({ id: editing.id, data: form }, { onSuccess: done, onError: () => setNotice('Vehicle update failed.') }) : create.mutate({ data: form }, { onSuccess: done, onError: () => setNotice('Vehicle creation failed.') });
  };

  const destroy = (id: number) => {
    if (confirm('Delete this vehicle from the master?')) {
      remove.mutate({ id }, { onSuccess: () => { setNotice('Vehicle deleted.'); qc.invalidateQueries({ queryKey: getGetVehiclesQueryKey() }); } });
    }
  };

  return (
    <>
      <PageHeader 
        eyebrow={isWhitelist ? "Security / Whitelist" : "Master data / vehicles"} 
        title={isWhitelist ? "Whitelist" : "Vehicle Master"} 
        description={isWhitelist ? "Pre-authorized plates permitted for automatic gate access." : "Authorised plates and their operating context."}
      >
        <Button onClick={() => start()} testId="button-create-vehicle"><Plus className="h-4 w-4" />{isWhitelist ? "Add to whitelist" : "Add vehicle"}</Button>
      </PageHeader>
      {notice && <Notice>{notice}</Notice>}
      <Card>
        <div className="border-b border-border/70 p-5">
          <SearchBox value={search} onChange={setSearch} placeholder="Search plate, owner, transporter" />
        </div>
        {vehicles.isLoading ? (
          <LoadingRows />
        ) : vehicles.isError ? (
          <ErrorState retry={() => vehicles.refetch()} />
        ) : !filtered.length ? (
          <EmptyState title="No vehicles found" detail={search ? 'Try a different plate or owner.' : 'Add the first authorised vehicle to begin.'} />
        ) : (
          <DataTable headers={['S.No', 'Plate', 'Type', 'Owner', 'Transporter', 'Authorisation', 'Status', 'Actions']}>
            <tbody>
              {filtered.map((v, idx) => (
                <tr key={v.id} className="border-b border-border/50 hover:bg-secondary/30">
                  <td className="data-text px-5 py-4 text-xs font-mono text-muted-foreground">{idx + 1}</td>
                  <td className="plate-text px-5 py-4 text-sm font-bold">{v.plate}</td>
                  <td className="px-5 py-4 text-xs">{v.type}</td>
                  <td className="px-5 py-4 text-xs">{v.owner}</td>
                  <td className="px-5 py-4 text-xs text-muted-foreground">{v.transporter}</td>
                  <td className="px-5 py-4"><StatusPill value={v.authorized ? 'Authorized' : 'Blocked'} /></td>
                  <td className="px-5 py-4"><StatusPill value={v.status} /></td>
                  <td className="px-5 py-4">
                    <div className="flex gap-1">
                      <IconButton label="Edit vehicle" testId={`button-edit-vehicle-${v.id}`} onClick={() => start(v)}><Pencil className="h-3.5 w-3.5" /></IconButton>
                      <IconButton label="Delete vehicle" testId={`button-delete-vehicle-${v.id}`} onClick={() => destroy(v.id)}><XCircle className="h-3.5 w-3.5 text-red-300" /></IconButton>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </DataTable>
        )}
      </Card>
      {open && (
        <Modal title={editing ? 'Edit vehicle' : 'Add vehicle'} onClose={() => setOpen(false)}>
          <form className="space-y-4" onSubmit={submit}>
            <div className="grid gap-4 sm:grid-cols-2">
              <Field label="Plate" value={form.plate} onChange={v => setForm({ ...form, plate: v.toUpperCase() })} required />
              <SelectField label="Vehicle type" value={form.type} onChange={v => setForm({ ...form, type: v })} options={['Truck', 'Tanker', 'Trailer', 'Car', 'Bus']} />
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <Field label="Owner" value={form.owner} onChange={v => setForm({ ...form, owner: v })} required />
              <Field label="Transporter" value={form.transporter} onChange={v => setForm({ ...form, transporter: v })} required />
            </div>
            <label className="flex items-center gap-2 text-sm">
              <input type="checkbox" checked={form.authorized} onChange={e => setForm({ ...form, authorized: e.target.checked })} />
              Authorised for entry
            </label>
            <div className="flex justify-end gap-2">
              <Button variant="ghost" onClick={() => setOpen(false)}>Cancel</Button>
              <Button type="submit" disabled={create.isPending || updateVehicle.isPending} testId="button-submit-vehicle">
                {create.isPending || updateVehicle.isPending ? <Busy label="Saving" /> : 'Save vehicle'}
              </Button>
            </div>
          </form>
        </Modal>
      )}
    </>
  );
}

function DriversPage() {
  const qc = useQueryClient();
  const drivers = useGetDrivers({ query: { queryKey: getGetDriversQueryKey() } });
  const create = useCreateDriver();
  const [open, setOpen] = useState(false);
  const [notice, setNotice] = useState('');
  const [form, setForm] = useState({ name: '', license: '', phone: '', vehicle: '' });
  const [search, setSearch] = useState('');
  const filtered = (drivers.data ?? []).filter(d => `${d.name} ${d.license} ${d.vehicle}`.toLowerCase().includes(search.toLowerCase()));

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    create.mutate(
      { data: form },
      {
        onSuccess: () => {
          setOpen(false);
          setForm({ name: '', license: '', phone: '', vehicle: '' });
          setNotice('Driver added to master.');
          qc.invalidateQueries({ queryKey: getGetDriversQueryKey() });
        },
        onError: () => setNotice('Driver creation failed.'),
      }
    );
  };

  return (
    <>
      <PageHeader eyebrow="Master data / people" title="Drivers" description="Keep the person behind the plate visible at every gate.">
        <Button onClick={() => setOpen(true)} testId="button-create-driver"><Plus className="h-4 w-4" />Add driver</Button>
      </PageHeader>
      {notice && <Notice>{notice}</Notice>}
      <Card>
        <div className="border-b border-border/70 p-5">
          <SearchBox value={search} onChange={setSearch} placeholder="Search name, licence, vehicle" />
        </div>
        {drivers.isLoading ? (
          <LoadingRows />
        ) : drivers.isError ? (
          <ErrorState retry={() => drivers.refetch()} />
        ) : !filtered.length ? (
          <EmptyState title="No drivers found" detail="Create a driver record to connect people with gate movements." />
        ) : (
          <DataTable headers={['S.No', 'Driver', 'Licence', 'Phone', 'Assigned vehicle', 'Status']}>
            <tbody>
              {filtered.map((d, idx) => (
                <tr key={d.id} className="border-b border-border/50 hover:bg-secondary/30">
                  <td className="data-text px-5 py-4 text-xs font-mono text-muted-foreground">{idx + 1}</td>
                  <td className="px-5 py-4">
                    <p className="text-sm font-semibold">{d.name}</p>
                    <p className="text-[10px] text-muted-foreground">ID {String(d.id).padStart(4, '0')}</p>
                  </td>
                  <td className="data-text px-5 py-4 text-xs">{d.license}</td>
                  <td className="data-text px-5 py-4 text-xs text-muted-foreground">{d.phone}</td>
                  <td className="plate-text px-5 py-4 text-xs">{d.vehicle || 'Unassigned'}</td>
                  <td className="px-5 py-4"><StatusPill value={d.status} /></td>
                </tr>
              ))}
            </tbody>
          </DataTable>
        )}
      </Card>
      {open && (
        <Modal title="Add driver" onClose={() => setOpen(false)}>
          <form className="space-y-4" onSubmit={submit}>
            <Field label="Full name" value={form.name} onChange={v => setForm({ ...form, name: v })} required />
            <div className="grid gap-4 sm:grid-cols-2">
              <Field label="Licence number" value={form.license} onChange={v => setForm({ ...form, license: v.toUpperCase() })} required />
              <Field label="Phone" value={form.phone} onChange={v => setForm({ ...form, phone: v })} required />
            </div>
            <Field label="Assigned vehicle plate" value={form.vehicle} onChange={v => setForm({ ...form, vehicle: v.toUpperCase() })} placeholder="Optional" />
            <div className="flex justify-end gap-2">
              <Button variant="ghost" onClick={() => setOpen(false)}>Cancel</Button>
              <Button type="submit" disabled={create.isPending} testId="button-submit-driver">
                {create.isPending ? <Busy label="Saving" /> : 'Add driver'}
              </Button>
            </div>
          </form>
        </Modal>
      )}
    </>
  );
}

function AlertsPage() {
  const [location] = useLocation();
  const isWatchlist = location === '/watchlist';
  const qc = useQueryClient();
  const alerts = useGetAlerts({ query: { queryKey: getGetAlertsQueryKey() } });
  const mark = useMarkAlertRead();
  const createVehicle = useCreateVehicle();
  const [filter, setFilter] = useState('Unread');
  const [notice, setNotice] = useState('');
  
  // Quick Add to Vehicle Master state
  const [registerModal, setRegisterModal] = useState<{ open: boolean; plate: string; alertId?: number }>({
    open: false,
    plate: '',
  });
  const [vehicleForm, setVehicleForm] = useState({
    plate: '',
    type: 'Truck',
    owner: '',
    transporter: 'BlueDart Logistics',
    authorized: true,
  });

  const openRegisterVehicle = (plate: string, alertId?: number) => {
    setRegisterModal({ open: true, plate, alertId });
    setVehicleForm({
      plate,
      type: 'Truck',
      owner: '',
      transporter: 'BlueDart Logistics',
      authorized: true,
    });
  };

  const handleSaveVehicle = (e: React.FormEvent) => {
    e.preventDefault();
    createVehicle.mutate(
      { data: vehicleForm },
      {
        onSuccess: () => {
          setNotice(`Vehicle "${vehicleForm.plate}" registered and authorized in Vehicle Master.`);
          if (registerModal.alertId) {
            mark.mutate({ id: registerModal.alertId }, {
              onSuccess: () => qc.invalidateQueries({ queryKey: getGetAlertsQueryKey() }),
            });
          }
          setRegisterModal({ open: false, plate: '' });
          qc.invalidateQueries({ queryKey: getGetVehiclesQueryKey() });
          qc.invalidateQueries({ queryKey: getGetEventsQueryKey() });
          qc.invalidateQueries({ queryKey: getGetAlertsQueryKey() });
        },
        onError: () => setNotice('Failed to register vehicle.'),
      }
    );
  };

  const read = (id: number) => {
    mark.mutate({ id }, {
      onSuccess: () => {
        setNotice('Alert marked as read.');
        qc.invalidateQueries({ queryKey: getGetAlertsQueryKey() });
      },
    });
  };

  const dismissAlert = async (id: number) => {
    try {
      await fetch(`/api/alerts/${id}`, { method: 'DELETE' });
      setNotice('Alert dismissed.');
      qc.invalidateQueries({ queryKey: getGetAlertsQueryKey() });
    } catch {
      setNotice('Failed to dismiss alert.');
    }
  };

  const items = (alerts.data ?? [])
    .filter(a => (!isWatchlist || a.severity === 'critical' || a.type.toLowerCase().includes('watchlist') || a.type.toLowerCase().includes('unauthorized') || a.type.toLowerCase().includes('blacklisted')))
    .filter(a => filter === 'All' || (filter === 'Unread' ? !a.isRead : a.severity === filter.toLowerCase()));

  const unreadCount = (alerts.data ?? []).filter(a => !a.isRead).length;

  return (
    <>
      <PageHeader 
        eyebrow={isWatchlist ? "Security / Watchlist" : "Signals / inbox"} 
        title={isWatchlist ? "Watchlist" : "Alerts"} 
        description={isWatchlist ? "High-priority vehicles flagged for security review or denial of entry." : "Exceptions that need an operator decision, not background noise."}
      >
        <div className="flex items-center gap-3">
          <div className="flex rounded-md border border-border p-0.5">
            {['Unread', 'All', 'Critical', 'Warning'].map(x => (
              <button
                key={x}
                data-testid={`button-filter-alert-${x.toLowerCase()}`}
                onClick={() => setFilter(x)}
                className={`rounded px-3 py-1.5 text-xs font-semibold transition ${filter === x ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:text-foreground'}`}
              >
                {x} {x === 'Unread' && unreadCount > 0 && `(${unreadCount})`}
              </button>
            ))}
          </div>
        </div>
      </PageHeader>

      {notice && <Notice kind={notice.includes('Failed') ? 'bad' : 'good'}>{notice}</Notice>}

      <Card title="Security & Gate Alerts" action={<span className="data-text text-xs text-muted-foreground">{items.length} alerts in view</span>}>
        {alerts.isLoading ? (
          <LoadingRows count={6} />
        ) : alerts.isError ? (
          <ErrorState retry={() => alerts.refetch()} />
        ) : !items.length ? (
          <EmptyState 
            title={filter === 'Unread' ? "All caught up!" : "Inbox clear"} 
            detail={filter === 'Unread' ? "No pending unread alerts. All gate events are reviewed." : "No alerts match this view."} 
          />
        ) : (
          <div className="divide-y divide-border/60">
            {items.map(a => (
              <AlertRow 
                key={a.id} 
                alert={a} 
                onRead={() => read(a.id)} 
                onDismiss={() => dismissAlert(a.id)}
                onAddVehicle={(plate) => openRegisterVehicle(plate, a.id)}
                busy={mark.isPending} 
              />
            ))}
          </div>
        )}
      </Card>

      {/* Quick Add Unknown Vehicle to Master Modal */}
      {registerModal.open && (
        <Modal title={`Add Vehicle "${registerModal.plate}" to Master`} onClose={() => setRegisterModal({ open: false, plate: '' })}>
          <form className="space-y-4" onSubmit={handleSaveVehicle}>
            <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 text-xs text-amber-200">
              This vehicle was detected at the gate as an unregistered plate. Authorizing it will add it to the database and allow seamless gate access in the future.
            </div>
            
            <Field label="License Plate Number" value={vehicleForm.plate} onChange={v => setVehicleForm({ ...vehicleForm, plate: v.toUpperCase() })} required />
            
            <div className="grid gap-4 sm:grid-cols-2">
              <SelectField label="Vehicle Type" value={vehicleForm.type} onChange={v => setVehicleForm({ ...vehicleForm, type: v })} options={['Truck', 'Car', 'Bus', 'Two wheeler', 'Van']} />
              <SelectField label="Transporter / Fleet" value={vehicleForm.transporter} onChange={v => setVehicleForm({ ...vehicleForm, transporter: v })} options={['BlueDart Logistics', 'Rana Freight', 'Eastline Carriers', 'Apex Haulage', 'Guest / Private', 'In-house Fleet']} />
            </div>

            <Field label="Owner / Company Name" value={vehicleForm.owner} onChange={v => setVehicleForm({ ...vehicleForm, owner: v })} placeholder="e.g. Reliable Transports Pvt Ltd" required />

            <div className="flex items-center gap-2 pt-2">
              <input 
                type="checkbox" 
                id="auth-check" 
                checked={vehicleForm.authorized} 
                onChange={e => setVehicleForm({ ...vehicleForm, authorized: e.target.checked })} 
                className="h-4 w-4 rounded border-gray-600 bg-gray-800 text-primary focus:ring-primary"
              />
              <label htmlFor="auth-check" className="text-xs font-semibold text-gray-200 cursor-pointer">
                Authorize for automatic entry (Whitelist)
              </label>
            </div>

            <div className="flex justify-end gap-2 pt-4 border-t border-border/40">
              <Button variant="ghost" onClick={() => setRegisterModal({ open: false, plate: '' })}>Cancel</Button>
              <Button type="submit" disabled={createVehicle.isPending} testId="button-submit-add-vehicle-from-alert">
                {createVehicle.isPending ? <Busy label="Adding" /> : 'Add to Vehicle Master'}
              </Button>
            </div>
          </form>
        </Modal>
      )}
    </>
  );
}

function AlertRow({ 
  alert, 
  onRead, 
  onDismiss,
  onAddVehicle, 
  busy 
}: { 
  alert: Alert; 
  onRead: () => void; 
  onDismiss: () => void;
  onAddVehicle: (plate: string) => void; 
  busy: boolean 
}) {
  const hasPlate = alert.plate && alert.plate !== '—' && alert.plate !== 'UNKNOWN';

  return (
    <div className={`flex items-start gap-4 p-5 transition ${!alert.isRead ? 'bg-primary/[.03]' : 'opacity-75'}`}>
      <span className={`mt-0.5 grid h-9 w-9 place-items-center rounded-lg ${alert.severity.toLowerCase() === 'high' || alert.severity.toLowerCase() === 'critical' ? 'bg-red-400/15 text-red-400 border border-red-500/20' : 'bg-amber-300/12 text-amber-200 border border-amber-500/20'}`}>
        {alert.severity.toLowerCase() === 'high' || alert.severity.toLowerCase() === 'critical' ? <Siren className="h-4 w-4" /> : <AlertTriangle className="h-4 w-4" />}
      </span>
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-2">
          <StatusPill value={alert.severity} />
          <span className="text-[10px] uppercase font-bold tracking-[.12em] text-muted-foreground">{alert.type}</span>
          {!alert.isRead ? (
            <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[9px] font-bold uppercase bg-red-500/15 text-red-300 border border-red-500/30">
              Unread
            </span>
          ) : (
            <span className="inline-flex items-center gap-1 text-[10px] font-semibold text-emerald-400">
              <Check className="h-3 w-3" /> Read
            </span>
          )}
        </div>
        <p className="mt-1.5 text-sm font-semibold text-foreground">{alert.message}</p>
        <p className="mt-1 text-xs text-muted-foreground">
          {hasPlate && <span className="plate-text font-bold text-foreground mr-1.5">{alert.plate}</span>}
          {alert.gate && <span>· {alert.gate} </span>}
          · <span className="data-text">{formatTime(alert.time)}</span>
        </p>
      </div>

      <div className="flex items-center gap-2 shrink-0">
        {/* If vehicle plate detected, offer Add to Vehicle Master */}
        {hasPlate && (
          <Button 
            variant="secondary" 
            onClick={() => onAddVehicle(alert.plate)}
            title="Add this vehicle to the Vehicle Master database"
            testId={`button-add-vehicle-${alert.id}`}
            className="text-xs bg-amber-500/15 hover:bg-amber-500/25 text-amber-300 border border-amber-500/30"
          >
            <Plus className="h-3.5 w-3.5" />
            Add to Master
          </Button>
        )}

        {/* Mark as read button (only if not read) */}
        {!alert.isRead && (
          <Button 
            variant="ghost" 
            onClick={onRead} 
            disabled={busy} 
            testId={`button-mark-alert-read-${alert.id}`}
            className="text-xs border border-border/60 hover:bg-secondary"
          >
            <Check className="h-3.5 w-3.5 text-emerald-400" />
            Mark as read
          </Button>
        )}

        {/* Dismiss alert */}
        <IconButton label="Dismiss alert" onClick={onDismiss}>
          <XCircle className="h-4 w-4 text-gray-400 hover:text-red-400" />
        </IconButton>
      </div>
    </div>
  );
}

function ReviewPage() {
  const qc = useQueryClient();
  const queue = useGetReviewQueue({ query: { queryKey: getGetReviewQueueQueryKey() } });
  const correct = useCorrectPlate();
  const [selected, setSelected] = useState<any>(null);
  const [plate, setPlate] = useState('');
  const [notice, setNotice] = useState('');

  const open = (item: any) => {
    setSelected(item);
    setPlate(item.plate || item.rawText);
  };

  const submit = () => {
    if (!selected) return;
    correct.mutate(
      { id: selected.id, data: { correctedPlate: plate.toUpperCase() } },
      {
        onSuccess: () => {
          setSelected(null);
          setNotice('Plate correction saved to the audit trail.');
          qc.invalidateQueries({ queryKey: getGetReviewQueueQueryKey() });
          qc.invalidateQueries({ queryKey: getGetEventsQueryKey() });
        },
        onError: () => setNotice('Correction could not be saved.'),
      }
    );
  };

  return (
    <>
      <PageHeader eyebrow="Exceptions / review" title="Manual Review" description="Resolve uncertain reads with an explicit correction before the record leaves the queue." />
      {notice && <Notice kind={notice.includes('could') ? 'bad' : 'good'}>{notice}</Notice>}
      <Card title="Review queue" action={<span className="data-text text-xs text-muted-foreground">{queue.data?.length ?? 0} pending</span>}>
        {queue.isLoading ? (
          <LoadingRows count={5} />
        ) : queue.isError ? (
          <ErrorState retry={() => queue.refetch()} />
        ) : !(queue.data ?? []).length ? (
          <EmptyState title="Queue is clear" detail="No uncertain reads are waiting for an operator." />
        ) : (
          <DataTable headers={['S.No', 'Captured plate', 'Confidence', 'Gate', 'Timestamp', 'Reason', 'Status', 'Action']}>
            <tbody>
              {(queue.data ?? []).map((item, idx) => (
                <tr key={item.id} className="border-b border-border/50 hover:bg-secondary/30">
                  <td className="data-text px-5 py-4 text-xs font-mono text-muted-foreground">{idx + 1}</td>
                  <td className="px-5 py-4">
                    <p className="plate-text text-sm font-bold">{item.plate}</p>
                    <p className="plate-text mt-1 text-[10px] text-muted-foreground">raw {item.rawText}</p>
                  </td>
                  <td className="px-5 py-4">
                    <ConfidenceBadge value={item.confidence} />
                  </td>
                  <td className="px-5 py-4 text-xs">{item.gate}</td>
                  <td className="data-text px-5 py-4 text-[10px] text-muted-foreground">{formatTime(item.timestamp)}</td>
                  <td className="px-5 py-4 text-xs text-muted-foreground">{item.reason}</td>
                  <td className="px-5 py-4"><StatusPill value={item.status} /></td>
                  <td className="px-5 py-4">
                    <Button variant="secondary" onClick={() => open(item)} testId={`button-review-item-${item.id}`}>
                      <Pencil className="h-3.5 w-3.5" />Review
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </DataTable>
        )}
      </Card>
      {selected && (
        <Modal title="Resolve plate read" onClose={() => setSelected(null)}>
          <div className="rounded-lg border border-border bg-background/35 p-4">
            <p className="text-[10px] uppercase tracking-[.14em] text-muted-foreground">Frame consensus</p>
            <p className="plate-text mt-2 text-2xl font-bold text-amber-200">{selected.rawText}</p>
            <p className="mt-1 text-xs text-muted-foreground">{selected.reason} · {formatConfidence(selected.confidence)} confidence</p>
          </div>
          <div className="mt-5">
            <Field label="Corrected plate" value={plate} onChange={v => setPlate(v.toUpperCase())} required />
          </div>
          <div className="mt-5 flex justify-end gap-2">
            <Button variant="ghost" onClick={() => setSelected(null)}>Cancel</Button>
            <Button onClick={submit} disabled={correct.isPending || !plate} testId="button-save-correction">
              {correct.isPending ? <Busy label="Saving" /> : 'Save correction'}
            </Button>
          </div>
        </Modal>
      )}
    </>
  );
}

const GATE_CHART_COLORS = ['#F59E0B', '#3B82F6', '#10B981', '#8B5CF6', '#EC4899', '#06B6D4'];

const DECISION_COLOR_MAP: Record<string, string> = {
  allow: '#10B981',
  granted: '#10B981',
  approved: '#10B981',
  manual_review: '#F59E0B',
  review: '#F59E0B',
  pending: '#F59E0B',
  deny: '#EF4444',
  denied: '#EF4444',
  blocked: '#EF4444',
};

const CustomChartTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="rounded-lg border border-border/80 bg-[#0F172A]/95 p-3 text-xs shadow-2xl backdrop-blur-md">
        <p className="font-semibold text-white mb-1.5">{label || payload[0].payload?.label || payload[0].name}</p>
        {payload.map((entry: any, index: number) => (
          <div key={`item-${index}`} className="flex items-center justify-between gap-4 font-mono text-[11px] py-0.5">
            <span className="flex items-center gap-1.5" style={{ color: entry.color || entry.fill || '#94A3B8' }}>
              <span className="h-2 w-2 rounded-full" style={{ backgroundColor: entry.color || entry.fill || '#94A3B8' }} />
              {entry.name || 'Count'}:
            </span>
            <span className="font-bold text-white">{entry.value}</span>
          </div>
        ))}
      </div>
    );
  }
  return null;
};

function GateVolumePieChart({ points }: { points: { label: string; value: number; secondary: number | null }[] }) {
  const total = points.reduce((acc, p) => acc + p.value, 0);
  return (
    <Card title="Gate Volume" action={<span className="data-text text-xs text-muted-foreground">{total} total movements</span>}>
      <div className="p-5">
        {points.length ? (
          <div>
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <RechartsTooltip content={<CustomChartTooltip />} />
                  <Pie
                    data={points}
                    dataKey="value"
                    nameKey="label"
                    cx="50%"
                    cy="50%"
                    innerRadius={55}
                    outerRadius={85}
                    paddingAngle={4}
                    stroke="none"
                  >
                    {points.map((_, idx) => (
                      <Cell key={`cell-${idx}`} fill={GATE_CHART_COLORS[idx % GATE_CHART_COLORS.length]} />
                    ))}
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="mt-3 flex flex-wrap items-center justify-center gap-3 border-t border-border/50 pt-3">
              {points.map((p, idx) => {
                const color = GATE_CHART_COLORS[idx % GATE_CHART_COLORS.length];
                const pct = total ? Math.round((p.value / total) * 100) : 0;
                return (
                  <div key={p.label} className="flex items-center gap-1.5 text-xs">
                    <span className="h-2.5 w-2.5 rounded-full shrink-0" style={{ backgroundColor: color }} />
                    <span className="text-muted-foreground">{p.label}:</span>
                    <span className="font-bold text-foreground">{p.value}</span>
                    <span className="text-[10px] text-muted-foreground">({pct}%)</span>
                  </div>
                );
              })}
            </div>
          </div>
        ) : (
          <EmptyState title="No gate data" detail="Gate traffic will appear once vehicle events are logged." />
        )}
      </div>
    </Card>
  );
}

function TransporterVolumeBarChart({ points }: { points: { label: string; value: number; secondary: number | null }[] }) {
  return (
    <Card title="Transporter Volume" action={<span className="data-text text-xs text-muted-foreground">{points.length} transporters</span>}>
      <div className="p-5">
        {points.length ? (
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={points} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                <defs>
                  <linearGradient id="transporterBarGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#3B82F6" stopOpacity={1} />
                    <stop offset="100%" stopColor="#1D4ED8" stopOpacity={0.7} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3} vertical={false} />
                <XAxis 
                  dataKey="label" 
                  tick={{ fill: '#94A3B8', fontSize: 11 }} 
                  axisLine={{ stroke: '#334155' }}
                  tickLine={false}
                />
                <YAxis 
                  tick={{ fill: '#94A3B8', fontSize: 10 }} 
                  axisLine={{ stroke: '#334155' }}
                  tickLine={false}
                />
                <RechartsTooltip content={<CustomChartTooltip />} />
                <Bar 
                  dataKey="value" 
                  name="Vehicles" 
                  fill="url(#transporterBarGrad)" 
                  radius={[6, 6, 0, 0]}
                  barSize={32}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <EmptyState title="No transporter data" detail="Transporter volume will appear as fleet vehicles are registered." />
        )}
      </div>
    </Card>
  );
}

function DwellTrendAreaChart({ points }: { points: { label: string; value: number; secondary: number | null }[] }) {
  const avgDwell = points.length ? Math.round(points.reduce((acc, p) => acc + p.value, 0) / points.length) : 0;
  return (
    <Card title="Dwell Time Trend" action={<span className="data-text text-xs text-amber-400 font-semibold">{avgDwell} min avg dwell</span>}>
      <div className="p-5">
        {points.length ? (
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={points} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                <defs>
                  <linearGradient id="dwellAreaGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#F59E0B" stopOpacity={0.4} />
                    <stop offset="95%" stopColor="#F59E0B" stopOpacity={0.0} />
                  </linearGradient>
                  <linearGradient id="secondaryDwellGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#06B6D4" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#06B6D4" stopOpacity={0.0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3} vertical={false} />
                <XAxis 
                  dataKey="label" 
                  tick={{ fill: '#94A3B8', fontSize: 11 }} 
                  axisLine={{ stroke: '#334155' }}
                  tickLine={false}
                />
                <YAxis 
                  unit="m" 
                  tick={{ fill: '#94A3B8', fontSize: 10 }} 
                  axisLine={{ stroke: '#334155' }}
                  tickLine={false}
                />
                <RechartsTooltip content={<CustomChartTooltip />} />
                <Area 
                  type="monotone" 
                  dataKey="value" 
                  name="Avg Dwell (min)" 
                  stroke="#F59E0B" 
                  strokeWidth={2.5}
                  fill="url(#dwellAreaGrad)" 
                  activeDot={{ r: 5, fill: '#F59E0B', stroke: '#fff', strokeWidth: 1.5 }}
                />
                {points.some(p => p.secondary !== null) && (
                  <Area 
                    type="monotone" 
                    dataKey="secondary" 
                    name="Peak Dwell (min)" 
                    stroke="#06B6D4" 
                    strokeWidth={2}
                    strokeDasharray="4 4"
                    fill="url(#secondaryDwellGrad)" 
                    activeDot={{ r: 4, fill: '#06B6D4' }}
                  />
                )}
              </AreaChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <EmptyState title="No dwell trend data" detail="Dwell metrics will graph once trips complete." />
        )}
      </div>
    </Card>
  );
}

function DecisionsDonutChart({ points }: { points: { label: string; value: number; secondary: number | null }[] }) {
  const total = points.reduce((acc, p) => acc + p.value, 0);
  const allowItem = points.find(p => p.label.toLowerCase().includes('allow')) || { value: 0 };
  const approvalRate = total ? Math.round((allowItem.value / total) * 100) : 0;

  return (
    <Card title="Gate Decisions" action={<span className="data-text text-xs text-emerald-400 font-semibold">{approvalRate}% Approved</span>}>
      <div className="p-5">
        {points.length ? (
          <div>
            <div className="relative h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <RechartsTooltip content={<CustomChartTooltip />} />
                  <Pie
                    data={points}
                    dataKey="value"
                    nameKey="label"
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={85}
                    paddingAngle={5}
                    stroke="none"
                  >
                    {points.map((p, idx) => {
                      const lk = p.label.toLowerCase();
                      const color = DECISION_COLOR_MAP[lk] || (lk.includes('allow') ? '#10B981' : lk.includes('deny') ? '#EF4444' : '#F59E0B');
                      return <Cell key={`cell-dec-${idx}`} fill={color} />;
                    })}
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
              <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center text-center">
                <span className="text-2xl font-bold font-mono text-white">{approvalRate}%</span>
                <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Approval Rate</span>
              </div>
            </div>
            <div className="mt-3 grid grid-cols-3 gap-2 border-t border-border/50 pt-3 text-center">
              {points.map(p => {
                const lk = p.label.toLowerCase();
                const color = DECISION_COLOR_MAP[lk] || (lk.includes('allow') ? '#10B981' : lk.includes('deny') ? '#EF4444' : '#F59E0B');
                const pct = total ? Math.round((p.value / total) * 100) : 0;
                return (
                  <div key={p.label} className="rounded-lg bg-secondary/40 p-2 border border-border/40">
                    <p className="text-[11px] font-semibold" style={{ color }}>{p.label}</p>
                    <p className="data-text text-sm font-bold text-white mt-0.5">{p.value}</p>
                    <p className="text-[10px] text-muted-foreground font-mono">{pct}%</p>
                  </div>
                );
              })}
            </div>
          </div>
        ) : (
          <EmptyState title="No decision data" detail="Access decisions will record as vehicles reach the gate." />
        )}
      </div>
    </Card>
  );
}

function ReportsPage() {
  const reports = useGetReportsOverview();
  const r = reports.data;
  return (
    <>
      <PageHeader eyebrow="Intelligence / reports" title="Operational reports" description="A compact read on volume, decisions, dwell, and recognition quality." />
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <Metric label="Repeat visitors" value={r?.repeatVisitors ?? '—'} detail="Returning plates" accent="accent" />
        <Metric label="Overstays" value={r?.overstays ?? '—'} detail="Past expected departure" accent="danger" />
        <Metric label="Corrected reads" value={r?.correctedReads ?? '—'} detail="Manual intervention" />
        <Metric label="Total reads" value={r?.totalReads ?? '—'} detail="Selected period" accent="good" />
      </div>
      {reports.isLoading ? (
        <div className="mt-6 grid gap-6 md:grid-cols-2">
          {[1, 2].map(i => <div key={i} className="h-72 animate-pulse rounded-xl bg-card" />)}
        </div>
      ) : reports.isError ? (
        <Card className="mt-6"><ErrorState retry={() => reports.refetch()} /></Card>
      ) : (
        <div className="mt-6 grid gap-6 md:grid-cols-2">
          <GateVolumePieChart points={r?.gateVolume ?? []} />
          <TransporterVolumeBarChart points={r?.transporterVolume ?? []} />
          <DwellTrendAreaChart points={r?.dwellTrend ?? []} />
          <DecisionsDonutChart points={r?.decisions ?? []} />
        </div>
      )}
    </>
  );
}

function CamerasPage() {
  const [, setLocation] = useLocation();
  const qc = useQueryClient();
  const cameras = useGetCameras({ query: { queryKey: getGetCamerasQueryKey() } });
  const [open, setOpen] = useState(false);
  const [editingCamera, setEditingCamera] = useState<any>(null);
  const [notice, setNotice] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [testingCamId, setTestingCamId] = useState<number | null>(null);
  const [form, setForm] = useState({
    name: '',
    gate: 'North Gate',
    direction: 'Entry',
    rtspUrl: '',
  });

  const openCreate = () => {
    setEditingCamera(null);
    setForm({ name: '', gate: 'North Gate', direction: 'Entry', rtspUrl: '' });
    setOpen(true);
  };

  const openEdit = (cam: any) => {
    setEditingCamera(cam);
    setForm({
      name: cam.name,
      gate: cam.gate,
      direction: cam.direction || 'Entry',
      rtspUrl: cam.rtspUrl || '',
    });
    setOpen(true);
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    try {
      if (editingCamera) {
        const res = await fetch(`/api/cameras/${editingCamera.id}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(form),
        });
        if (!res.ok) throw new Error('Failed to update camera');
        setNotice(`Camera "${form.name}" updated successfully.`);
      } else {
        const res = await fetch('/api/cameras', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(form),
        });
        if (!res.ok) throw new Error('Failed to register camera');
        setNotice(`Camera "${form.name}" registered successfully.`);
      }
      setOpen(false);
      qc.invalidateQueries({ queryKey: getGetCamerasQueryKey() });
    } catch (err: any) {
      setNotice(err.message || 'Camera operation failed.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const deleteCam = async (id: number, name: string) => {
    if (!confirm(`Are you sure you want to unregister camera "${name}"?`)) return;
    try {
      const res = await fetch(`/api/cameras/${id}`, { method: 'DELETE' });
      if (!res.ok) throw new Error('Failed to delete camera');
      setNotice(`Camera "${name}" removed from registry.`);
      qc.invalidateQueries({ queryKey: getGetCamerasQueryKey() });
    } catch (err: any) {
      setNotice(err.message || 'Delete failed.');
    }
  };

  const testConnection = async (cam: any) => {
    setTestingCamId(cam.id);
    setNotice(`Testing connection to ${cam.name}...`);
    try {
      const res = await fetch(`/api/cameras/${cam.id}/test`, { method: 'POST' });
      const data = await res.json();
      if (data.success) {
        setNotice(`🟢 ${cam.name} is ONLINE & reachable: ${data.message}`);
      } else {
        setNotice(`🔴 ${cam.name} connection test failed: ${data.message}`);
      }
      qc.invalidateQueries({ queryKey: getGetCamerasQueryKey() });
    } catch (err: any) {
      setNotice(`🔴 Error testing connection: ${err.message}`);
    } finally {
      setTestingCamId(null);
    }
  };

  const activateStream = async (cam: any) => {
    const url = (cam.rtspUrl || '').trim();
    if (!url) {
      setNotice(`Please configure an RTSP stream URL for "${cam.name}" first.`);
      return;
    }
    setNotice(`Connecting to live stream for ${cam.name}...`);
    try {
      const res = await fetch('http://localhost:5001/api/cctv/connect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rtsp_url: url }),
      });
      const data = await res.json();
      if (data.success && data.status !== 'error') {
        setNotice(`Connected to ${cam.name}. Redirecting to Live Gate View...`);
        setTimeout(() => setLocation('/'), 400);
      } else {
        setNotice(`Failed to connect to stream: ${data.message}`);
      }
    } catch (err: any) {
      setNotice(`Stream connection failed: ${err.message}`);
    }
  };

  return (
    <>
      <PageHeader eyebrow="Infrastructure / cameras" title="Camera Management" description="Register, configure, test, and manage real-time CCTV feeds for all gate lanes.">
        <Button onClick={openCreate} testId="button-create-camera">
          <Plus className="h-4 w-4" />Register camera
        </Button>
      </PageHeader>

      {notice && <Notice kind={notice.includes('🔴') || notice.includes('failed') ? 'bad' : 'good'}>{notice}</Notice>}

      <Card title="Connected CCTV Cameras & Lane Registry" action={<span className="data-text text-xs text-muted-foreground">{cameras.data?.length ?? 0} registered</span>}>
        {cameras.isLoading ? (
          <LoadingRows count={5} />
        ) : cameras.isError ? (
          <ErrorState retry={() => cameras.refetch()} />
        ) : !(cameras.data ?? []).length ? (
          <EmptyState 
            title="No cameras registered" 
            detail="Register your first CCTV or IP camera to start automated license plate recognition." 
            action={<Button onClick={openCreate} variant="secondary" testId="button-register-empty-camera">Register camera</Button>} 
          />
        ) : (
          <DataTable headers={['S.No', 'Camera & Stream URL', 'Gate', 'Direction', 'Status', 'Last seen', 'Live Actions']}>
            <tbody>
              {(cameras.data ?? []).map((c: any, idx) => (
                <tr key={c.id} className="border-b border-border/50 hover:bg-secondary/30">
                  <td className="data-text px-5 py-4 text-xs font-mono text-muted-foreground">{idx + 1}</td>
                  <td className="px-5 py-4">
                    <div className="flex items-center gap-3">
                      <span className="grid h-9 w-9 place-items-center rounded-lg bg-accent/10 text-accent">
                        <Camera className="h-4 w-4" />
                      </span>
                      <div className="min-w-0">
                        <p className="text-sm font-semibold text-foreground">{c.name}</p>
                        <p className="data-text text-[10px] text-muted-foreground">CAM-{String(c.id).padStart(3, '0')}</p>
                        {c.rtspUrl ? (
                          <p className="font-mono text-[10px] text-amber-400/80 truncate max-w-[240px] mt-0.5" title={c.rtspUrl}>
                            {c.rtspUrl}
                          </p>
                        ) : (
                          <p className="text-[10px] text-gray-500 italic mt-0.5">No RTSP stream configured</p>
                        )}
                      </div>
                    </div>
                  </td>
                  <td className="px-5 py-4 text-xs font-medium">{c.gate}</td>
                  <td className="px-5 py-4 text-xs">
                    <span className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider ${
                      c.direction?.toLowerCase() === 'entry' ? 'bg-emerald-500/10 text-emerald-300 border border-emerald-500/20' :
                      c.direction?.toLowerCase() === 'exit' ? 'bg-blue-500/10 text-blue-300 border border-blue-500/20' :
                      'bg-purple-500/10 text-purple-300 border border-purple-500/20'
                    }`}>
                      {c.direction || 'Entry'}
                    </span>
                  </td>
                  <td className="px-5 py-4"><StatusPill value={c.status} /></td>
                  <td className="data-text px-5 py-4 text-[10px] text-muted-foreground">{formatTime(c.lastSeen)}</td>
                  <td className="px-5 py-4">
                    <div className="flex items-center gap-1.5">
                      {/* Activate Live Stream */}
                      <button
                        type="button"
                        onClick={() => activateStream(c)}
                        title="Stream & Connect Live Feed"
                        className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-md bg-emerald-500/15 hover:bg-emerald-500/25 text-emerald-300 text-[11px] font-semibold transition cursor-pointer border border-emerald-500/30"
                      >
                        <Play className="h-3 w-3 fill-emerald-300" />
                        Stream
                      </button>

                      {/* Test Connection / Ping */}
                      <button
                        type="button"
                        disabled={testingCamId === c.id}
                        onClick={() => testConnection(c)}
                        title="Test CCTV Connection / Ping"
                        className="inline-flex items-center gap-1 px-2 py-1.5 rounded-md bg-secondary hover:bg-secondary/80 text-gray-300 text-[11px] font-medium transition cursor-pointer border border-border/60 disabled:opacity-60"
                      >
                        {testingCamId === c.id ? (
                          <RefreshCw className="h-3 w-3 animate-spin text-amber-400" />
                        ) : (
                          <Activity className="h-3 w-3 text-blue-400" />
                        )}
                        Ping
                      </button>

                      {/* Edit Camera */}
                      <IconButton label="Edit camera" onClick={() => openEdit(c)}>
                        <Pencil className="h-3.5 w-3.5 text-gray-300" />
                      </IconButton>

                      {/* Delete Camera */}
                      <IconButton label="Delete camera" onClick={() => deleteCam(c.id, c.name)}>
                        <XCircle className="h-3.5 w-3.5 text-red-400" />
                      </IconButton>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </DataTable>
        )}
      </Card>

      {open && (
        <Modal title={editingCamera ? 'Edit CCTV Camera' : 'Register CCTV Camera'} onClose={() => setOpen(false)}>
          <form className="space-y-4" onSubmit={submit}>
            <Field 
              label="Camera name" 
              value={form.name} 
              onChange={v => setForm({ ...form, name: v })} 
              placeholder="e.g. North Gate Inbound Lane 01" 
              required 
            />
            
            <Field 
              label="RTSP Stream URL / Device Index" 
              value={form.rtspUrl} 
              onChange={v => setForm({ ...form, rtspUrl: v })} 
              placeholder="rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101 or 0 for USB" 
            />

            <div className="grid gap-4 sm:grid-cols-2">
              <SelectField 
                label="Assigned Gate" 
                value={form.gate} 
                onChange={v => setForm({ ...form, gate: v })} 
                options={['North Gate', 'South Gate', 'Warehouse Gate', 'Gate 01', 'Gate 02', 'Gate 03']} 
              />
              <SelectField 
                label="Movement Direction" 
                value={form.direction} 
                onChange={v => setForm({ ...form, direction: v })} 
                options={['Entry', 'Exit', 'Both']} 
              />
            </div>

            <div className="flex justify-end gap-2 pt-2 border-t border-border/40">
              <Button variant="ghost" onClick={() => setOpen(false)}>Cancel</Button>
              <Button type="submit" disabled={isSubmitting} testId="button-submit-camera">
                {isSubmitting ? <Busy label="Saving" /> : editingCamera ? 'Update camera' : 'Register camera'}
              </Button>
            </div>
          </form>
        </Modal>
      )}
    </>
  );
}



function Info({ label, value, className }: { label: string; value: string; className?: string }) {
  return (
    <div className={`rounded-lg border border-border/70 bg-background/35 p-3 ${className || ''}`}>
      <p className="text-[10px] uppercase tracking-[.14em] text-muted-foreground">{label}</p>
      <p className="data-text mt-1 text-xs font-semibold">{value}</p>
    </div>
  );
}

function formatConfidence(conf?: number | null): string {
  if (conf == null || Number.isNaN(conf)) return '—';
  const val = conf <= 1.0 ? conf * 100 : conf;
  return `${Math.round(val)}%`;
}

function ConfidenceBadge({ value }: { value?: number | null }) {
  if (value == null || Number.isNaN(value)) return <span className="text-muted-foreground">—</span>;
  const pct = Math.round(value <= 1.0 ? value * 100 : value);
  const colorClass = 
    pct >= 90 ? 'bg-emerald-500/10 text-emerald-300 border-emerald-500/20' :
    pct >= 75 ? 'bg-amber-500/10 text-amber-300 border-amber-500/20' :
    'bg-red-500/10 text-red-300 border-red-500/20';

  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded font-mono text-xs font-bold border ${colorClass}`}>
      {pct}%
    </span>
  );
}

function formatTime(value?: string | null) {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return `${date.toLocaleDateString('en-IN', { day: '2-digit', month: 'short' })} ${date.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', hour12: false })}`;
}

export default App;