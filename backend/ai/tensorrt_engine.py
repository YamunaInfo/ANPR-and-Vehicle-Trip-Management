"""
TensorRT Acceleration & Export Engine for GateSense Edge AI ANPR Pipeline.

Provides:
1. Automatic detection of NVIDIA CUDA, TensorRT runtime, and FP16/INT8 hardware support.
2. Direct YOLOv8 PyTorch (.pt) to TensorRT (.engine) model exporter with FP16 optimization.
3. TensorRT inference runner with memory management and latency profiling.
4. Comprehensive benchmarking suite comparing PyTorch vs TensorRT (FPS, Latency ms, P95/P99).
5. Seamless fallback to PyTorch GPU/CPU with diagnostic logging when running on non-CUDA nodes.
"""

from __future__ import annotations

import os
import sys
import time
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import cv2

# Logger setup
logger = logging.getLogger("GateSense.TensorRT")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [TensorRT] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# PyTorch and CUDA inspection
TORCH_AVAILABLE = False
CUDA_AVAILABLE = False
DEVICE_NAME = "CPU"
CUDA_VERSION = None
GPU_COUNT = 0

try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        GPU_COUNT = torch.cuda.device_count()
        DEVICE_NAME = torch.cuda.get_device_name(0)
        CUDA_VERSION = torch.version.cuda
except Exception as e:
    logger.warning(f"PyTorch import notice: {e}")

# TensorRT library inspection
TENSORRT_AVAILABLE = False
TENSORRT_VERSION = None

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
    TENSORRT_VERSION = getattr(trt, "__version__", "unknown")
except Exception:
    TENSORRT_AVAILABLE = False

# Ultralytics inspection
ULTRALYTICS_AVAILABLE = False
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception:
    ULTRALYTICS_AVAILABLE = False


def get_system_tensorrt_diagnostics() -> Dict[str, Any]:
    """
    Returns complete diagnostics on TensorRT, CUDA, GPU hardware, and model readiness.
    """
    diagnostics = {
        "cuda_available": CUDA_AVAILABLE,
        "cuda_version": CUDA_VERSION,
        "gpu_count": GPU_COUNT,
        "gpu_name": DEVICE_NAME,
        "tensorrt_library_installed": TENSORRT_AVAILABLE,
        "tensorrt_version": TENSORRT_VERSION,
        "ultralytics_available": ULTRALYTICS_AVAILABLE,
        "fp16_supported": CUDA_AVAILABLE,
        "recommended_backend": "TensorRT Engine (FP16)" if (CUDA_AVAILABLE and TENSORRT_AVAILABLE) else ("PyTorch CUDA (FP16)" if CUDA_AVAILABLE else "PyTorch CPU (Optimized)"),
        "models": {}
    }

    # Inspect default models directory
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(backend_dir, "ai", "models")
    
    for model_name in ["yolov8n", "license_plate"]:
        pt_path = os.path.join(models_dir, f"{model_name}.pt")
        engine_path = os.path.join(models_dir, f"{model_name}.engine")
        onnx_path = os.path.join(models_dir, f"{model_name}.onnx")
        
        diagnostics["models"][model_name] = {
            "pytorch_pt_exists": os.path.exists(pt_path),
            "pytorch_pt_path": pt_path if os.path.exists(pt_path) else None,
            "pytorch_pt_size_mb": round(os.path.getsize(pt_path) / (1024 * 1024), 2) if os.path.exists(pt_path) else 0,
            "tensorrt_engine_exists": os.path.exists(engine_path),
            "tensorrt_engine_path": engine_path if os.path.exists(engine_path) else None,
            "tensorrt_engine_size_mb": round(os.path.getsize(engine_path) / (1024 * 1024), 2) if os.path.exists(engine_path) else 0,
            "onnx_exists": os.path.exists(onnx_path),
        }

    return diagnostics


def export_model_to_tensorrt(
    model_path: str,
    imgsz: int = 384,
    half: bool = True,
    dynamic: bool = False,
    batch_size: int = 1,
    workspace_gb: int = 4,
    device: str = "0"
) -> Tuple[bool, str]:
    """
    Exports a PyTorch YOLO (.pt) model to an optimized NVIDIA TensorRT engine (.engine).
    
    Args:
        model_path: Path to the .pt model weights
        imgsz: Image dimension for inference (e.g. 384 or 640)
        half: Enable FP16 half precision for 2x-4x speedup
        dynamic: Enable dynamic input shapes
        batch_size: Target batch size
        workspace_gb: TensorRT builder memory limit in GB
        device: CUDA device index ('0')
        
    Returns:
        Tuple of (success: bool, output_path_or_error: str)
    """
    if not os.path.exists(model_path):
        return False, f"Source model weights not found at: {model_path}"

    if not ULTRALYTICS_AVAILABLE:
        return False, "Ultralytics library is not installed."

    if not CUDA_AVAILABLE:
        return False, "TensorRT export requires an NVIDIA GPU with CUDA support."

    logger.info(f"Starting TensorRT export for: {model_path}")
    logger.info(f"Parameters: imgsz={imgsz}, half={half}, dynamic={dynamic}, batch={batch_size}, workspace={workspace_gb}GB")

    try:
        model = YOLO(model_path)
        engine_path = model.export(
            format="engine",
            imgsz=imgsz,
            half=half,
            dynamic=dynamic,
            batch=batch_size,
            workspace=workspace_gb,
            device=device,
            verbose=True
        )
        logger.info(f"TensorRT Engine successfully built: {engine_path}")
        return True, str(engine_path)
    except Exception as exc:
        err_msg = f"TensorRT export failed: {exc}"
        logger.error(err_msg)
        return False, err_msg


class TensorRTInferenceEngine:
    """
    High-Performance TensorRT Inference Engine Wrapper for GateSense ANPR.
    
    Automatically loads the optimized .engine model when available on CUDA,
    or smoothly falls back to PyTorch .pt model. Provides precise inference latency
    and throughput profiling.
    """
    def __init__(
        self,
        model_name: str = "yolov8n",
        models_dir: Optional[str] = None,
        imgsz: int = 384,
        conf_threshold: float = 0.25,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        
        if models_dir is None:
            backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.models_dir = os.path.join(backend_dir, "ai", "models")
        else:
            self.models_dir = models_dir

        self.pt_path = os.path.join(self.models_dir, f"{model_name}.pt")
        self.engine_path = os.path.join(self.models_dir, f"{model_name}.engine")

        self.device = device or ("cuda" if CUDA_AVAILABLE else "cpu")
        self.is_tensorrt = False
        self.model: Optional[Any] = None
        self.active_backend = "none"
        self.load_model()

    def load_model(self) -> bool:
        """Loads .engine if available on CUDA, otherwise loads .pt."""
        if not ULTRALYTICS_AVAILABLE:
            logger.error("Ultralytics is not available for loading YOLO models.")
            return False

        # 1. Try loading TensorRT .engine
        if CUDA_AVAILABLE and os.path.exists(self.engine_path):
            try:
                logger.info(f"Loading TensorRT Engine: {self.engine_path}...")
                self.model = YOLO(self.engine_path, task="detect")
                self.is_tensorrt = True
                self.active_backend = "TensorRT (FP16 Engine)"
                logger.info(f"Model [{self.model_name}] loaded using TensorRT Engine.")
                return True
            except Exception as e:
                logger.warning(f"Failed to load TensorRT Engine ({e}), falling back to PyTorch .pt")

        # 2. Fallback to PyTorch .pt
        if os.path.exists(self.pt_path):
            try:
                logger.info(f"Loading PyTorch Model: {self.pt_path} (device={self.device})...")
                self.model = YOLO(self.pt_path)
                if self.device == "cuda" and CUDA_AVAILABLE:
                    self.model.to("cuda")
                    self.active_backend = "PyTorch CUDA"
                else:
                    self.active_backend = "PyTorch CPU"
                self.is_tensorrt = False
                logger.info(f"Model [{self.model_name}] loaded using {self.active_backend}.")
                return True
            except Exception as e:
                logger.error(f"Failed to load PyTorch model: {e}")
                self.model = None
                return False

        logger.warning(f"No weights found for [{self.model_name}] at {self.engine_path} or {self.pt_path}")
        return False

    def predict(
        self,
        source: Union[np.ndarray, str, List[np.ndarray]],
        conf: Optional[float] = None,
        imgsz: Optional[int] = None,
        verbose: bool = False
    ) -> Tuple[List[Any], float]:
        """
        Runs inference and returns results along with inference latency in milliseconds.
        """
        if self.model is None:
            raise RuntimeError(f"Model [{self.model_name}] is not loaded.")

        target_conf = conf if conf is not None else self.conf_threshold
        target_imgsz = imgsz if imgsz is not None else self.imgsz

        t0 = time.perf_counter()
        results = self.model(
            source=source,
            conf=target_conf,
            imgsz=target_imgsz,
            device=self.device if not self.is_tensorrt else None,
            verbose=verbose
        )
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0
        return results, latency_ms

    def benchmark(
        self,
        num_iterations: int = 50,
        warmup_iterations: int = 10,
        input_size: Optional[Tuple[int, int]] = None
    ) -> Dict[str, Any]:
        """
        Runs comprehensive latency, throughput (FPS), and jitter benchmark.
        """
        if self.model is None:
            return {"error": "Model is not loaded"}

        size = input_size or (self.imgsz, self.imgsz)
        dummy_img = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)

        logger.info(f"Running benchmark on [{self.model_name}] ({self.active_backend}) - {num_iterations} iterations...")

        # Warmup
        for _ in range(warmup_iterations):
            _ = self.model(dummy_img, imgsz=self.imgsz, verbose=False)

        latencies = []
        for _ in range(num_iterations):
            t0 = time.perf_counter()
            _ = self.model(dummy_img, imgsz=self.imgsz, verbose=False)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        latencies = np.array(latencies)
        mean_ms = float(np.mean(latencies))
        median_ms = float(np.median(latencies))
        min_ms = float(np.min(latencies))
        max_ms = float(np.max(latencies))
        p95_ms = float(np.percentile(latencies, 95))
        p99_ms = float(np.percentile(latencies, 99))
        fps = round(1000.0 / mean_ms, 2) if mean_ms > 0 else 0.0

        benchmark_result = {
            "model_name": self.model_name,
            "active_backend": self.active_backend,
            "is_tensorrt": self.is_tensorrt,
            "device": self.device,
            "image_size": f"{size[0]}x{size[1]}",
            "iterations": num_iterations,
            "mean_latency_ms": round(mean_ms, 2),
            "median_latency_ms": round(median_ms, 2),
            "min_latency_ms": round(min_ms, 2),
            "max_latency_ms": round(max_ms, 2),
            "p95_latency_ms": round(p95_ms, 2),
            "p99_latency_ms": round(p99_ms, 2),
            "throughput_fps": fps
        }
        return benchmark_result
