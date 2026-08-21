# 🛡️ GateSense / ANPRX — Industrial Edge-AI ANPR & Trip Management Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.9-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF)](https://github.com/ultralytics/ultralytics)
[![NVIDIA TensorRT](https://img.shields.io/badge/NVIDIA-TensorRT_Accelerated-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/tensorrt)
[![MySQL / PostgreSQL](https://img.shields.io/badge/Database-MySQL_8.x_%7C_PostgreSQL-4479A1?logo=mysql&logoColor=white)](https://www.mysql.com/)

**GateSense (ANPRX)** is an enterprise-grade Edge-AI Automatic Number Plate Recognition (ANPR) and Vehicle Trip Management platform. Engineered for industrial manufacturing plants, logistics hubs, toll plazas, and high-security facilities, it delivers sub-second automated gate access decisions, multi-camera surveillance integration, multi-frame OCR fusion, and an interactive Security Operations Center (SOC) control room.

---

## 📑 Table of Contents

1. [Key Features](#-key-features)
2. [System Architecture](#-system-architecture)
3. [AI & Computer Vision Pipeline](#-ai--computer-vision-pipeline)
4. [Demo Credentials & User Roles](#-demo-credentials--user-roles)
5. [Tech Stack](#-tech-stack)
6. [Repository Structure](#-repository-structure)
7. [Prerequisites & System Requirements](#-prerequisites--system-requirements)
8. [Installation & Setup](#-installation--setup)
9. [Running the Platform](#-running-the-platform)
10. [Hardware Acceleration (NVIDIA TensorRT)](#-hardware-acceleration-nvidia-tensorrt)
11. [Database Schema & Architecture](#-database-schema--architecture)
12. [Core REST API Reference](#-core-rest-api-reference)
13. [Testing & Verification](#-testing--verification)
14. [Environment Configuration (.env)](#-environment-configuration-env)
15. [License](#-license)

---

## 🚀 Key Features

### 👁️ 1. Edge-AI Computer Vision & ANPR
- **Multi-Source Video Processing**: Ingest live RTSP/HTTP CCTV streams, laptop webcams, or pre-recorded `.mp4` video files.
- **Two-Stage Detection**: YOLOv8 vehicle classification (Truck, Car, Bus, Two-Wheeler) paired with high-precision license plate localization.
- **PaddleOCR PP-OCRv4 + EasyOCR Fallback**: Dual-engine text recognition with adaptive deskewing, grayscale contrast equalization, and character isolation.
- **Multi-Frame OCR Fusion**: Character-level temporal voting across consecutive video frames with optical confusion correction matrix (`0/O`, `1/I`, `8/B`, `5/S`, `2/Z`).
- **Indian Plate Standard Validation**: Built-in syntax validators conforming to standard state formats (e.g., `KA 02 MM 9091`) and the Bharat (`BH`) series.

### 🏭 2. Industrial Trip & Yard Management
- **Automated Gate Decisions**: Real-time `ALLOW`, `DENY`, or `MANUAL_REVIEW` triggering based on whitelist rules, active trip schedules, and driver authorization.
- **Trip Lifecycle State Machine**: Full lifecycle tracking: `Scheduled` ➔ `Arrived` ➔ `Entry Approved` ➔ `Inside Plant` ➔ `At Destination` ➔ `Exit Detected` ➔ `Completed`.
- **Dwell Time Violation Tracking**: Active timers tracking total turnaround time inside the yard with automated overstay alerts.
- **Transporter & Fleet Master**: Centralized catalog of authorized transporters, drivers, and registered vehicles.

### 🖥️ 3. SOC Control Room & Human-in-the-Loop (HITL)
- **Live Gate Monitoring**: High-FPS visual feed with dynamic bounding box overlays, confidence metrics, and instant audio alerts.
- **Manual Review Queue**: Low-confidence detections and OCR discrepancies are routed to an operator queue with side-by-side plate crops for one-click correction.
- **Security Watchlist & Blacklist**: Instant siren and high-priority alerts when blacklisted or unregistered plates hit the gate.
- **Real-Time Traffic Simulator**: Integrated testing tool to simulate multi-gate entry/exit traffic bursts without physical hardware.

### 📊 4. Reporting & Audit Governance
- **Comprehensive Audit Trail**: Every entry/exit event records high-resolution vehicle crops, plate crops, timestamps, confidence scores, and operator overrides.
- **Analytics & Heatmaps**: Gate throughput trends, peak traffic hours, transporter turnaround averages, and model accuracy breakdown.

---

## 🏛️ System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        EDGE CCTV CAMERAS / VIDEO                       │
│              (RTSP Streams / Webcams / Recorded MP4 Files)              │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                       AI INFERENCE PIPELINE                            │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────┐ │
│  │ YOLOv8 Vehicle Det   │─►│ YOLOv8 Plate Det     │─►│ PaddleOCR /  │ │
│  │ (TensorRT FP16/INT8) │  │ (TensorRT FP16/INT8) │  │ EasyOCR      │ │
│  └──────────────────────┘  └──────────────────────┘  └──────┬───────┘ │
│                                                             │         │
│  ┌──────────────────────────────────────────────────────────▼───────┐ │
│  │ Multi-Frame OCR Fusion & Indian Plate Syntax RegEx Validation    │ │
│  └──────────────────────────────────────────────────────────┬───────┘ │
└─────────────────────────────────────────────────────────────┼──────────┘
                                                              │
                                                              ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      BACKEND SERVICES & ENGINE                         │
│  ┌──────────────────────────────┐    ┌───────────────────────────────┐ │
│  │ FastAPI Python Microservice  │    │ Node.js / Express API Service │ │
│  │ (Video Stream, OCR, TensorRT)│    │ (Trip Logic, Auth, RBAC)      │ │
│  └──────────────┬───────────────┘    └───────────────┬───────────────┘ │
│                 └───────────────────┬────────────────┘                 │
│                                     ▼                                  │
│             ┌───────────────────────────────────────────────┐          │
│             │ Database Layer (MySQL 8.x / PostgreSQL)       │          │
│             │ Drizzle ORM & SQLAlchemy 24+ Production Tables│          │
│             └───────────────────────────────────────────────┘          │
└─────────────────────────────────────┬──────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      GATESENSE REACT 18 SPA FRONTEND                   │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌─────────────┐ │
│  │ Live Gate SOC │ │ Vehicle/Driver│ │ Trip Schedule │ │ Review Queue│ │
│  │ Monitoring    │ │ Master Record │ │ & Overstay    │ │ & Audit Log │ │
│  └───────────────┘ └───────────────┘ └───────────────┘ └─────────────┘ │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 AI & Computer Vision Pipeline

```
[Raw Frame @ 1080p]
       │
       ▼
[Motion Filter / Frame Sampler] (Skip empty frames)
       │
       ▼
[Vehicle Detection (YOLOv8n)] ──► Bounding Box & Class (Truck/Car/Bus/Bike)
       │
       ▼
[Plate Localization (YOLOv8-Plate)] ──► Crop License Plate Region
       │
       ▼
[Image Pre-Processing] ──► Bilateral Filter ➔ Grayscale ➔ Otsu Binarization ➔ Deskew
       │
       ▼
[Text Recognition (PP-OCRv4)] ──► Extract Character Candidates + Confidence Scores
       │
       ▼
[Multi-Frame OCR Fusion] ──► Character-by-Character Majority Voting & Confusion Map
       │
       ▼
[Syntax Validation] ──► RegEx: ^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$ (or BH-Series)
       │
       ▼
[Master Matcher] ──► Check Whitelist / Watchlist / Scheduled Trips
       │
       ▼
[Decision Engine] ──► ALLOW (Barrier Open) | DENY (Sound Alarm) | MANUAL REVIEW
```

---

## 🔑 Demo Credentials & User Roles

Access the web interface at **[http://localhost:5173](http://localhost:5173)**.

| Role | Default Email | Permissions |
| :--- | :--- | :--- |
| **Plant Administrator** | `admin@gatesense.in` | Full system control, user management, camera config, data export |
| **Gate Supervisor** | `supervisor@gatesense.in` | Trip overrides, manual review queue, gate barrier manual controls |
| **Security Operator** | `operator@gatesense.in` | Live gate view, incident reporting, alert acknowledgments |
| **Audit Viewer** | `audit@gatesense.in` | Read-only access to event logs, compliance reports, and analytics |

*Note: Any email address can be used to log in during local demo mode.*

---

## 💻 Tech Stack

### **Frontend**
- **Framework**: React 18, TypeScript, Vite
- **Routing & State**: Wouter, TanStack Query (React Query v5)
- **Styling & UI**: TailwindCSS, GateSense Custom Industrial Design System, Lucide Icons
- **Data Visualization**: Recharts (Throughput, Dwell Times, Peak Traffic)

### **Backend & APIs**
- **Python Backend**: FastAPI, Uvicorn, Pydantic v2
- **Node.js Backend**: Express.js, TypeScript, TSX
- **Database ORM**: SQLAlchemy 2.0 (Python) & Drizzle ORM (TypeScript)
- **Database Engines**: MySQL 8.x / PostgreSQL 15+

### **AI / Machine Learning & Computer Vision**
- **Object Detection**: YOLOv8 (Ultralytics)
- **OCR Engines**: PaddleOCR (PP-OCRv4), EasyOCR
- **Hardware Acceleration**: NVIDIA TensorRT (FP16 / INT8), CUDA 11.8 / 12.x, cuDNN
- **Image Processing**: OpenCV (cv2), NumPy

---

## 📁 Repository Structure

```text
apnrlive-master/
├── ai/                         # Edge TypeScript AI engine and detector bindings
│   └── src/
│       ├── detectors/          # YOLO vehicle & plate detection interfaces
│       ├── ocr/                # Client-side OCR abstraction & syntax validators
│       ├── tracking/           # Multi-frame tracking & temporal fusion logic
│       └── anpr-pipeline.ts    # Main pipeline orchestrator
├── backend/                    # Python FastAPI & Node.js backend microservices
│   ├── ai/                     # Python AI deep learning modules
│   │   ├── cctv_manager.py     # Live RTSP/CCTV multi-stream manager
│   │   ├── multi_frame_fusion.py # Multi-frame character voting algorithm
│   │   ├── tensorrt_engine.py  # NVIDIA TensorRT runtime & export utilities
│   │   └── video_processor.py  # Video chunk sampler & plate recognition
│   ├── db/                     # SQLAlchemy models, sessions, and seed data
│   │   ├── models.py           # 24+ Production database table definitions
│   │   ├── seed.py             # Industrial demo seed data generator
│   │   └── session.py          # MySQL / PostgreSQL connection pooling
│   ├── routes_gatesense.py     # REST API routing for GateSense operations
│   ├── ocr_service.py          # FastAPI OCR microservice entrypoint
│   └── TENSORRT_GUIDE.md       # TensorRT setup and benchmarking guide
├── frontend/                   # React 18 single page application
│   ├── src/
│   │   ├── components/         # GateSense industrial UI component library
│   │   ├── pages/              # Views (Login, Profile, Live View, etc.)
│   │   ├── lib/                # Auth context, API client, utilities
│   │   └── App.tsx             # Master router and navigation shell
│   └── vite.config.ts          # Vite build and proxy configuration
├── lib/                        # Shared workspace libraries
│   ├── api-client-react/       # Generated React Query hooks for REST API
│   ├── api-spec/               # OpenAPI 3.0 specification
│   ├── api-zod/                # Zod schemas for runtime payload validation
│   └── db/                     # Drizzle ORM schema for TypeScript
├── scripts/                    # Build, migration, and automation scripts
├── package.json                # Root pnpm monorepo workspace configuration
└── README.md                   # Project documentation
```

---

## 📦 Prerequisites & System Requirements

### Minimum Requirements (CPU Mode)
- **OS**: Windows 10/11, Ubuntu 20.04+, or macOS
- **Node.js**: `v20.x` or `v22.x`+
- **Package Manager**: `pnpm` (`npm install -g pnpm`)
- **Python**: `3.10` or `3.11`
- **Memory**: 8 GB RAM

### Recommended Requirements (GPU / TensorRT Mode)
- **GPU**: NVIDIA RTX 3060+ / Tesla T4 / Jetson Orin
- **Drivers**: NVIDIA CUDA 11.8 / 12.x & cuDNN 8.x
- **TensorRT**: `tensorrt >= 8.6`

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/gatesense-anpr.git
cd gatesense-anpr
```

### 2. Install Workspace Node.js Dependencies
```bash
pnpm install
```

### 3. Setup Python Virtual Environment
```bash
# Create and activate virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Linux/macOS:
source .venv/bin/activate

# Install Python requirements
pip install fastapi uvicorn pydantic opencv-python-headless numpy torch torchvision ultralytics paddleocr easyocr sqlalchemy pymysql python-multipart
```

### 4. Configure Environment Variables
Copy `.env.example` in `backend/`:
```bash
cp backend/.env.example backend/.env
```
*(Configure database credentials and engine flags as needed)*.

---

## 🚦 Running the Platform

### Quick Start (Development Mode)

#### Option 1: Run via Concurrent Workspace Commands
```bash
# Terminal 1: Run Python Edge-AI & Backend Microservice (Port 5001 / 5000)
cd backend
python ocr_service.py

# Terminal 2: Run GateSense React Frontend (Port 5173)
pnpm --filter @workspace/frontend run dev
```

#### Option 2: Run via Node.js TSX Backend
```bash
# Terminal 1: Run TS Backend
pnpm --filter @workspace/backend run dev

# Terminal 2: Run Frontend
pnpm --filter @workspace/frontend run dev
```

Visit **[http://localhost:5173](http://localhost:5173)** in your browser.

---

## ⚡ Hardware Acceleration (NVIDIA TensorRT)

GateSense supports TensorRT FP16/INT8 inference engines, reducing detection latencies down to **< 10ms**.

### Exporting YOLO Models to TensorRT
```bash
cd backend
python test_tensorrt.py --export
```

### Benchmark Latency & Throughput
```bash
python test_tensorrt_benchmark.py --iterations 50
```

### 📊 Performance Comparison Table

| Hardware / Acceleration | Precision | Resolution | Mean Latency | Throughput | Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Intel Core i7 (12th Gen)** | FP32 | 384×384 | 45.2 ms | ~22 FPS | 1.0x (Baseline) |
| **NVIDIA RTX 4060 (PyTorch CUDA)** | FP32 | 384×384 | 14.8 ms | ~67 FPS | 3.1x |
| **NVIDIA RTX 4060 (TensorRT)** | **FP16** | **384×384** | **4.9 ms** | **~204 FPS** | **9.2x** |
| **NVIDIA Jetson AGX Orin** | **FP16** | **384×384** | **6.2 ms** | **~161 FPS** | **7.3x** |

---

## 🗄️ Database Schema & Architecture

GateSense uses a 24-table relational architecture for industrial compliance and tracking:

| Table Name | Description |
| :--- | :--- |
| `users` | System operators, administrators, roles, and credential hashes |
| `vehicles` | Master vehicle registry, vehicle classes, authorized status, assigned transporter |
| `drivers` | Driver licenses, biometric IDs, phone numbers, and assigned vehicles |
| `cameras` | CCTV RTSP streams, gate assignments (Entry/Exit), health status |
| `scheduled_trips` | Scheduled dock appointments, dwell times, and cargo manifests |
| `gate_events` | Immutable log of every entry/exit event, plate crop URI, and decision |
| `review_queue` | HITL manual correction queue for low-confidence reads (< 75%) |
| `manual_corrections` | Audit trail of operator edits for continuous model retraining |
| `alerts` | System notifications (Overstay, Unauthorized Vehicle, Watchlist Hit) |
| `whitelist_entries` | Fast-path automated barrier opening authorization rules |
| `watchlist_entries` | Blacklist/Hotlist flags with automated siren triggering |
| `daily_gate_summaries`| Aggregated daily analytics and gate efficiency metrics |

---

## 📡 Core REST API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/login` | Authenticate operator and retrieve JWT token |
| `GET` | `/api/dashboard/summary` | Real-time counts (Vehicles inside, entries today, active alerts) |
| `POST` | `/api/ocr/predict` | Process single frame / Base64 image through ANPR pipeline |
| `POST` | `/api/video/process` | Upload recorded `.mp4` video for background plate extraction |
| `GET` | `/api/trips/active` | Get all active trips currently inside plant premises |
| `POST` | `/api/trips/schedule` | Create a new scheduled transport trip |
| `GET` | `/api/events` | Paginated query of all historical gate entry/exit logs |
| `GET` | `/api/review-queue` | Retrieve pending low-confidence manual review items |
| `POST` | `/api/review-queue/correct` | Submit manual operator correction for a plate |
| `POST` | `/api/simulate-traffic` | Trigger simulated vehicle traffic for testing |
| `GET` | `/api/cameras` | List connected CCTV cameras and ping status |
| `GET` | `/api/healthz` | System health check and database connectivity verification |

---

## 🧪 Testing & Verification

Run automated test suites from the `backend/` directory:

```bash
# Run End-to-End ANPR and Database Integration Test
python test_anprx_mysql_e2e.py

# Run Multi-Frame OCR Fusion Unit Tests
python test_multi_frame_fusion.py

# Run Acceptance and Edge Pipeline Verification
python test_final_acceptance.py

# Typecheck workspace TypeScript packages
pnpm run typecheck
```

---

## ⚙️ Environment Configuration (.env)

```ini
# Database Connection (MySQL 8.x or PostgreSQL)
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=anprx_app
MYSQL_PASSWORD=your_secure_password
MYSQL_DATABASE=anprx
DATABASE_URL=mysql+pymysql://anprx_app:your_secure_password@localhost:3306/anprx

# Server Settings
PORT=5001
HOST=0.0.0.0
ENVIRONMENT=production

# AI & OCR Engine Settings
OCR_ENGINE=PaddleOCR
USE_GPU=True
CONFIDENCE_THRESHOLD=0.75
DEDUPLICATION_WINDOW_SECONDS=15
```

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <b>GateSense / ANPRX Platform</b> — Built for Enterprise Edge AI & Smart Facility Logistics.
</p>
