Edge-AI ANPR & Vehicle Trip Management Platform

GateSense is an industrial vehicle entry/exit intelligence platform that reads number plates from camera feeds & recorded videos, fuses multi-frame OCR reads into high-confidence plate numbers, matches vehicles to scheduled trips, and provides a real-time SOC control room dashboard.

---

## 🔑 Demo Login Credentials

- **URL**: [http://localhost:5173](http://localhost:5173)
- **Operator Email**: `operator@gatesense.in` *(or any email address)*
- **Access Roles** (Selectable from dropdown):
  - `Gate supervisor` *(Default)*
  - `Security operator`
  - `Plant administrator`
  - `Audit viewer`

---

## 🛠️ Build & Run Commands

### Prerequisites
- Node.js 20+ / 24+
- `pnpm` or `npx pnpm`

### 1. Install Dependencies
```bash
npx pnpm install
```

### 2. Typecheck & Build Project
```bash
# Typecheck workspace packages & applications
npx pnpm run typecheck

# Build production bundles
npx pnpm run build
```

### 3. Run Local Dev Servers (API + Frontend)

#### Option A: Concurrent Start (Both Servers)
```bash
# Terminal 1: Run API Server (Port 5000)
$env:PORT="5000"; npx tsx backend/src/index.ts

# Terminal 2: Run GateSense React Frontend (Port 5173)
$env:PORT="5173"; $env:BASE_PATH="/"; npx vite --config frontend/vite.config.ts frontend
```

#### Option B: pnpm Workspace Commands
```bash
# Start API Server
npx pnpm --filter @workspace/backend run dev

# Start Frontend App
npx pnpm --filter @workspace/frontend run dev
```

---

## 🚀 Key Features Included

1. **Recorded Video ANPR Extraction**: Upload recorded vehicle footage (`.mp4`) to sample frames and extract license plates (`KA02MM9091`).
2. **Laptop Webcam Live ANPR**: Open laptop camera stream for live plate scanning and entry/exit verification.
3. **Multi-Frame OCR Fusion Engine**: Fuses multiple frame reads and applies character confusion maps (`0/O`, `1/I`, `8/B`).
4. **Live SOC Control Room**: Real-time traffic simulator, active trip dwell timers, alerts inbox, and manual review queue.
5. **PostgreSQL Database Models**: Drizzle ORM schemas in `@workspace/db`.

---


