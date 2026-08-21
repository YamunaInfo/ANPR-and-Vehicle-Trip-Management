# ==============================================================================
# Stage 1: Build the React 18 Frontend
# ==============================================================================
FROM node:20-slim AS frontend-builder
WORKDIR /app

# Install pnpm
RUN npm install -g pnpm

# Copy workspace package manifests
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml tsconfig.base.json tsconfig.json ./
COPY lib ./lib
COPY frontend ./frontend
COPY ai ./ai
COPY backend/package.json ./backend/package.json
COPY scripts/package.json ./scripts/package.json

# Install dependencies and build frontend
RUN pnpm install --frozen-lockfile
RUN pnpm --filter @workspace/frontend run build

# ==============================================================================
# Stage 2: Production Python Backend with AI & Static Frontend Serving
# ==============================================================================
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV and FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy backend files
COPY backend/ai ./backend/ai
COPY backend/db ./backend/db
COPY backend/routes_gatesense.py ./backend/
COPY backend/ocr_service.py ./backend/
COPY backend/.env.example ./backend/.env

# Install Python packages
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    opencv-python-headless \
    numpy \
    torch --extra-index-url https://download.pytorch.org/whl/cpu \
    torchvision --extra-index-url https://download.pytorch.org/whl/cpu \
    ultralytics \
    paddleocr \
    easyocr \
    sqlalchemy \
    pymysql \
    python-multipart

# Copy built frontend assets from Stage 1 into the frontend dist location
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

# Expose server port (Railway supplies $PORT dynamically)
ENV PORT=5001
ENV HOST=0.0.0.0
EXPOSE 5001

WORKDIR /app/backend

# Start unified FastAPI server with dynamic PORT binding
CMD ["sh", "-c", "uvicorn ocr_service:app --host 0.0.0.0 --port ${PORT:-5001}"]
