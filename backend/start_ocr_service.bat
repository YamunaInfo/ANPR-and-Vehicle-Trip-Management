@echo off
REM Start EasyOCR / PaddleOCR microservice on port 5001

setlocal enabledelayedexpansion

echo ====================================================
echo ANPRX - Edge ANPR & Trip Management Backend Launcher
echo ====================================================
echo.

set PYTHON_CMD=
py -3.11 --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=py -3.11
    goto found_py
)

py -3.12 --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=py -3.12
    goto found_py
)

python --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=python
    goto found_py
)

echo ERROR: Python 3.11+ is not installed or not in PATH
pause
exit /b 1

:found_py
for /f "tokens=*" %%i in ('%PYTHON_CMD% --version') do set PYTHON_VERSION=%%i
echo Using Python: %PYTHON_VERSION%

echo.
echo ====================================================
echo Starting ANPRX Backend on http://0.0.0.0:5001
echo Health Check: http://localhost:5001/api/healthz
echo Operations API: http://localhost:5001/api/...
echo Swagger Docs:  http://localhost:5001/docs
echo ====================================================
echo.

cd /d "%~dp0"
%PYTHON_CMD% -m uvicorn ocr_service:app --host 0.0.0.0 --port 5001 --log-level info

pause
