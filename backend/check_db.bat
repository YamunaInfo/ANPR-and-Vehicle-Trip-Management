@echo off
REM Run ANPRX MySQL Database Storage Check

setlocal enabledelayedexpansion

cd /d "%~dp0"

set PYTHON_CMD=
py -3.11 --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=py -3.11
    goto run_check
)

py -3.12 --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=py -3.12
    goto run_check
)

python --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=python
    goto run_check
)

echo ERROR: Python 3.11+ is not found.
pause
exit /b 1

:run_check
%PYTHON_CMD% check_db_storage.py

pause
