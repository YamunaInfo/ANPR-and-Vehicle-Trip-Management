@echo off
REM Search for a vehicle plate in the ANPRX MySQL database

setlocal enabledelayedexpansion
cd /d "%~dp0"

set PLATE=%1
if "%PLATE%"=="" (
    set /p PLATE="Enter License Plate number (e.g. HR26FC2782): "
)

py -3.11 find_vehicle.py %PLATE%

pause
