@echo off
title LUNAR Launcher
chcp 65001 >nul
echo === LUNAR System Launcher ===
echo.

set ROOT=%~dp0
if "%ROOT:~-1%"=="\" set ROOT=%ROOT:~0,-1%

set PYTHON=%ROOT%\lunar_env\python.exe
set BACKEND_DIR=%ROOT%\pipeline_code
set FRONTEND_DIR=%ROOT%\lunar_web

REM full preprocessing ON (set to 0 to disable)
set "USE_FULL_PREPROCESS=1"

echo [CHECK] ROOT      = %ROOT%
echo [CHECK] PYTHON    = %PYTHON%
echo [CHECK] BACKEND   = %BACKEND_DIR%
echo [CHECK] FRONTEND  = %FRONTEND_DIR%
echo.

if not exist "%PYTHON%" (
    echo [ERROR] lunar_env\python.exe not found at:
    echo         %PYTHON%
    pause
    exit /b 1
)
echo [OK] python.exe found

"%PYTHON%" --version
echo.

if not exist "%BACKEND_DIR%\main.py" (
    echo [ERROR] pipeline_code\main.py not found
    pause
    exit /b 1
)
echo [OK] main.py found

where pnpm >nul 2>&1
if errorlevel 1 (
    echo [SETUP] Installing pnpm ...
    npm install -g pnpm
)

if not exist "%FRONTEND_DIR%\node_modules" (
    echo [SETUP] Installing frontend packages ...
    cd /d "%FRONTEND_DIR%"
    pnpm install
)

echo.
echo [1/3] Starting backend ...
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":8000 "') do (
    taskkill /f /pid %%a >nul 2>&1
)
timeout /t 1 /nobreak >nul
start "LUNAR-Backend" cmd /k "cd /d "%BACKEND_DIR%" && "%PYTHON%" -m uvicorn main:app --port 8000"

echo [2/3] Starting frontend ...
start "LUNAR-Frontend" cmd /k "cd /d "%FRONTEND_DIR%" && pnpm dev"

echo [3/3] Opening browser in 15 seconds ...
timeout /t 15 /nobreak >nul
start "" "http://localhost:3000"

echo.
echo Running at http://localhost:3000
echo Close LUNAR-Backend and LUNAR-Frontend windows to stop.
pause
