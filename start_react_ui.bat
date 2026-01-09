@echo off
echo ========================================
echo  InspecAI - React UI Startup Script
echo ========================================
echo.
echo This script will start both the Flask backend and React frontend
echo.

REM Check if node_modules exists in frontend
if not exist "frontend\node_modules" (
    echo [ERROR] Frontend dependencies not installed!
    echo Please run: cd frontend ^&^& npm install
    echo.
    pause
    exit /b 1
)

echo Starting Flask Backend...
start "InspecAI Backend" cmd /k "python -m app.main"

echo Waiting for backend to start...
timeout /t 5 /nobreak > nul

echo Starting React Frontend...
cd frontend
start "InspecAI Frontend" cmd /k "npm run dev"

echo.
echo ========================================
echo  InspecAI is starting up!
echo ========================================
echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:3000
echo.
echo The React UI will open in your browser automatically.
echo.
echo Press any key to exit this window...
pause > nul

