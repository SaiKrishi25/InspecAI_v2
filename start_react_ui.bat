@echo off
echo ========================================
echo  InspecAI - Startup Script
echo ========================================
echo.
echo Starting InspecAI with AI Assistant...
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
start "InspecAI Backend" cmd /k "python -m flask --app app.main run --host 0.0.0.0 --port 8000"

echo Waiting for backend to start...
timeout /t 5 /nobreak > nul

echo Starting React Frontend with Gemini AI...
cd frontend
start "InspecAI Frontend" cmd /k "npm run dev"

echo.
echo ========================================
echo  InspecAI is ready!
echo ========================================
echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:5173
echo.
echo AI Assistant: Powered by Gemini API (Direct)
echo.
echo Open http://localhost:5173 in your browser
echo Click the AI icon to use the Digital Assistant
echo.
echo Press any key to exit this window...
pause > nul

