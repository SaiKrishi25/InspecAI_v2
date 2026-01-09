#!/bin/bash
echo "========================================"
echo " InspecAI - React UI Startup Script"
echo "========================================"
echo ""
echo "This script will start both the Flask backend and React frontend"
echo ""

# Check if node_modules exists in frontend
if [ ! -d "frontend/node_modules" ]; then
    echo "[ERROR] Frontend dependencies not installed!"
    echo "Please run: cd frontend && npm install"
    echo ""
    exit 1
fi

echo "Starting Flask Backend..."
python -m app.main &
BACKEND_PID=$!

echo "Waiting for backend to start..."
sleep 5

echo "Starting React Frontend..."
cd frontend
npm run dev &
FRONTEND_PID=$!

echo ""
echo "========================================"
echo " InspecAI is running!"
echo "========================================"
echo ""
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo ""
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Wait for user interrupt
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait

