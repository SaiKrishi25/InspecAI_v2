@echo off
echo ========================================
echo    Vehicle Inspection Pipeline (AI)
echo ========================================

set FLASK_APP=app/main.py
set UPLOAD_DIR=app/static/uploads
set OUTPUT_DIR=app/static/outputs
set PORT=8000

REM Defect Detection Configuration
set MODEL_TYPE=yolo
set MODEL_PATH=best.pt
set CONFIDENCE_THRESHOLD=0.5

REM SAM2 Surface Defect Detection Configuration
set SAM2_ENABLED=true
set SAM2_MODEL=facebook/sam2-hiera-tiny

echo Configuration:
echo - Model Type: %MODEL_TYPE% (YOLO or FasterRCNN)
echo - Model Path: %MODEL_PATH%
echo - Confidence Threshold: %CONFIDENCE_THRESHOLD%
echo - SAM2 Enabled: %SAM2_ENABLED%
echo - SAM2 Model: %SAM2_MODEL%
echo - Port: %PORT%
echo.
echo Supported detection types: 
echo   * Structural: Dents, Scratches (YOLO)
echo   * Surface: Paint Defects, Contamination, Corrosion, Water Spots (SAM2)
echo.
echo Starting Vehicle Inspection Server...
echo Access the application at: http://localhost:%PORT%
echo.

python -m flask run --host=0.0.0.0 --port=%PORT%
