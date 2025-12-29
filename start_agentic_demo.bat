@echo off
echo ========================================
echo E-RAKSHA AGENTIC SYSTEM - BIAS CORRECTED
echo ========================================
echo Starting bias-corrected deepfake detection system...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python and try again
    pause
    exit /b 1
)

REM Install required packages if needed
echo Installing/updating required packages...
pip install fastapi uvicorn python-multipart >nul 2>&1

REM Start the demo
echo.
echo Starting servers...
python run_agentic_demo.py

pause