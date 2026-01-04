@echo off
REM E-Raksha Quick Build and Run Script for Windows

echo [RUN] E-Raksha Quick Deployment
echo ============================

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not installed. Please install Docker Desktop first.
    echo    Visit: https://docs.docker.com/desktop/windows/
    pause
    exit /b 1
)

REM Check if Docker Compose is available
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker Compose is not installed. Please install Docker Compose first.
    pause
    exit /b 1
)

REM Check if model file exists
if not exist "models\baseline_student.pt" (
    echo [SETUP] Running setup to download models...
    python scripts\setup\setup.py
    if %errorlevel% neq 0 (
        echo [ERROR] Setup failed. Please check the error messages above.
        pause
        exit /b 1
    )
) else (
    echo [OK] Model file found
)

echo.
echo [SETUP] Building and starting E-Raksha...
echo This may take 2-3 minutes on first run...
echo.

REM Build and start the application
docker-compose up --build

echo.
echo 🛑 E-Raksha has been stopped.
echo To restart: docker-compose up
echo To rebuild: docker-compose up --build
pause