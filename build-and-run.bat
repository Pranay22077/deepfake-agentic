@echo off
REM E-Raksha Quick Build and Run Script for Windows

echo 🚀 E-Raksha Quick Deployment
echo ============================

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not installed. Please install Docker Desktop first.
    echo    Visit: https://docs.docker.com/desktop/windows/
    pause
    exit /b 1
)

REM Check if Docker Compose is available
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker Compose is not installed. Please install Docker Compose first.
    pause
    exit /b 1
)

REM Check if model file exists
if not exist "fixed_deepfake_model.pt" (
    echo 🔧 Running setup to download models...
    python setup.py
    if %errorlevel% neq 0 (
        echo ❌ Setup failed. Please check the error messages above.
        pause
        exit /b 1
    )
) else (
    echo ✅ Model file found
)

echo.
echo 🔧 Building and starting E-Raksha...
echo This may take 2-3 minutes on first run...
echo.

REM Build and start the application
docker-compose up --build

echo.
echo 🛑 E-Raksha has been stopped.
echo To restart: docker-compose up
echo To rebuild: docker-compose up --build
pause