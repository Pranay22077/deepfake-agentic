@echo off
REM E-Raksha Windows Deployment Script

echo E-Raksha Deployment Script
echo ================================

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Docker is not installed. Please install Docker Desktop first.
    pause
    exit /b 1
)

echo Docker found

REM Stop existing container
echo Stopping existing container...
docker stop eraksha-site >nul 2>&1
docker rm eraksha-site >nul 2>&1

REM Build new image
echo Building E-Raksha image...
docker build -t eraksha-site -f docker/Dockerfile .

if %errorlevel% neq 0 (
    echo Image build failed
    pause
    exit /b 1
)

echo Image built successfully

REM Run container
echo Starting E-Raksha container...
docker run -d -p 8080:80 --name eraksha-site eraksha-site

if %errorlevel% neq 0 (
    echo Container failed to start
    pause
    exit /b 1
)

echo Container started successfully
echo Website available at: http://localhost:8080
echo Model downloads available at: http://localhost:8080/models/

REM Wait and test
echo Waiting for service to be ready...
timeout /t 5 /nobreak >nul

echo Deployment complete!
pause