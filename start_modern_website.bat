@echo off
echo 🚀 E-Raksha Modern UI
echo ==================================================
echo ✅ Starting React development server...
echo 🌐 URL: http://localhost:3001
echo 📁 Project: eraksha-modern-ui
echo 🎯 Features:
echo • Modern React/TypeScript with Vite
echo • Tailwind CSS with shadcn/ui components
echo • Dark/Light theme toggle
echo • Framer Motion animations
echo • Interactive file upload with drag & drop
echo • Real-time analysis with progress tracking
echo • Detailed results with model predictions
echo • Analysis history and statistics
echo • Charts and performance metrics
echo • Responsive design with modern aesthetics
echo 🚀 Installing dependencies and starting...

cd eraksha-modern-ui
call npm install
if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

echo ✅ Dependencies installed successfully
echo 🚀 Starting development server...
call npm run dev

pause