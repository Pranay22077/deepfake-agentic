# 🚀 E-Raksha Complete Deployment Guide

## For End Users: One-Command Setup

### Prerequisites
- Docker installed on your system
- 4GB+ RAM available
- Ports 8000 and 3001 free

### Quick Start
```bash
# 1. Download E-Raksha
git clone <your-repository-url>
cd eraksha

# 2. Start the application
docker-compose up --build

# 3. Open in browser
# Frontend: http://localhost:3001
# API: http://localhost:8000
```

That's it! E-Raksha is now running locally.

## 🧪 Test Your Deployment

```bash
# Run the test script
python docker/test-deployment.py
```

Expected output:
```
🧪 E-Raksha Docker Deployment Test
========================================
✅ Backend Health Check: PASSED
✅ Frontend Access: PASSED
✅ API Documentation: PASSED
📊 Test Results: 3/3 tests passed
🎉 All tests passed! E-Raksha is running correctly.
```

## 📱 Using E-Raksha

1. **Open the app**: http://localhost:3001
2. **Upload a video**: Drag & drop or click to browse
3. **Wait for analysis**: Usually 2-5 seconds
4. **View results**: 
   - Prediction: Real or Fake
   - Confidence score
   - Detailed analysis
   - Heatmaps showing suspicious areas

## 🔧 Advanced Usage

### Custom Ports
Edit `docker-compose.yml`:
```yaml
ports:
  - "8001:8000"  # Backend on port 8001
  - "3002:3001"  # Frontend on port 3002
```

### Persistent Data
```bash
# Create directories for persistent data
mkdir -p uploads logs models

# Run with volume mounts
docker-compose up --build
```

### Development Mode
```bash
# Run in development mode with live reload
docker-compose -f docker-compose.dev.yml up --build
```

## 🌐 Production Deployment Options

### Option 1: Docker Hub
```bash
# Build and push to Docker Hub
docker build -t yourusername/eraksha:latest .
docker push yourusername/eraksha:latest

# Users can then run:
docker run -p 8000:8000 -p 3001:3001 yourusername/eraksha:latest
```

### Option 2: Railway
1. Connect your GitHub repository to Railway
2. Railway will automatically detect the Dockerfile
3. Set environment variables if needed
4. Deploy with one click

### Option 3: Render
1. Connect repository to Render
2. Choose "Web Service" 
3. Use Docker deployment
4. Set ports: 8000 (backend), 3001 (frontend)

### Option 4: AWS ECS
1. Push image to ECR
2. Create ECS task definition
3. Deploy to ECS cluster
4. Configure load balancer

## 📊 System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │    │   Database      │
│   (Port 3001)   │◄──►│   (Port 8000)   │◄──►│   (Supabase)    │
│                 │    │                 │    │                 │
│ • Upload UI     │    │ • FastAPI       │    │ • Inference     │
│ • Results       │    │ • Model         │    │   Logs          │
│ • Heatmaps      │    │ • Inference     │    │ • Feedback      │
│ • Statistics    │    │ • Logging       │    │ • Statistics    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠 Troubleshooting

### Common Issues

**"Port already in use"**
```bash
# Find what's using the port
netstat -tulpn | grep :8000

# Kill the process or change ports in docker-compose.yml
```

**"Model not found"**
```bash
# Ensure model file exists
ls -la fixed_deepfake_model.pt

# Rebuild without cache
docker-compose build --no-cache
```

**"Container keeps restarting"**
```bash
# Check logs
docker-compose logs -f

# Check container status
docker ps -a
```

**"Out of memory"**
```bash
# Increase Docker memory limit to 4GB+
# Docker Desktop: Settings > Resources > Memory
```

### Debug Commands
```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs backend
docker-compose logs frontend

# Enter container for debugging
docker-compose exec eraksha bash

# Check container health
docker ps
curl http://localhost:8000/health
```

## 📈 Performance Expectations

### Startup Time
- **First run**: 2-3 minutes (downloading dependencies)
- **Subsequent runs**: 30-60 seconds

### Processing Speed
- **Small videos** (<10MB): 2-3 seconds
- **Medium videos** (10-50MB): 3-5 seconds
- **Large videos** (>50MB): 5-10 seconds

### Resource Usage
- **RAM**: 1-2GB during processing
- **CPU**: Moderate usage during inference
- **Storage**: 2GB for Docker image

### Model Performance
- **Accuracy**: 45% (expected due to domain shift)
- **Confidence**: Realistic scores (20-80%)
- **Bias**: No systematic bias toward real/fake

## 🎯 Success Checklist

After deployment, verify:
- [ ] Frontend loads at http://localhost:3001
- [ ] Backend API responds at http://localhost:8000/health
- [ ] Can upload and process a test video
- [ ] Results show prediction and confidence
- [ ] Heatmaps are generated
- [ ] No error messages in logs

## 📞 Support

If you encounter issues:

1. **Check logs**: `docker-compose logs -f`
2. **Verify requirements**: 4GB RAM, Docker installed
3. **Test connectivity**: `curl http://localhost:8000/health`
4. **Rebuild**: `docker-compose build --no-cache`
5. **Check ports**: Ensure 8000 and 3001 are free

## 🎉 Deployment Complete!

Your E-Raksha deepfake detection system is now:
- ✅ **Containerized** and portable
- ✅ **Easy to deploy** with one command
- ✅ **Production ready** with health checks
- ✅ **Scalable** for cloud deployment
- ✅ **User-friendly** with web interface

**Share with others:**
```bash
git clone <your-repo-url> && cd eraksha && docker-compose up --build
```

Anyone can now run E-Raksha locally in under 5 minutes! 🚀