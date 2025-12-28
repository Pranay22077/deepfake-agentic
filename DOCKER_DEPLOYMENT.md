# 🐳 E-Raksha Docker Deployment

## One-Command Setup for Anyone

This Docker setup allows anyone to download and run E-Raksha locally with a single command.

## 🚀 For End Users

### Quick Start (Recommended)
```bash
# Download and run E-Raksha
git clone <your-repo-url>
cd eraksha
docker-compose up --build

# Open in browser: http://localhost:3001
```

### Alternative: Direct Docker Run
```bash
# If you have the Docker image
docker run -p 8000:8000 -p 3001:3001 eraksha:latest
```

## 📦 What's Included

- ✅ **Complete Web Application**: Frontend + Backend
- ✅ **Trained Model**: ResNet18 deepfake detection model
- ✅ **All Dependencies**: Python packages, system libraries
- ✅ **Auto-startup**: Both servers start automatically
- ✅ **Health Checks**: Automatic service monitoring

## 🌐 Access Points

After running `docker-compose up --build`:

- **Main App**: http://localhost:3001
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📊 System Requirements

- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB for Docker image
- **CPU**: Any modern processor (GPU not required)
- **OS**: Windows, macOS, Linux with Docker

## 🔧 For Developers

### Build Process
```bash
# Build the image
docker build -t eraksha:latest .

# Run with development volumes
docker run -p 8000:8000 -p 3001:3001 \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/logs:/app/logs \
  eraksha:latest
```

### Development Mode
```bash
# Use docker-compose for development
docker-compose -f docker-compose.dev.yml up --build
```

## 📋 Features Included

### Frontend (Port 3001)
- Drag & drop video upload
- Real-time processing status
- Detailed results with confidence scores
- Interactive heatmaps
- User feedback system
- Statistics dashboard

### Backend (Port 8000)
- FastAPI REST API
- Model inference engine
- Supabase database integration
- Comprehensive logging
- Health monitoring
- Swagger documentation

## 🛠 Troubleshooting

### Common Issues

**Port already in use:**
```bash
# Change ports in docker-compose.yml
ports:
  - "8001:8000"  # Backend
  - "3002:3001"  # Frontend
```

**Model not found:**
- Ensure `fixed_deepfake_model.pt` exists in project root
- Rebuild: `docker-compose build --no-cache`

**Slow startup:**
- First run takes 2-3 minutes (downloading dependencies)
- Subsequent runs are faster (cached layers)

**Memory issues:**
- Increase Docker memory limit to 4GB+
- Close other applications

### Logs
```bash
# View container logs
docker-compose logs -f

# View specific service logs
docker-compose logs backend
docker-compose logs frontend
```

## 🚀 Production Deployment

### Docker Hub
```bash
# Build and push to Docker Hub
docker build -t yourusername/eraksha:latest .
docker push yourusername/eraksha:latest

# Users can then run:
docker run -p 8000:8000 -p 3001:3001 yourusername/eraksha:latest
```

### Cloud Deployment
- **AWS ECS**: Use the provided Dockerfile
- **Google Cloud Run**: Supports container deployment
- **Azure Container Instances**: Direct Docker image deployment
- **Railway/Render**: Git-based deployment with Dockerfile

## 📈 Performance

### Expected Performance
- **Startup Time**: 30-60 seconds
- **Video Processing**: 2-5 seconds per video
- **Memory Usage**: 1-2GB RAM
- **Model Accuracy**: 45% (realistic for domain shift)

### Optimization
- Model is CPU-optimized (no GPU required)
- Efficient face detection and preprocessing
- Minimal memory footprint
- Fast inference pipeline

## 🎯 Success Criteria

After running `docker-compose up --build`, you should see:

```
✅ Backend API is running
✅ Frontend Server is running
🎉 E-Raksha is now running!
📱 Frontend: http://localhost:3001
🔧 Backend API: http://localhost:8000
```

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. View container logs: `docker-compose logs`
3. Verify system requirements
4. Try rebuilding: `docker-compose build --no-cache`

---

**Ready for deployment!** 🚀

Anyone can now download your repository and run E-Raksha locally with just:
```bash
git clone <repo-url> && cd eraksha && docker-compose up --build
```