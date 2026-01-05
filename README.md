# Interceptor: Agentic AI for Deepfake Detection & Authenticity Verification

**Advanced AI-Powered Media Authentication System**  
*Intelligent deepfake detection using ensemble neural networks and autonomous agent routing*

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Model Performance](#model-performance)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Model Training](#model-training)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Interceptor is an advanced agentic AI system designed for autonomous deepfake detection and media authenticity verification. The system addresses the critical challenge of detecting manipulated media, verifying authenticity, and strengthening digital trust across platforms and operational environments.

### Key Innovations

- **Agentic Architecture**: Autonomous agent that intelligently routes video analysis through multiple specialist models
- **Domain Specialists**: Six specialized neural networks trained for specific manipulation types
- **Bias Correction**: Sophisticated mechanism for balanced detection across different scenarios
- **Edge-Ready**: Optimized for deployment on resource-constrained devices

![System Architecture](Images/Diagrams/entire-workflow.png)

### Performance Metrics

| Metric | Value | Target |
|--------|:-----:|:------:|
| **Overall Detection Confidence** | 94.9% | >90% |
| **Average Processing Time** | 2.1s | <5s |
| **Model Ensemble Size** | 6 models | — |
| **Total Parameters** | 47.2M | <100M |
| **Inference Memory** | 512 MB | <1GB |

---

## System Architecture

Interceptor employs a four-layer architecture designed for scalability, efficiency, and accuracy:

![Tech Stack](Images/Diagrams/tech_stack.png)

### Layer 1: Input Processing
- **Video Decoder**: Handles multiple video formats (MP4, AVI, MOV, MKV, WebM)
- **Frame Sampler**: Intelligent frame extraction based on video characteristics
- **Face Detector**: MTCNN-based face detection and alignment
- **Audio Extractor**: Audio waveform extraction for audio-visual analysis

### Layer 2: Model Bank
Six specialized neural networks, each optimized for specific deepfake detection scenarios:

#### BG-Model (Baseline Generalist)
![BG Model](Images/Diagrams/BG-model.png)
- **Architecture**: MobileNetV3-Small backbone
- **Parameters**: 2.1M
- **Size**: 8.2 MB
- **Accuracy**: 86.25%
- **Purpose**: Fast baseline detection for all videos

#### AV-Model (Audio-Visual Specialist)
![AV Model](Images/Diagrams/AV-model.png)
- **Architecture**: ResNet18 + Audio CNN + Lip-Sync Detector
- **Parameters**: 15.8M
- **Size**: 60.4 MB
- **Accuracy**: 93.0%
- **Purpose**: Audio-visual correlation analysis

#### Specialist Models (CM, RR, LL)
![Specialist Models](Images/Diagrams/LL-RR-CM-model.png)
- **CM-Model**: Compression artifact detection (80.83% accuracy)
- **RR-Model**: Re-recording pattern detection (85.0% accuracy)
- **LL-Model**: Low-light condition analysis (93.42% accuracy)

#### TM-Model (Temporal Specialist)
![TM Model](Images/Diagrams/TM-model.png)
- **Architecture**: ResNet18 + LSTM
- **Parameters**: 14.2M
- **Size**: 54.3 MB
- **Accuracy**: 78.5%
- **Purpose**: Temporal consistency analysis

### Layer 3: Agentic Intelligence
- **LangGraph Agent**: Autonomous decision-making system
- **Routing Engine**: Intelligent model selection based on video characteristics
- **Confidence Evaluator**: Multi-model consensus analysis
- **Explanation Generator**: Human-interpretable justifications

### Layer 4: System Interface
- **React Frontend**: Modern, responsive user interface
- **FastAPI Backend**: High-performance API server
- **PostgreSQL Database**: Secure data storage
- **Monitoring**: Real-time system monitoring with Grafana

---

## Key Features

### Intelligent Routing
The system uses an autonomous agent to decide which models to invoke based on video characteristics:

- **High Confidence (≥85%)**: Accept baseline prediction (83% computation saved)
- **Medium Confidence (65-85%)**: Route to relevant specialists (50% computation saved)
- **Low Confidence (<65%)**: Use all specialists for maximum accuracy

### Bias Correction
Advanced bias correction mechanisms ensure balanced detection:
- Focal Loss for handling class imbalance
- Weighted sampling for balanced training
- Dynamic threshold adjustment
- Cross-validation with multiple datasets

### Edge Deployment
Optimized for field operations:
- CPU-only inference capability
- INT8 quantization support
- Sub-3 second processing time
- Offline operation without cloud dependency

---

## Model Performance

### Individual Model Accuracy

| Model | Accuracy | Parameters | Size | Specialization |
|-------|:--------:|:----------:|:----:|----------------|
| **BG-Model** | 86.25% | 2.1M | 8.2 MB | Baseline Generalist |
| **AV-Model** | 93.0% | 15.8M | 60.4 MB | Audio-Visual Analysis |
| **CM-Model** | 80.83% | 11.7M | 44.6 MB | Compression Detection |
| **RR-Model** | 85.0% | 11.7M | 44.6 MB | Re-recording Detection |
| **LL-Model** | 93.42% | 11.7M | 44.6 MB | Low-light Analysis |
| **TM-Model** | 78.5% | 14.2M | 54.3 MB | Temporal Consistency |

### Ensemble Performance
- **Overall Accuracy**: 94.9%
- **Precision**: 94.2%
- **Recall**: 95.6%
- **F1-Score**: 94.9%
- **AUC-ROC**: 0.987

---

## Technology Stack

### Frontend
- **Framework**: React 18.3.1 with TypeScript
- **UI Library**: Radix UI + Material-UI
- **Styling**: Tailwind CSS
- **Charts**: ApexCharts, Recharts, D3.js
- **3D Graphics**: Three.js
- **State Management**: React Context API
- **Routing**: React Router DOM

### Backend
- **Framework**: FastAPI (Python)
- **Database**: PostgreSQL with Supabase
- **ML Framework**: PyTorch 2.1.0
- **Computer Vision**: OpenCV 4.8.1
- **Agent Framework**: LangGraph
- **Authentication**: Supabase Auth

### DevOps & Deployment
- **Containerization**: Docker
- **Orchestration**: Docker Compose
- **Cloud Platform**: Vercel (Frontend), Render (Backend)
- **Monitoring**: Grafana
- **CI/CD**: GitHub Actions

---

## Installation

### Prerequisites
- Node.js 20.x
- Python 3.9+
- Docker (optional)
- CUDA-compatible GPU (optional, for training)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/interceptor.git
   cd interceptor
   ```

2. **Install dependencies**
   ```bash
   # Frontend
   npm install
   
   # Backend
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Run the application**
   ```bash
   # Development mode
   npm run dev
   
   # Or using Docker
   docker-compose up
   ```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or use the provided scripts
./build-and-run.sh  # Linux/Mac
build-and-run.bat   # Windows
```

---

## Usage

### Web Interface

![Dashboard](Images/Website/dashboard.png)

The web interface provides an intuitive way to:
- Upload videos for analysis
- View real-time detection results
- Access detailed explanations
- Monitor system performance

![Home Page](Images/Website/home-page.png)

### Analysis Features

![Analysis Animation](Images/Website/analysis-and-model-animation.png)

- **Real-time Processing**: Upload and analyze videos instantly
- **Confidence Scoring**: Detailed confidence levels for each prediction
- **Visual Explanations**: Grad-CAM heatmaps showing decision regions
- **Batch Processing**: Analyze multiple videos simultaneously

### Analytics Dashboard

![Stats 1](Images/Website/stats_1.png)
![Stats 2](Images/Website/stats_2.png)
![Stats 3](Images/Website/stats_3.png)

Comprehensive analytics including:
- Detection accuracy trends
- Model performance metrics
- Processing time statistics
- System resource utilization

### API Usage

```python
import requests

# Upload video for analysis
with open('video.mp4', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/analyze',
        files={'video': f}
    )

result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2f}")
```

---

## API Documentation

### Endpoints

#### POST /api/analyze
Analyze a video file for deepfake detection.

**Request:**
```json
{
  "video": "multipart/form-data"
}
```

**Response:**
```json
{
  "prediction": "REAL|FAKE",
  "confidence": 0.95,
  "processing_time": 2.1,
  "models_used": ["BG", "LL", "TM"],
  "explanation": "High confidence detection based on temporal consistency analysis",
  "metadata": {
    "duration": 10.5,
    "fps": 30,
    "resolution": [1920, 1080]
  }
}
```

#### GET /api/models
Get information about available models.

**Response:**
```json
{
  "models": [
    {
      "name": "BG-Model",
      "accuracy": 86.25,
      "parameters": 2100000,
      "size_mb": 8.2
    }
  ]
}
```

#### GET /api/health
Check system health status.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": 6,
  "gpu_available": true,
  "memory_usage": "45%"
}
```

---

## Model Training

### Dataset Preparation

The system supports training on various deepfake datasets:
- DFDC (Deepfake Detection Challenge)
- FaceForensics++
- CelebDF
- DeeperForensics

### Training Scripts

#### Basic Training
```bash
python train_model.py --model BG --dataset dfdc --epochs 50
```

#### Specialist Training
```bash
# Low-light specialist
python train_specialist.py --type ll --data low_light_videos/

# Compression specialist
python train_specialist.py --type cm --data compressed_videos/
```

#### Bias Correction Training
```bash
python complete_bias_corrected_training.py --checkpoint model.pt
```

### Training Configuration

```python
# Training hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 50
WEIGHT_DECAY = 1e-5

# Bias correction settings
FOCAL_LOSS_GAMMA = 2.0
CLASS_WEIGHTS = [1.0, 2.0]  # [real, fake]
```

---

## Deployment

### Production Deployment

1. **Environment Setup**
   ```bash
   # Set production environment
   export NODE_ENV=production
   export PYTHON_ENV=production
   ```

2. **Build Application**
   ```bash
   npm run build
   ```

3. **Deploy with Docker**
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

### Cloud Deployment

#### Vercel (Frontend)
```bash
vercel --prod
```

#### Render (Backend)
```bash
# Deploy using render.yaml configuration
git push origin main
```

### Edge Deployment

For field operations, the system can be deployed on edge devices:

```bash
# Optimize models for edge deployment
python optimize_for_edge.py --quantize int8 --prune 0.3

# Deploy on Raspberry Pi or similar
./deploy_edge.sh
```

---

## Security Considerations

- **Model Versioning**: Secure model updates with cryptographic signatures
- **Input Validation**: Comprehensive video file validation
- **Rate Limiting**: API rate limiting to prevent abuse
- **Data Privacy**: No video data stored permanently
- **Secure Communication**: HTTPS/TLS encryption for all communications

---

## Performance Optimization

### Model Optimization
- **Quantization**: INT8 quantization for 4x speed improvement
- **Pruning**: Remove redundant parameters (up to 30% reduction)
- **Knowledge Distillation**: Transfer learning from larger models

### System Optimization
- **Caching**: Intelligent caching of model predictions
- **Batch Processing**: Efficient batch inference
- **GPU Acceleration**: CUDA optimization for supported hardware

---

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   export BATCH_SIZE=8
   ```

2. **Model Loading Errors**
   ```bash
   # Download models
   python download_model.py --all
   ```

3. **Video Format Issues**
   ```bash
   # Convert video format
   ffmpeg -i input.avi -c:v libx264 output.mp4
   ```

### Performance Issues

- **Slow Inference**: Enable GPU acceleration or reduce model complexity
- **High Memory Usage**: Use model quantization or reduce batch size
- **Network Latency**: Deploy models closer to users (edge deployment)

---

## Contributing

We welcome contributions to Interceptor! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/new-feature
   ```
3. **Make your changes**
4. **Add tests**
5. **Submit a pull request**

### Development Setup

```bash
# Install development dependencies
npm install --dev
pip install -r requirements-dev.txt

# Run tests
npm test
python -m pytest

# Code formatting
npm run format
black . && isort .
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **PyTorch** and **React** communities for excellent frameworks
- **Open source contributors** who made this project possible
- **Research community** for advancing deepfake detection techniques

---

## Contact

For questions, support, or collaboration opportunities:

- **Project Repository**: [GitHub](https://github.com/your-username/interceptor)
- **Documentation**: [Technical Docs](INTERCEPTOR_TECHNICAL_DOCUMENTATION.md)
- **Issues**: [GitHub Issues](https://github.com/your-username/interceptor/issues)

---

**Built with precision for digital trust and security**