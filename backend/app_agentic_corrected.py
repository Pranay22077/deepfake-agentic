#!/usr/bin/env python3
"""
E-Raksha Agentic Backend - Bias Corrected Version
FastAPI backend with bias-corrected agentic system
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Add project root to path
sys.path.append('.')
sys.path.append('src')

# Import the bias-corrected agentic system
from eraksha_agent import ErakshAgent

# Initialize FastAPI app
app = FastAPI(
    title="E-Raksha Agentic API - Bias Corrected",
    description="Bias-corrected agentic deepfake detection system with multiple specialist models",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent instance
agent = None

@app.on_event("startup")
async def startup_event():
    """Initialize the agentic system on startup"""
    global agent
    print("🚀 Initializing E-Raksha Agentic System (Bias Corrected)...")
    try:
        agent = ErakshAgent()
        print("✅ Agentic system initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize agentic system: {e}")
        raise e

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "E-Raksha Agentic API - Bias Corrected",
        "version": "2.0.0",
        "status": "running",
        "features": [
            "Bias-corrected predictions",
            "Multi-model agentic routing",
            "Specialist model integration",
            "Balanced real/fake detection"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global agent
    
    if agent is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": "Agent not initialized"}
        )
    
    # Check model status
    model_status = {}
    for model_name, model in agent.models.items():
        model_status[model_name] = "loaded" if model is not None else "not_available"
    
    return {
        "status": "healthy",
        "agent_initialized": True,
        "models": model_status,
        "bias_correction": "enabled"
    }

@app.post("/predict")
async def predict_video(file: UploadFile = File(...)):
    """
    Predict if uploaded video is real or fake using bias-corrected agentic system
    """
    global agent
    
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Create temporary file
    temp_dir = tempfile.mkdtemp()
    temp_file_path = None
    
    try:
        # Save uploaded file
        file_extension = Path(file.filename).suffix if file.filename else '.mp4'
        temp_file_path = os.path.join(temp_dir, f"upload{file_extension}")
        
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Run prediction
        result = agent.predict(temp_file_path)
        
        if result['success']:
            # Format response
            response = {
                "success": True,
                "filename": file.filename,
                "prediction": result['prediction'],
                "confidence": round(result['confidence'] * 100, 1),
                "confidence_level": result['confidence_level'],
                "explanation": result['explanation'],
                "details": {
                    "best_model": result['best_model'],
                    "specialists_used": result['specialists_used'],
                    "all_predictions": {
                        model: {
                            "prediction": round(data['prediction'] * 100, 1),
                            "confidence": round(data['confidence'] * 100, 1)
                        }
                        for model, data in result['all_predictions'].items()
                    },
                    "video_characteristics": {
                        k: float(v) if isinstance(v, (int, float, bool)) else v 
                        for k, v in result['video_characteristics'].items()
                    },
                    "processing_time": round(result['processing_time'], 2)
                },
                "bias_correction": "applied"
            }
            
            return response
        else:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {result['error']}")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    
    finally:
        # Clean up temporary files
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)

@app.get("/models")
async def get_models():
    """Get information about loaded models"""
    global agent
    
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    models_info = {}
    
    for model_name, model in agent.models.items():
        if model is not None:
            models_info[model_name] = {
                "status": "loaded",
                "type": {
                    "student": "Baseline Generalist Model",
                    "av": "Audio-Visual Specialist",
                    "cm": "Compression Specialist", 
                    "rr": "Re-recording Specialist",
                    "ll": "Low-light Specialist",
                    "tm": "Temporal Specialist"
                }.get(model_name, "Unknown"),
                "bias_correction": {
                    "student": {"weight": 1.0, "bias": 0.0},
                    "av": {"weight": 1.5, "bias": 0.0},
                    "cm": {"weight": 1.0, "bias": 0.1},
                    "rr": {"weight": 0.8, "bias": 0.15},
                    "ll": {"weight": 1.4, "bias": -0.05},
                    "tm": {"weight": 1.1, "bias": 0.0}
                }.get(model_name, {"weight": 1.0, "bias": 0.0})
            }
        else:
            models_info[model_name] = {
                "status": "not_available",
                "type": "Unknown"
            }
    
    return {
        "models": models_info,
        "total_loaded": len([m for m in agent.models.values() if m is not None]),
        "bias_correction_enabled": True
    }

if __name__ == "__main__":
    print("🚀 Starting E-Raksha Agentic API Server (Bias Corrected)")
    uvicorn.run(
        "backend.app_agentic_corrected:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )