#!/usr/bin/env python3
"""
Simple E-Raksha Backend API for testing
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import tempfile
import uuid
from datetime import datetime
import random

app = FastAPI(title="E-Raksha Simple API", version="1.0.0")

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "E-Raksha Simple API",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict")
async def predict_deepfake(file: UploadFile = File(...)):
    """Simple prediction endpoint with realistic fake results"""
    
    # Validate file type
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="Please upload a video file")
    
    # Save uploaded file temporarily
    temp_dir = tempfile.gettempdir()
    temp_filename = f"{uuid.uuid4()}_{file.filename}"
    temp_path = os.path.join(temp_dir, temp_filename)
    
    try:
        # Save uploaded file
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Simulate realistic analysis
        import time
        time.sleep(2)  # Simulate processing time
        
        # Generate realistic results
        predictions = ["fake", "real"]
        prediction = random.choice(predictions)
        confidence = random.uniform(0.65, 0.95)
        faces_analyzed = random.randint(3, 8)
        
        result = {
            "prediction": prediction,
            "confidence": float(confidence),
            "faces_analyzed": faces_analyzed,
            "analysis": {
                "confidence_breakdown": {
                    "raw_confidence": float(confidence),
                    "quality_adjusted": float(confidence * 0.9),
                    "consistency": random.uniform(0.8, 0.95),
                    "quality_score": random.uniform(0.7, 0.9),
                },
                "heatmaps_generated": 2,
                "suspicious_frames": random.randint(1, 5) if prediction == "fake" else 0,
            },
            "filename": file.filename,
            "file_size": len(content),
            "timestamp": datetime.now().isoformat(),
            "processing_time": 2.0,
            "model_info": {
                "architecture": "ResNet18",
                "accuracy": "65%",
                "parameters": "11.2M",
                "version": "simple-v1"
            }
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/stats")
async def get_system_stats():
    """Get system statistics"""
    return {
        "system": {
            "status": "running",
            "timestamp": datetime.now().isoformat(),
            "total_predictions": random.randint(100, 500),
            "accuracy": "65%"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)