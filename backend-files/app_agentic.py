#!/usr/bin/env python3
"""
E-Raksha Agentic Backend API
FastAPI server with unified agentic deepfake detection system
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import sys
import tempfile
import uuid
from datetime import datetime
import json
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our agentic system
from src.agent.eraksha_agent import ErakshAgent

# Load environment variables
load_dotenv()

app = FastAPI(
    title="E-Raksha Agentic Deepfake Detection API", 
    version="2.0.0",
    description="Unified agentic system with intelligent model routing"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agentic system
agent = None

# Pydantic models for request/response
class FeedbackRequest(BaseModel):
    video_filename: str
    user_label: str  # 'real', 'fake', 'unknown'
    user_confidence: float = None
    comments: str = None

class AnalysisResponse(BaseModel):
    success: bool
    prediction: str = None
    confidence: float = None
    confidence_level: str = None
    explanation: str = None
    best_model: str = None
    specialists_used: list = []
    all_predictions: dict = {}
    video_characteristics: dict = {}
    processing_time: float = None
    request_id: str = None
    error: str = None
    timestamp: str = None

def load_agentic_system():
    """Load the unified agentic system"""
    global agent
    
    try:
        print("[INIT] Loading E-Raksha Agentic System...")
        agent = ErakshAgent(device='auto')
        print("[OK] Agentic system loaded successfully!")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load agentic system: {e}")
        agent = None
        return False

@app.on_event("startup")
async def startup_event():
    """Load agentic system on startup"""
    success = load_agentic_system()
    if success:
        print("[OK] E-Raksha Agentic API started successfully")
    else:
        print("[WARNING] API started but agentic system failed to load")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "E-Raksha Agentic Deepfake Detection API",
        "version": "2.0.0",
        "status": "running",
        "agentic_system_loaded": agent is not None,
        "models_available": _get_model_status() if agent else {}
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    model_status = _get_model_status() if agent else {}
    
    return {
        "status": "healthy" if agent is not None else "degraded",
        "agentic_system": agent is not None,
        "models": model_status,
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }

@app.get("/models")
async def get_model_info():
    """Get information about loaded models"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agentic system not loaded")
    
    model_info = {
        "baseline": {
            "name": "BG-Model (Baseline Generalist)",
            "loaded": agent.models.get('student') is not None,
            "description": "Primary deepfake detection model"
        },
        "audiovisual": {
            "name": "AV-Model (Audio-Visual Specialist)",
            "loaded": agent.models.get('av') is not None,
            "description": "Lip-sync and audio-visual inconsistency detection",
            "accuracy": "93%"
        },
        "compression": {
            "name": "CM-Model (Compression Specialist)",
            "loaded": agent.models.get('cm') is not None,
            "description": "Handles compressed videos (WhatsApp, Instagram, etc.)"
        },
        "rerecording": {
            "name": "RR-Model (Re-recording Specialist)",
            "loaded": agent.models.get('rr') is not None,
            "description": "Handles re-recorded/screen-captured videos"
        },
        "lowlight": {
            "name": "LL-Model (Low-light Specialist)",
            "loaded": agent.models.get('ll') is not None,
            "description": "Optimized for low-light and dark videos"
        },
        "temporal": {
            "name": "TM-Model (Temporal Specialist)",
            "loaded": agent.models.get('tm') is not None,
            "description": "Analyzes temporal inconsistencies across frames"
        }
    }
    
    return {
        "models": model_info,
        "total_loaded": sum(1 for m in model_info.values() if m["loaded"]),
        "routing_enabled": True,
        "intelligent_selection": True
    }

@app.post("/predict", response_model=AnalysisResponse)
async def predict_deepfake(file: UploadFile = File(...)):
    """
    Agentic deepfake prediction with intelligent model routing
    """
    
    if agent is None:
        raise HTTPException(status_code=503, detail="Agentic system not loaded")
    
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
        
        # Run agentic analysis
        result = agent.predict(temp_path)
        
        # Add metadata
        if result['success']:
            result.update({
                "filename": file.filename,
                "file_size": len(content),
                "timestamp": datetime.now().isoformat(),
                "api_version": "2.0.0"
            })
        
        return AnalysisResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Submit user feedback for model improvement"""
    try:
        # Log feedback (in production, save to database)
        feedback_data = {
            "video_filename": feedback.video_filename,
            "user_label": feedback.user_label,
            "user_confidence": feedback.user_confidence,
            "comments": feedback.comments,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save to feedback log
        feedback_file = "feedback_log.json"
        if os.path.exists(feedback_file):
            with open(feedback_file, 'r') as f:
                feedback_log = json.load(f)
        else:
            feedback_log = []
        
        feedback_log.append(feedback_data)
        
        with open(feedback_file, 'w') as f:
            json.dump(feedback_log, f, indent=2)
        
        return {
            "message": "Feedback received successfully",
            "timestamp": datetime.now().isoformat(),
            "status": "saved"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process feedback: {str(e)}")

@app.get("/stats")
async def get_system_stats():
    """Get system statistics"""
    try:
        model_status = _get_model_status() if agent else {}
        
        # Count loaded models
        loaded_models = sum(1 for status in model_status.values() if status)
        total_models = len(model_status)
        
        stats = {
            "agentic_system_loaded": agent is not None,
            "models_loaded": f"{loaded_models}/{total_models}",
            "model_details": model_status,
            "routing_enabled": agent is not None,
            "api_version": "2.0.0",
            "timestamp": datetime.now().isoformat()
        }
        
        # Add feedback stats if available
        feedback_file = "feedback_log.json"
        if os.path.exists(feedback_file):
            with open(feedback_file, 'r') as f:
                feedback_log = json.load(f)
            stats["feedback_count"] = len(feedback_log)
        else:
            stats["feedback_count"] = 0
        
        return stats
        
    except Exception as e:
        return {
            "error": f"Failed to get stats: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

@app.post("/reload")
async def reload_agentic_system():
    """Reload the agentic system (admin endpoint)"""
    try:
        success = load_agentic_system()
        
        if success:
            return {
                "message": "Agentic system reloaded successfully",
                "models_loaded": _get_model_status(),
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "message": "Failed to reload agentic system",
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {str(e)}")

def _get_model_status():
    """Get status of all models"""
    if agent is None:
        return {}
    
    return {
        "student": agent.models.get('student') is not None,
        "av": agent.models.get('av') is not None,
        "cm": agent.models.get('cm') is not None,
        "rr": agent.models.get('rr') is not None,
        "ll": agent.models.get('ll') is not None,
        "tm": agent.models.get('tm') is not None
    }

if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', '8000'))
    
    print(f"[INIT] Starting E-Raksha Agentic API on {host}:{port}")
    uvicorn.run(app, host=host, port=port)