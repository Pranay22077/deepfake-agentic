#!/usr/bin/env python3
"""
E-Raksha Backend API
FastAPI server for deepfake detection with your trained model
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import pickle
import os
import tempfile
import uuid
from datetime import datetime
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our enhanced modules
from db.supabase_client import supabase_client
from api.inference import EnhancedInference

app = FastAPI(title="E-Raksha Deepfake Detection API", version="1.0.0")

# Import and include API enhancements
try:
    from api_enhancements import router as enhancements_router, add_to_history
    app.include_router(enhancements_router, prefix="/api/v1", tags=["enhancements"])
    print("✅ API enhancements loaded successfully")
except ImportError as e:
    print(f"⚠️  API enhancements not available: {e}")
    add_to_history = None

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and enhanced inference
model = None
enhanced_inference = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Pydantic models for request/response
class FeedbackRequest(BaseModel):
    video_filename: str
    user_label: str  # 'real', 'fake', 'unknown'
    user_confidence: float = None
    comments: str = None

class StudentModel(nn.Module):
    """Kaggle-trained ResNet18 model (matches the downloaded model)"""
    def __init__(self, num_classes=2):
        super(StudentModel, self).__init__()
        
        # Use pretrained ResNet18 for better performance
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        
        # Replace final layer (matches Kaggle training architecture)
        in_features = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),  # 512 → 256
            nn.ReLU(),
            nn.BatchNorm1d(256),          # BatchNorm layer
            nn.Dropout(0.2),
            nn.Linear(256, 128),          # 256 → 128
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)   # 128 → 2
        )
    
    def forward(self, x):
        return self.backbone(x)

def load_model():
    """Load your trained model from Step 1"""
    global model, enhanced_inference
    
    # Look for model file (prioritize fixed model)
    model_paths = [
        "./fixed_deepfake_model.pt",  # New fixed model
        "../fixed_deepfake_model.pt",
        "../kaggle_outputs_20251228_043850/baseline_student.pkl",
        "../baseline_student.pkl",
        "./baseline_student.pkl",
        "../kaggle_outputs/baseline_student.pkl",
        "../kaggle_outputs/baseline_student.pt"
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        raise FileNotFoundError("Model file not found. Please place your trained model in the backend directory.")
    
    print(f"Loading model from: {model_path}")
    
    # Create model instance
    model = StudentModel(num_classes=2)
    
    try:
        if model_path.endswith('.pt'):
            # Load PyTorch format (Kaggle-trained model)
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded Kaggle-trained model from {model_path}")
                if 'best_acc' in checkpoint:
                    print(f"Model accuracy: {checkpoint['best_acc']:.2f}%")
                if 'epoch' in checkpoint:
                    print(f"Training epoch: {checkpoint['epoch']}")
            else:
                model.load_state_dict(checkpoint, strict=False)
                print(f"Loaded PyTorch state dict from {model_path}")
                
        elif model_path.endswith('.pkl'):
            # Load pickle format (old broken model)
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Convert numpy arrays back to tensors
            state_dict = {}
            for name, param_array in model_data.items():
                state_dict[name] = torch.from_numpy(param_array)
            
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded pickle model from {model_path} (strict=False for missing BatchNorm stats)")
            print("⚠️  WARNING: Using old broken model - consider replacing with fixed model")
        else:
            raise ValueError(f"Unsupported model format: {model_path}")
        
        model.to(device)
        model.eval()
        print(f"Model loaded on device: {device}")
        
        # Initialize enhanced inference
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        enhanced_inference = EnhancedInference(model, device, transform)
        print("✅ Enhanced inference initialized")
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

# Image preprocessing (matches your training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_faces_from_video(video_path, max_faces=5):
    """Extract faces from video (simplified version of your preprocessing)"""
    faces = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return faces
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return faces
    
    # Sample frames
    step = max(1, total_frames // (max_faces * 2))
    frame_count = 0
    
    # Simple face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    while len(faces) < max_faces and frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % step == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Try face detection
            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
            detected_faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(detected_faces) > 0:
                # Use largest face
                largest_face = max(detected_faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                
                # Add padding
                padding = 20
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(rgb_frame.shape[1], x + w + padding)
                y2 = min(rgb_frame.shape[0], y + h + padding)
                
                face = rgb_frame[y1:y2, x1:x2]
            else:
                # Fallback to center crop
                h, w = rgb_frame.shape[:2]
                size = min(h, w)
                y_start = (h - size) // 2
                x_start = (w - size) // 2
                face = rgb_frame[y_start:y_start+size, x_start:x_start+size]
            
            if face.size > 0:
                face_resized = cv2.resize(face, (224, 224))
                faces.append(face_resized)
        
        frame_count += 1
    
    cap.release()
    return faces

def predict_video(video_path):
    """Run inference on video"""
    if model is None:
        raise RuntimeError("Model not loaded")
    
    # Extract faces
    faces = extract_faces_from_video(video_path, max_faces=8)
    
    if not faces:
        return {
            "error": "No faces detected in video",
            "prediction": "unknown",
            "confidence": 0.0
        }
    
    # Process faces
    predictions = []
    confidences = []
    
    with torch.no_grad():
        for face in faces:
            # Convert to PIL and apply transforms
            pil_face = Image.fromarray(face)
            input_tensor = transform(pil_face).unsqueeze(0).to(device)
            
            # Get prediction
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            
            # Get prediction (0=real, 1=fake)
            pred_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][pred_class].item()
            
            predictions.append(pred_class)
            confidences.append(confidence)
    
    # Aggregate results
    avg_confidence = np.mean(confidences)
    fake_votes = sum(predictions)
    total_votes = len(predictions)
    
    # Final decision
    if fake_votes > total_votes / 2:
        final_prediction = "fake"
        final_confidence = avg_confidence
    else:
        final_prediction = "real"
        final_confidence = avg_confidence
    
    return {
        "prediction": final_prediction,
        "confidence": float(final_confidence),
        "faces_analyzed": len(faces),
        "fake_votes": fake_votes,
        "total_votes": total_votes,
        "individual_confidences": confidences
    }

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        load_model()
        print("E-Raksha API started successfully")
    except Exception as e:
        print(f"Failed to start API: {e}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "E-Raksha Deepfake Detection API",
        "status": "running",
        "model_loaded": model is not None,
        "device": str(device)
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict")
async def predict_deepfake(file: UploadFile = File(...)):
    """Enhanced prediction endpoint with heatmaps and detailed analysis"""
    
    if model is None or enhanced_inference is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
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
        
        # Run enhanced analysis
        result = enhanced_inference.analyze_video_comprehensive(temp_path)
        
        # Add metadata
        result.update({
            "filename": file.filename,
            "file_size": len(content),
            "timestamp": datetime.now().isoformat(),
            "model_info": {
                "architecture": "ResNet18",
                "accuracy": "65%",  # From Kaggle training
                "parameters": "11.2M",
                "version": "kaggle-v1"
            }
        })
        
        # Log to database
        if 'error' not in result:
            supabase_client.log_inference(
                video_filename=file.filename,
                result=result,
                confidence=result.get('confidence', 0.0)
            )
            
            # Add to history tracking
            if add_to_history:
                processing_time = result.get('processing_time', 0.0)
                add_to_history(
                    filename=file.filename,
                    prediction=result.get('prediction', 'unknown'),
                    confidence=result.get('confidence', 0.0),
                    processing_time=processing_time
                )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Enhanced feedback collection endpoint"""
    try:
        # Validate feedback data
        if feedback.user_label not in ['real', 'fake', 'unknown']:
            raise HTTPException(status_code=400, detail="Invalid label. Must be 'real', 'fake', or 'unknown'")
        
        # Save to database
        success = supabase_client.save_feedback(
            video_filename=feedback.video_filename,
            user_label=feedback.user_label,
            user_confidence=feedback.user_confidence
        )
        
        if success:
            return {
                "message": "Feedback received successfully",
                "timestamp": datetime.now().isoformat(),
                "status": "saved"
            }
        else:
            return {
                "message": "Feedback received but not saved to database",
                "timestamp": datetime.now().isoformat(),
                "status": "logged_only"
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process feedback: {str(e)}")

@app.get("/stats")
async def get_system_stats():
    """Get system statistics and database info"""
    try:
        # Get database stats
        db_stats = supabase_client.get_inference_stats()
        
        # Add system info
        system_stats = {
            "model_loaded": model is not None,
            "enhanced_inference": enhanced_inference is not None,
            "device": str(device),
            "timestamp": datetime.now().isoformat()
        }
        
        return {
            "system": system_stats,
            "database": db_stats
        }
        
    except Exception as e:
        return {
            "error": f"Failed to get stats: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', '8000'))
    
    uvicorn.run(app, host=host, port=port)