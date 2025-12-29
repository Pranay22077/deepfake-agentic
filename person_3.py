# ============================================
# PERSON 3: ENVIRONMENT SPECIALIST
# LL-Model (Low-Light) & TM-Model (Temporal)
# ============================================

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import cv2
import os
import numpy as np
from facenet_pytorch import MTCNN
from tqdm import tqdm
import json

# ============================================
# 1. LOW-LIGHT MODEL (LL-Model)
# ============================================

class LLModel(nn.Module):
    """Low-Light Specialist Model"""
    def __init__(self, num_classes=2):
        super(LLModel, self).__init__()
        
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        if len(x.shape) == 5:
            batch_size, num_frames = x.shape[:2]
            x = x.view(-1, *x.shape[2:])
            features = self.backbone(x)
            features = features.view(batch_size, num_frames, -1)
            features = features.mean(dim=1)
            return features
        else:
            return self.backbone(x)

# ============================================
# 2. TEMPORAL MODEL (TM-Model)
# ============================================

class TMModel(nn.Module):
    """Temporal Specialist Model with LSTM"""
    def __init__(self, num_classes=2, hidden_size=256, num_layers=2):
        super(TMModel, self).__init__()
        
        # CNN backbone for frame features
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        self.backbone.fc = nn.Identity()  # Remove classifier
        
        # LSTM for temporal analysis
        self.lstm = nn.LSTM(
            input_size=512,  # ResNet18 feature size
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # x shape: [batch, frames, channels, height, width]
        batch_size, num_frames = x.shape[:2]
        
        # Extract features for each frame
        x = x.view(-1, *x.shape[2:])  # [batch*frames, C, H, W]
        features = self.backbone(x)   # [batch*frames, 512]
        features = features.view(batch_size, num_frames, -1)  # [batch, frames, 512]
        
        # Temporal analysis with LSTM
        lstm_out, _ = self.lstm(features)  # [batch, frames, hidden]
        
        # Use last hidden state for classification
        last_hidden = lstm_out[:, -1, :]  # [batch, hidden]
        
        output = self.classifier(last_hidden)
        return output

# ============================================
# 3. DATASET AUGMENTATION FUNCTIONS
# ============================================

def apply_lowlight_augmentation(frame, brightness_factor=0.4, noise_std=15):
    """
    Simulate low-light conditions:
    1. Reduce brightness
    2. Add Gaussian noise
    """
    # Reduce brightness
    frame = frame.astype(np.float32) * brightness_factor
    
    # Add Gaussian noise
    noise = np.random.normal(0, noise_std, frame.shape)
    frame = frame + noise
    
    # Clip to valid range
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    
    return frame

def create_lowlight_dataset(video_dir, output_dir, brightness_levels=[0.3, 0.4, 0.5]):
    """Create low-light versions of all videos"""
    os.makedirs(output_dir, exist_ok=True)
    
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    
    for video_file in tqdm(video_files, desc="Creating low-light dataset"):
        input_path = os.path.join(video_dir, video_file)
        
        for brightness in brightness_levels:
            output_name = f"lowlight_{int(brightness*100)}_{video_file}"
            output_path = os.path.join(output_dir, output_name)
            
            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = apply_lowlight_augmentation(frame, brightness)
                out.write(frame)
            
            cap.release()
            out.release()
    
    print(f"✅ Created low-light dataset with {len(brightness_levels)} brightness levels")

# ============================================
# 4. MODEL LOADING FUNCTIONS
# ============================================

def load_ll_model(model_path="ll_model_student (1).pt", device='cpu'):
    """Load the trained Low-Light model"""
    model = LLModel(num_classes=2)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✅ Loaded LL-Model from {model_path}")
    else:
        print(f"⚠️ Model file not found: {model_path}")
    
    model.to(device)
    model.eval()
    return model

def load_tm_model(model_path="tm_model_student.pt", device='cpu'):
    """Load the trained Temporal model"""
    model = TMModel(num_classes=2, hidden_size=256, num_layers=2)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✅ Loaded TM-Model from {model_path}")
    else:
        print(f"⚠️ Model file not found: {model_path}")
    
    model.to(device)
    model.eval()
    return model

# ============================================
# 5. INFERENCE FUNCTIONS
# ============================================

def extract_faces_from_video(video_path, max_frames=10, device='cpu'):
    """Extract faces from video for inference"""
    cap = cv2.VideoCapture(video_path)
    detector = MTCNN(keep_all=False, device=device)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_rate = max(1, total_frames // max_frames)
    
    faces = []
    for i in range(0, total_frames, sample_rate):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            face = detector(frame_rgb)
            if face is not None:
                faces.append(face)
        except:
            pass
        
        if len(faces) >= max_frames:
            break
    
    cap.release()
    
    if len(faces) == 0:
        raise ValueError("No faces detected in video")
    
    # Normalize faces
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    normalized_faces = [transform(face) for face in faces]
    return torch.stack(normalized_faces)

def predict_with_ll_model(video_path, model_path="ll_model_student (1).pt"):
    """Predict using Low-Light model"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    model = load_ll_model(model_path, device)
    
    # Extract faces
    faces = extract_faces_from_video(video_path, max_frames=10, device=device)
    faces = faces.to(device)
    
    # Predict
    with torch.no_grad():
        logits = model(faces.unsqueeze(0))  # Add batch dimension
        probs = torch.softmax(logits, dim=1)
        prediction = probs[0, 1].item()  # Probability of fake
    
    confidence = prediction if prediction > 0.5 else 1 - prediction
    result = 'FAKE' if prediction > 0.5 else 'REAL'
    
    return {
        'prediction': result,
        'confidence': confidence,
        'fake_probability': prediction,
        'model_used': 'LL-Model (Low-Light Specialist)'
    }

def predict_with_tm_model(video_path, model_path="tm_model_student.pt"):
    """Predict using Temporal model"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    model = load_tm_model(model_path, device)
    
    # Extract more frames for temporal analysis
    faces = extract_faces_from_video(video_path, max_frames=15, device=device)
    faces = faces.to(device)
    
    # Predict
    with torch.no_grad():
        logits = model(faces.unsqueeze(0))  # Add batch dimension
        probs = torch.softmax(logits, dim=1)
        prediction = probs[0, 1].item()  # Probability of fake
    
    confidence = prediction if prediction > 0.5 else 1 - prediction
    result = 'FAKE' if prediction > 0.5 else 'REAL'
    
    return {
        'prediction': result,
        'confidence': confidence,
        'fake_probability': prediction,
        'model_used': 'TM-Model (Temporal Specialist)'
    }

# ============================================
# 6. VIDEO ANALYSIS FUNCTIONS
# ============================================

def analyze_video_brightness(video_path):
    """Analyze average brightness of video"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    brightness_samples = []
    for i in range(0, total_frames, max(1, total_frames // 10)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness_samples.append(np.mean(gray))
    
    cap.release()
    avg_brightness = np.mean(brightness_samples) if brightness_samples else 128
    return avg_brightness

def select_best_model_for_video(video_path):
    """Select the best model based on video characteristics"""
    brightness = analyze_video_brightness(video_path)
    
    if brightness < 50:  # Dark video
        return 'll_model', 'Low-light conditions detected'
    else:
        return 'tm_model', 'Using temporal analysis'

# ============================================
# 7. MAIN INFERENCE FUNCTION
# ============================================

def predict_deepfake(video_path, model_choice='auto'):
    """
    Main function to predict deepfake using Person 3's models
    
    Args:
        video_path: Path to video file
        model_choice: 'll_model', 'tm_model', or 'auto'
    
    Returns:
        Dictionary with prediction results
    """
    try:
        if model_choice == 'auto':
            model_choice, reason = select_best_model_for_video(video_path)
            print(f"Auto-selected {model_choice}: {reason}")
        
        if model_choice == 'll_model':
            result = predict_with_ll_model(video_path)
        elif model_choice == 'tm_model':
            result = predict_with_tm_model(video_path)
        else:
            raise ValueError(f"Unknown model choice: {model_choice}")
        
        # Add video analysis
        brightness = analyze_video_brightness(video_path)
        result['video_brightness'] = brightness
        result['video_path'] = video_path
        
        return result
        
    except Exception as e:
        return {
            'error': str(e),
            'prediction': 'ERROR',
            'confidence': 0.0,
            'model_used': 'None'
        }

# ============================================
# 8. TESTING FUNCTIONS
# ============================================

def test_models():
    """Test both models with sample videos"""
    test_videos = ['test_video_short.mp4', 'test_video_long.mp4']
    
    for video in test_videos:
        if os.path.exists(video):
            print(f"\n🎬 Testing {video}")
            print("=" * 50)
            
            try:
                # Test LL-Model
                print("Testing LL-Model...")
                ll_result = predict_with_ll_model(video)
                print(f"LL-Model: {ll_result['prediction']} ({ll_result['confidence']:.3f})")
            except Exception as e:
                print(f"LL-Model Error: {e}")
            
            try:
                # Test TM-Model
                print("Testing TM-Model...")
                tm_result = predict_with_tm_model(video)
                print(f"TM-Model: {tm_result['prediction']} ({tm_result['confidence']:.3f})")
            except Exception as e:
                print(f"TM-Model Error: {e}")
            
            try:
                # Test Auto-selection
                print("Testing Auto-selection...")
                auto_result = predict_deepfake(video, 'auto')
                print(f"Auto: {auto_result['prediction']} ({auto_result['confidence']:.3f}) - {auto_result['model_used']}")
            except Exception as e:
                print(f"Auto-selection Error: {e}")
        else:
            print(f"⚠️ Test video not found: {video}")

def test_model_loading():
    """Test if models can be loaded successfully"""
    print("\n🔧 Testing Model Loading...")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Test LL-Model loading
    try:
        ll_model = load_ll_model("ll_model_student (1).pt", device)
        print("✅ LL-Model loaded successfully")
        
        # Test with dummy input
        dummy_input = torch.randn(1, 10, 3, 224, 224).to(device)
        with torch.no_grad():
            output = ll_model(dummy_input)
            print(f"   LL-Model output shape: {output.shape}")
    except Exception as e:
        print(f"❌ LL-Model loading failed: {e}")
    
    # Test TM-Model loading
    try:
        tm_model = load_tm_model("tm_model_student.pt", device)
        print("✅ TM-Model loaded successfully")
        
        # Test with dummy input
        dummy_input = torch.randn(1, 15, 3, 224, 224).to(device)
        with torch.no_grad():
            output = tm_model(dummy_input)
            print(f"   TM-Model output shape: {output.shape}")
    except Exception as e:
        print(f"❌ TM-Model loading failed: {e}")

def model_info():
    """Display information about the models"""
    print("🛡️ Person 3 - Environment Specialist Models")
    print("=" * 50)
    print("📱 LL-Model (Low-Light Specialist)")
    print("   - Detects deepfakes in dark/noisy videos")
    print("   - Trained on low-brightness augmented data")
    print("   - File: ll_model_student (1).pt")
    print()
    print("⏰ TM-Model (Temporal Specialist)")
    print("   - Detects frame inconsistencies over time")
    print("   - Uses LSTM for temporal analysis")
    print("   - File: tm_model_student.pt")
    print()
    print("🤖 Auto-selection Logic:")
    print("   - Brightness < 50: Use LL-Model")
    print("   - Brightness >= 50: Use TM-Model")

# ============================================
# 9. MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    print("🛡️ E-Raksha Person 3: Environment Specialist")
    print("=" * 50)
    
    # Display model info
    model_info()
    
    # Test model loading first
    test_model_loading()
    
    # Test models if test videos exist
    print("\n🧪 Running Model Tests...")
    test_models()
    
    # Example usage
    print("\n📖 Example Usage:")
    print("from person_3 import predict_deepfake")
    print("result = predict_deepfake('your_video.mp4')")
    print("print(result)")