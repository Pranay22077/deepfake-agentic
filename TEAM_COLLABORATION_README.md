# 🛡️ E-RAKSHA: Agentic Deepfake Detection System
## Team Collaboration Guide - PS-2 Hackathon

---

## 📋 TEAM OVERVIEW

| Member | Role | Primary Responsibility |
|--------|------|----------------------|
| **Person 1 (Pranay)** | Team Lead & Integration | BG-Model (Baseline), System Integration, Web App |
| **Person 2** | Compression Specialist | CM-Model (Compression), RR-Model (Re-recording) |
| **Person 3** | Environment Specialist | LL-Model (Low-Light), TM-Model (Temporal) |
| **Person 4** | Agent Developer | LangGraph Agent, AV-Model (Audio-Visual) |

---

## 🎯 PROJECT GOAL

Build an **intelligent, agentic deepfake detection system** that:
- Uses **multiple specialized neural networks** (not just one model)
- Has a **LangGraph-based agent** that decides which model to use
- Provides **explainable results** with Grad-CAM heatmaps
- Handles different video types (compressed, re-recorded, low-light, etc.)

---

## 📊 DATASET INFORMATION

**Primary Dataset: DFDC (Deepfake Detection Challenge)**
- Size: ~4-5 GB subset
- Location: Kaggle
- Link: https://www.kaggle.com/c/deepfake-detection-challenge

**All team members will use the SAME dataset but create DIFFERENT training variations.**

---

## 🔧 COMMON SETUP (ALL TEAM MEMBERS)

### Step 1: Fork the Repository
```bash
# Go to: https://github.com/Pranay22077/deepfake-agentic
# Click "Fork" to create your own copy
```

### Step 2: Clone Your Fork
```bash
git clone https://github.com/YOUR_USERNAME/deepfake-agentic.git
cd deepfake-agentic
```


### Step 3: Kaggle Setup (For Training)
```bash
# 1. Create Kaggle account: https://www.kaggle.com
# 2. Go to Account Settings → API → Create New Token
# 3. Download kaggle.json
# 4. Upload to Kaggle notebook when needed
```

### Step 4: Create Your Branch
```bash
git checkout -b your-name/your-task
# Example: git checkout -b person2/compression-model
```

---

## 📁 PROJECT STRUCTURE

```
deepfake-agentic/
├── src/
│   ├── models/
│   │   ├── baseline.py          # Person 1: BG-Model
│   │   ├── compression.py       # Person 2: CM-Model
│   │   ├── rerecording.py       # Person 2: RR-Model
│   │   ├── lowlight.py          # Person 3: LL-Model
│   │   ├── temporal.py          # Person 3: TM-Model
│   │   └── audiovisual.py       # Person 4: AV-Model
│   ├── agent/
│   │   ├── graph.py             # Person 4: LangGraph Agent
│   │   ├── nodes.py             # Person 4: Agent Nodes
│   │   └── policy.py            # Person 4: Decision Policy
│   ├── train/
│   │   ├── train_baseline.py    # Person 1
│   │   ├── train_compression.py # Person 2
│   │   ├── train_lowlight.py    # Person 3
│   │   └── distill_student.py   # All (knowledge distillation)
│   └── preprocess/
│       ├── extract_faces.py     # Shared
│       ├── augmentation.py      # Shared
│       └── create_datasets.py   # Person 2, 3
├── models/                      # Trained model weights (.pt files)
│   ├── bg_model_student.pt      # Person 1
│   ├── cm_model_student.pt      # Person 2
│   ├── rr_model_student.pt      # Person 2
│   ├── ll_model_student.pt      # Person 3
│   ├── tm_model_student.pt      # Person 3
│   └── av_model_student.pt      # Person 4
├── kaggle_notebooks/            # Kaggle training notebooks
│   ├── person1_baseline.py
│   ├── person2_compression.py
│   ├── person3_lowlight.py
│   └── person4_audiovisual.py
├── backend/                     # Person 1: Web API
├── frontend/                    # Person 1: Web Interface
└── config/
    └── agent_config.yaml        # Person 4: Agent configuration
```

---


# 👤 PERSON 1 (PRANAY) - TEAM LEAD & INTEGRATION

## Your Responsibilities:
1. **BG-Model (Baseline Generalist)** - Train the main baseline model
2. **System Integration** - Combine all team members' work
3. **Web Application** - Backend API + Frontend
4. **Final Deployment** - Docker, testing, documentation

---

## Task 1.1: Train BG-Model (Baseline Generalist)

### What is BG-Model?
The baseline model trained on ALL video types. It's the "first pass" model that handles general cases.

### Kaggle Notebook Code:

```python
# ============================================
# PERSON 1: BG-MODEL (BASELINE GENERALIST)
# ============================================
# Run this in Kaggle with GPU enabled

!pip install facenet-pytorch pytorch-grad-cam

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

# ============================================
# 1. DATASET SETUP
# ============================================

class DFDCDataset(Dataset):
    """DFDC Dataset for baseline training"""
    def __init__(self, video_dir, metadata_path, transform=None, max_frames=10):
        self.video_dir = video_dir
        self.transform = transform
        self.max_frames = max_frames
        self.detector = MTCNN(keep_all=False, device='cuda')
        
        # Load metadata
        import json
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.video_files = list(self.metadata.keys())
        
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        video_name = self.video_files[idx]
        video_path = os.path.join(self.video_dir, video_name)
        label = 1 if self.metadata[video_name]['label'] == 'FAKE' else 0
        
        # Extract faces from video
        faces = self.extract_faces(video_path)
        
        if len(faces) == 0:
            # Return dummy tensor if no faces found
            faces = [torch.zeros(3, 224, 224)]
        
        # Stack faces
        faces = torch.stack(faces[:self.max_frames])
        
        return faces, label
    
    def extract_faces(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        faces = []
        sample_rate = max(1, total_frames // self.max_frames)
        
        for i in range(0, total_frames, sample_rate):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            try:
                face = self.detector(frame_rgb)
                if face is not None:
                    if self.transform:
                        face = self.transform(face)
                    faces.append(face)
            except:
                pass
            
            if len(faces) >= self.max_frames:
                break
        
        cap.release()
        return faces

# ============================================
# 2. MODEL ARCHITECTURE
# ============================================

class BGModel(nn.Module):
    """Baseline Generalist Model - ResNet18 based"""
    def __init__(self, num_classes=2):
        super(BGModel, self).__init__()
        
        # Load pretrained ResNet18
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        
        # Replace classifier with custom head
        in_features = self.backbone.fc.in_features  # 512
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
        # x shape: [batch, frames, channels, height, width]
        if len(x.shape) == 5:
            batch_size, num_frames = x.shape[:2]
            x = x.view(-1, *x.shape[2:])  # Flatten batch and frames
            features = self.backbone(x)
            features = features.view(batch_size, num_frames, -1)
            features = features.mean(dim=1)  # Average over frames
            return features
        else:
            return self.backbone(x)

# ============================================
# 3. TRAINING LOOP
# ============================================

def train_bg_model():
    # Hyperparameters
    BATCH_SIZE = 8
    EPOCHS = 20
    LR = 1e-4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Training on: {DEVICE}")
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Dataset (adjust paths for Kaggle)
    train_dataset = DFDCDataset(
        video_dir='/kaggle/input/deepfake-detection-challenge/train_sample_videos',
        metadata_path='/kaggle/input/deepfake-detection-challenge/train_sample_videos/metadata.json',
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    # Model
    model = BGModel(num_classes=2).to(DEVICE)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Training
    best_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')
        for faces, labels in pbar:
            faces = faces.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(faces)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': total_loss/total, 'acc': 100.*correct/total})
        
        scheduler.step()
        
        epoch_acc = 100. * correct / total
        print(f'Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Acc={epoch_acc:.2f}%')
        
        # Save best model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc,
                'config': {
                    'architecture': 'ResNet18',
                    'num_classes': 2,
                    'model_type': 'BG-Model (Baseline Generalist)'
                }
            }, 'bg_model_student.pt')
            print(f'✅ Saved best model with accuracy: {best_acc:.2f}%')
    
    print(f'\n🎉 Training complete! Best accuracy: {best_acc:.2f}%')
    return model

# Run training
if __name__ == '__main__':
    model = train_bg_model()
```

### After Training:
1. Download `bg_model_student.pt` from Kaggle
2. Place in `models/` folder
3. Commit and push to your branch

---


## Task 1.2: System Integration

After all team members complete their models, you will:

1. **Collect all model weights** from team members:
   - `bg_model_student.pt` (yours)
   - `cm_model_student.pt` (Person 2)
   - `rr_model_student.pt` (Person 2)
   - `ll_model_student.pt` (Person 3)
   - `tm_model_student.pt` (Person 3)
   - `av_model_student.pt` (Person 4)

2. **Integrate the LangGraph agent** (from Person 4)

3. **Build the web application** (backend + frontend)

4. **Test the complete system**

5. **Deploy with Docker**

---

## Task 1.3: Web Application

### Backend (FastAPI)

Create `backend/app.py`:

```python
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import os
import uuid

# Import the agent (from Person 4)
from src.agent.graph import run_inference_pipeline

app = FastAPI(title="E-Raksha Deepfake Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/v1/analyze")
async def analyze_video(file: UploadFile = File(...)):
    # Save uploaded file
    video_id = str(uuid.uuid4())
    video_path = f"temp/{video_id}_{file.filename}"
    
    os.makedirs("temp", exist_ok=True)
    with open(video_path, "wb") as f:
        f.write(await file.read())
    
    # Run agent inference
    result = run_inference_pipeline(video_path)
    
    # Cleanup
    os.remove(video_path)
    
    return result

@app.get("/health")
def health_check():
    return {"status": "healthy"}
```

---

# 👤 PERSON 2 - COMPRESSION & RE-RECORDING SPECIALIST

## Your Responsibilities:
1. **CM-Model (Compression Model)** - Detect deepfakes in compressed videos
2. **RR-Model (Re-Recording Model)** - Detect deepfakes in screen-recorded videos
3. **Create augmented datasets** for compression and re-recording

---

## Task 2.1: Create Compression Dataset

### What is Compression Augmentation?
WhatsApp, Instagram, and other platforms heavily compress videos. This destroys some deepfake artifacts but creates new compression artifacts. Your model learns to detect deepfakes even in compressed videos.

### Kaggle Notebook Code:

```python
# ============================================
# PERSON 2: COMPRESSION DATASET CREATION
# ============================================

import cv2
import subprocess
import os
from tqdm import tqdm

def create_compressed_video(input_path, output_path, bitrate_kbps):
    """
    Re-encode video at specified bitrate using FFmpeg
    Lower bitrate = more compression = more artifacts
    """
    cmd = [
        'ffmpeg', '-y',
        '-i', input_path,
        '-c:v', 'libx264',
        '-b:v', f'{bitrate_kbps}k',
        '-preset', 'medium',
        '-c:a', 'aac',
        '-b:a', '128k',
        output_path
    ]
    subprocess.run(cmd, capture_output=True)

def create_compression_dataset(video_dir, output_dir, bitrates=[200, 400, 600, 800]):
    """
    Create compressed versions of all videos at different bitrates
    """
    os.makedirs(output_dir, exist_ok=True)
    
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    
    for video_file in tqdm(video_files, desc="Creating compressed dataset"):
        input_path = os.path.join(video_dir, video_file)
        
        for bitrate in bitrates:
            output_name = f"{video_file[:-4]}_compressed_{bitrate}kbps.mp4"
            output_path = os.path.join(output_dir, output_name)
            
            create_compressed_video(input_path, output_path, bitrate)
    
    print(f"✅ Created compressed dataset with {len(bitrates)} bitrate levels")

# Run on Kaggle
create_compression_dataset(
    video_dir='/kaggle/input/deepfake-detection-challenge/train_sample_videos',
    output_dir='/kaggle/working/compressed_videos',
    bitrates=[200, 400, 600, 800]
)
```

---

## Task 2.2: Train CM-Model (Compression Specialist)

```python
# ============================================
# PERSON 2: CM-MODEL (COMPRESSION SPECIALIST)
# ============================================

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm

class CMModel(nn.Module):
    """Compression Specialist Model"""
    def __init__(self, num_classes=2):
        super(CMModel, self).__init__()
        
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

def train_cm_model():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EPOCHS = 15
    BATCH_SIZE = 8
    LR = 5e-5  # Lower LR for fine-tuning
    
    # Load compressed dataset (created in Task 2.1)
    # ... (same dataset loading as Person 1, but with compressed videos)
    
    model = CMModel(num_classes=2).to(DEVICE)
    
    # Optional: Load baseline weights for transfer learning
    # baseline_weights = torch.load('bg_model_student.pt')
    # model.load_state_dict(baseline_weights['model_state_dict'], strict=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    best_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        correct = 0
        total = 0
        
        # Training loop (same as Person 1)
        # ...
        
        epoch_acc = 100. * correct / total
        
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc,
                'config': {
                    'architecture': 'ResNet18',
                    'num_classes': 2,
                    'model_type': 'CM-Model (Compression Specialist)',
                    'trained_on': 'Compressed videos (200-800 kbps)'
                }
            }, 'cm_model_student.pt')
    
    print(f'🎉 CM-Model training complete! Best accuracy: {best_acc:.2f}%')

train_cm_model()
```

---

## Task 2.3: Create Re-Recording Dataset & Train RR-Model

```python
# ============================================
# PERSON 2: RE-RECORDING DATASET & RR-MODEL
# ============================================

import cv2
import numpy as np

def apply_moire_pattern(frame, frequency=0.1):
    """Simulate moiré pattern from screen recording"""
    h, w = frame.shape[:2]
    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)
    
    # Create moiré pattern
    pattern = np.sin(2 * np.pi * frequency * X) * np.sin(2 * np.pi * frequency * Y)
    pattern = (pattern * 10).astype(np.uint8)
    
    # Add pattern to frame
    frame = cv2.add(frame, np.stack([pattern]*3, axis=-1))
    return np.clip(frame, 0, 255).astype(np.uint8)

def apply_screen_grid(frame, grid_size=3):
    """Simulate pixel grid from screen recording"""
    h, w = frame.shape[:2]
    
    # Add subtle grid lines
    for i in range(0, h, grid_size):
        frame[i, :] = frame[i, :] * 0.95
    for j in range(0, w, grid_size):
        frame[:, j] = frame[:, j] * 0.95
    
    return frame.astype(np.uint8)

def create_rerecording_dataset(video_dir, output_dir):
    """Create re-recorded versions of videos"""
    os.makedirs(output_dir, exist_ok=True)
    
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    
    for video_file in tqdm(video_files, desc="Creating re-recording dataset"):
        input_path = os.path.join(video_dir, video_file)
        output_path = os.path.join(output_dir, f"rerecorded_{video_file}")
        
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
            
            # Apply re-recording artifacts
            frame = apply_moire_pattern(frame)
            frame = apply_screen_grid(frame)
            
            out.write(frame)
        
        cap.release()
        out.release()
    
    print(f"✅ Created re-recording dataset")

# Train RR-Model (same architecture as CM-Model, different dataset)
# Save as: rr_model_student.pt
```

### After Training:
1. Download `cm_model_student.pt` and `rr_model_student.pt`
2. Send to Person 1 (Pranay)
3. Create Pull Request with your code

---


# 👤 PERSON 3 - ENVIRONMENT SPECIALIST

## Your Responsibilities:
1. **LL-Model (Low-Light Model)** - Detect deepfakes in dark/noisy videos
2. **TM-Model (Temporal Model)** - Detect frame inconsistencies over time
3. **Create augmented datasets** for low-light and temporal analysis

---

## Task 3.1: Create Low-Light Dataset

### What is Low-Light Augmentation?
Videos recorded in dark environments have high noise and low contrast. Your model learns to detect deepfakes even in poor lighting conditions.

```python
# ============================================
# PERSON 3: LOW-LIGHT DATASET CREATION
# ============================================

import cv2
import numpy as np
from tqdm import tqdm
import os

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

# Run on Kaggle
create_lowlight_dataset(
    video_dir='/kaggle/input/deepfake-detection-challenge/train_sample_videos',
    output_dir='/kaggle/working/lowlight_videos',
    brightness_levels=[0.3, 0.4, 0.5]
)
```

---

## Task 3.2: Train LL-Model (Low-Light Specialist)

```python
# ============================================
# PERSON 3: LL-MODEL (LOW-LIGHT SPECIALIST)
# ============================================

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

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

def train_ll_model():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EPOCHS = 15
    BATCH_SIZE = 8
    LR = 5e-5
    
    model = LLModel(num_classes=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    best_acc = 0
    for epoch in range(EPOCHS):
        # Training loop (same structure as others)
        # Use low-light dataset created in Task 3.1
        pass
        
        # Save best model
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'best_acc': best_acc,
            'config': {
                'architecture': 'ResNet18',
                'num_classes': 2,
                'model_type': 'LL-Model (Low-Light Specialist)',
                'trained_on': 'Low-light videos (30-50% brightness)'
            }
        }, 'll_model_student.pt')
    
    print(f'🎉 LL-Model training complete!')

train_ll_model()
```

---

## Task 3.3: Train TM-Model (Temporal Specialist)

### What is Temporal Analysis?
Deepfakes often have frame-to-frame inconsistencies. The temporal model analyzes sequences of frames to detect unnatural motion or flickering.

```python
# ============================================
# PERSON 3: TM-MODEL (TEMPORAL SPECIALIST)
# ============================================

import torch
import torch.nn as nn
import torchvision.models as models

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

def train_tm_model():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EPOCHS = 15
    BATCH_SIZE = 4  # Smaller batch due to LSTM memory
    LR = 1e-4
    
    model = TMModel(num_classes=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    best_acc = 0
    for epoch in range(EPOCHS):
        # Training loop
        # IMPORTANT: Use more frames per video (15-20) for temporal analysis
        pass
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'best_acc': best_acc,
            'config': {
                'architecture': 'ResNet18 + LSTM',
                'num_classes': 2,
                'model_type': 'TM-Model (Temporal Specialist)',
                'lstm_hidden': 256,
                'lstm_layers': 2
            }
        }, 'tm_model_student.pt')
    
    print(f'🎉 TM-Model training complete!')

train_tm_model()
```

### After Training:
1. Download `ll_model_student.pt` and `tm_model_student.pt`
2. Send to Person 1 (Pranay)
3. Create Pull Request with your code

---


# 👤 PERSON 4 - AGENT DEVELOPER

## Your Responsibilities:
1. **LangGraph Agent** - Build the intelligent routing system
2. **AV-Model (Audio-Visual Model)** - Detect lip-sync mismatches
3. **Policy Logic** - Implement confidence thresholds and routing
4. **Explainability** - Grad-CAM heatmaps and explanations

---

## Task 4.1: Install LangGraph

```bash
pip install langgraph langchain
```

---

## Task 4.2: Implement Agent State

Create `src/agent/state.py`:

```python
# ============================================
# PERSON 4: AGENT STATE DEFINITION
# ============================================

from typing import TypedDict, List, Optional
import torch

class AgentState(TypedDict):
    # Input
    video_path: str
    request_id: str
    
    # Metadata
    metadata: dict
    bitrate: int
    fps: float
    resolution: tuple
    avg_brightness: float
    
    # Processing
    frames: List[torch.Tensor]
    faces_detected: int
    
    # Model predictions
    student_prediction: float
    student_confidence: float
    specialist_prediction: float
    specialist_confidence: float
    selected_specialist: str
    
    # Decision
    final_prediction: str
    confidence: float
    explanation: str
    heatmaps: List
    
    # Routing
    next_action: str  # 'ACCEPT', 'DOMAIN', 'HUMAN'
```

---

## Task 4.3: Implement Agent Nodes

Create `src/agent/nodes.py`:

```python
# ============================================
# PERSON 4: AGENT NODES
# ============================================

import cv2
import torch
import torch.nn.functional as F
import numpy as np
import os
import uuid
from facenet_pytorch import MTCNN
from torchvision import transforms
from pytorch_grad_cam import GradCAM

from .state import AgentState

# ============================================
# NODE 1: INGEST
# ============================================

def ingest_node(state: AgentState) -> AgentState:
    """Validate input video"""
    video_path = state['video_path']
    
    if not os.path.exists(video_path):
        raise ValueError(f'Video not found: {video_path}')
    
    valid_formats = ['.mp4', '.avi', '.mov', '.mkv']
    if not any(video_path.endswith(fmt) for fmt in valid_formats):
        raise ValueError('Invalid video format')
    
    file_size = os.path.getsize(video_path) / (1024 * 1024)
    if file_size > 500:
        raise ValueError('Video too large (max 500MB)')
    
    state['request_id'] = str(uuid.uuid4())
    print(f'[INGEST] Request {state["request_id"]}: {video_path}')
    
    return state

# ============================================
# NODE 2: METADATA
# ============================================

def metadata_node(state: AgentState) -> AgentState:
    """Extract video metadata"""
    video_path = state['video_path']
    cap = cv2.VideoCapture(video_path)
    
    state['fps'] = cap.get(cv2.CAP_PROP_FPS)
    state['resolution'] = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
    
    # Estimate bitrate
    file_size = os.path.getsize(video_path)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / state['fps']
    state['bitrate'] = int((file_size * 8) / duration) if duration > 0 else 0
    
    # Calculate average brightness
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    brightness_samples = []
    
    for i in range(0, total_frames, max(1, total_frames // 10)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness_samples.append(np.mean(gray))
    
    state['avg_brightness'] = np.mean(brightness_samples) if brightness_samples else 128
    
    state['metadata'] = {
        'fps': state['fps'],
        'resolution': state['resolution'],
        'bitrate': state['bitrate'],
        'avg_brightness': state['avg_brightness']
    }
    
    print(f'[METADATA] FPS: {state["fps"]:.1f}, Bitrate: {state["bitrate"]}, '
          f'Brightness: {state["avg_brightness"]:.1f}')
    
    cap.release()
    return state

# ============================================
# NODE 3: PREPROCESS
# ============================================

def preprocess_node(state: AgentState) -> AgentState:
    """Extract faces from video"""
    video_path = state['video_path']
    cap = cv2.VideoCapture(video_path)
    
    detector = MTCNN(keep_all=False, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_rate = max(1, int(state['fps']))  # 1 frame per second
    
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
        
        if len(faces) >= 10:
            break
    
    cap.release()
    
    state['faces_detected'] = len(faces)
    
    if len(faces) == 0:
        raise ValueError('No faces detected in video')
    
    # Normalize faces
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    state['frames'] = [transform(face) for face in faces]
    print(f'[PREPROCESS] Extracted {len(faces)} face frames')
    
    return state

# ============================================
# NODE 4: STUDENT INFERENCE
# ============================================

def student_inference_node(state: AgentState) -> AgentState:
    """Run baseline model inference"""
    frames = torch.stack(state['frames'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load baseline model
    from src.models.baseline import BGModel
    model = BGModel(num_classes=2)
    checkpoint = torch.load('models/bg_model_student.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        frames = frames.to(device)
        logits = model(frames.unsqueeze(0))  # Add batch dimension
        probs = F.softmax(logits, dim=1)
        prediction = probs[0, 1].item()  # Probability of fake
    
    # Calculate confidence
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    state['student_prediction'] = prediction
    state['student_confidence'] = confidence
    
    print(f'[STUDENT] Prediction: {prediction:.3f}, Confidence: {confidence:.3f}')
    
    return state

# ============================================
# NODE 5: POLICY DECISION
# ============================================

def policy_decision_node(state: AgentState) -> AgentState:
    """Decide routing based on confidence"""
    confidence = state['student_confidence']
    
    if confidence >= 0.85:
        state['next_action'] = 'ACCEPT'
        print('[POLICY] High confidence -> ACCEPT')
    elif 0.40 <= confidence < 0.85:
        state['next_action'] = 'DOMAIN'
        print('[POLICY] Medium confidence -> DOMAIN SPECIALIST')
    else:
        state['next_action'] = 'HUMAN'
        print('[POLICY] Low confidence -> HUMAN REVIEW')
    
    return state

def route_decision(state: AgentState) -> str:
    """Return routing decision"""
    return state['next_action']

# ============================================
# NODE 6: DOMAIN INFERENCE
# ============================================

def select_specialist(metadata: dict) -> str:
    """Select appropriate specialist based on metadata"""
    bitrate = metadata.get('bitrate', 1000000)
    brightness = metadata.get('avg_brightness', 128)
    
    if bitrate < 500000:  # Low bitrate = compressed
        return 'cm_model'
    elif brightness < 50:  # Dark video
        return 'll_model'
    else:
        return 'tm_model'  # Default to temporal

def domain_inference_node(state: AgentState) -> AgentState:
    """Run specialist model inference"""
    specialist = select_specialist(state['metadata'])
    state['selected_specialist'] = specialist
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load specialist model
    model_path = f'models/{specialist}_student.pt'
    
    # Use same architecture as baseline
    from src.models.baseline import BGModel
    model = BGModel(num_classes=2)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f'[WARNING] Specialist model not found: {model_path}, using baseline')
        checkpoint = torch.load('models/bg_model_student.pt', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    
    frames = torch.stack(state['frames']).to(device)
    
    with torch.no_grad():
        logits = model(frames.unsqueeze(0))
        probs = F.softmax(logits, dim=1)
        prediction = probs[0, 1].item()
    
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    state['specialist_prediction'] = prediction
    state['specialist_confidence'] = confidence
    
    print(f'[DOMAIN] {specialist}: Prediction: {prediction:.3f}, Confidence: {confidence:.3f}')
    
    return state

# ============================================
# NODE 7: HUMAN REVIEW (Placeholder)
# ============================================

def human_review_node(state: AgentState) -> AgentState:
    """Placeholder for human review escalation"""
    print('[HUMAN] Escalated to human review')
    
    # In production, this would:
    # 1. Save video to review queue
    # 2. Notify human reviewers
    # 3. Wait for human decision
    
    # For now, use student prediction
    state['specialist_prediction'] = state['student_prediction']
    state['specialist_confidence'] = state['student_confidence']
    state['selected_specialist'] = 'human_review'
    
    return state

# ============================================
# NODE 8: EXPLANATION
# ============================================

def explanation_node(state: AgentState) -> AgentState:
    """Generate final prediction and explanation"""
    
    # Determine final prediction
    if state['next_action'] == 'ACCEPT':
        final_pred = state['student_prediction']
        confidence = state['student_confidence']
    else:
        final_pred = state['specialist_prediction']
        confidence = state['specialist_confidence']
    
    state['final_prediction'] = 'FAKE' if final_pred > 0.5 else 'REAL'
    state['confidence'] = confidence
    
    # Generate Grad-CAM heatmaps (simplified)
    state['heatmaps'] = []  # Would generate actual heatmaps in production
    
    # Generate explanation
    pred = state['final_prediction']
    conf = confidence * 100
    specialist = state.get('selected_specialist', 'baseline')
    
    if pred == 'FAKE':
        state['explanation'] = (
            f"This video is classified as FAKE with {conf:.1f}% confidence. "
            f"Analysis by {specialist} detected inconsistencies in facial regions."
        )
    else:
        state['explanation'] = (
            f"This video appears REAL with {conf:.1f}% confidence. "
            f"No significant artifacts or inconsistencies were detected."
        )
    
    print(f'[EXPLAIN] Final: {pred}, Confidence: {conf:.1f}%')
    
    return state
```

---


## Task 4.4: Build LangGraph Agent

Create `src/agent/graph.py`:

```python
# ============================================
# PERSON 4: LANGGRAPH AGENT
# ============================================

from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import (
    ingest_node,
    metadata_node,
    preprocess_node,
    student_inference_node,
    policy_decision_node,
    domain_inference_node,
    human_review_node,
    explanation_node,
    route_decision
)

def build_agent():
    """Build the LangGraph agent"""
    
    # Create graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node('ingest', ingest_node)
    workflow.add_node('metadata', metadata_node)
    workflow.add_node('preprocess', preprocess_node)
    workflow.add_node('student', student_inference_node)
    workflow.add_node('policy', policy_decision_node)
    workflow.add_node('domain', domain_inference_node)
    workflow.add_node('human_review', human_review_node)
    workflow.add_node('explain', explanation_node)
    
    # Define linear flow
    workflow.set_entry_point('ingest')
    workflow.add_edge('ingest', 'metadata')
    workflow.add_edge('metadata', 'preprocess')
    workflow.add_edge('preprocess', 'student')
    workflow.add_edge('student', 'policy')
    
    # Conditional routing from policy
    workflow.add_conditional_edges(
        'policy',
        route_decision,
        {
            'ACCEPT': 'explain',
            'DOMAIN': 'domain',
            'HUMAN': 'human_review'
        }
    )
    
    # Convergence paths
    workflow.add_edge('domain', 'explain')
    workflow.add_edge('human_review', 'explain')
    workflow.add_edge('explain', END)
    
    # Compile
    app = workflow.compile()
    
    return app

# Global agent instance
agent = build_agent()

def run_inference_pipeline(video_path: str) -> dict:
    """Run the complete inference pipeline"""
    try:
        # Initialize state
        initial_state = {
            'video_path': video_path,
            'request_id': '',
            'metadata': {},
            'bitrate': 0,
            'fps': 0.0,
            'resolution': (0, 0),
            'avg_brightness': 0.0,
            'frames': [],
            'faces_detected': 0,
            'student_prediction': 0.0,
            'student_confidence': 0.0,
            'specialist_prediction': 0.0,
            'specialist_confidence': 0.0,
            'selected_specialist': '',
            'final_prediction': '',
            'confidence': 0.0,
            'explanation': '',
            'heatmaps': [],
            'next_action': ''
        }
        
        # Run agent
        result = agent.invoke(initial_state)
        
        return {
            'success': True,
            'prediction': result['final_prediction'],
            'confidence': result['confidence'],
            'explanation': result['explanation'],
            'specialist_used': result.get('selected_specialist', 'baseline'),
            'request_id': result['request_id']
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'request_id': 'unknown'
        }
```

---

## Task 4.5: Train AV-Model (Audio-Visual Specialist)

```python
# ============================================
# PERSON 4: AV-MODEL (AUDIO-VISUAL SPECIALIST)
# ============================================

import torch
import torch.nn as nn
import torchvision.models as models
import torchaudio

class AVModel(nn.Module):
    """Audio-Visual Specialist for lip-sync detection"""
    def __init__(self, num_classes=2):
        super(AVModel, self).__init__()
        
        # Visual branch (ResNet18)
        self.visual_backbone = models.resnet18(weights='IMAGENET1K_V1')
        self.visual_backbone.fc = nn.Identity()
        
        # Audio branch (simple CNN on mel spectrogram)
        self.audio_backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(512 + 256, 256),  # visual + audio
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, video_frames, audio_mel=None):
        # Visual features
        if len(video_frames.shape) == 5:
            batch_size, num_frames = video_frames.shape[:2]
            video_frames = video_frames.view(-1, *video_frames.shape[2:])
            visual_feat = self.visual_backbone(video_frames)
            visual_feat = visual_feat.view(batch_size, num_frames, -1).mean(dim=1)
        else:
            visual_feat = self.visual_backbone(video_frames)
        
        # Audio features (if available)
        if audio_mel is not None:
            audio_feat = self.audio_backbone(audio_mel)
        else:
            audio_feat = torch.zeros(visual_feat.shape[0], 256, device=visual_feat.device)
        
        # Fusion
        combined = torch.cat([visual_feat, audio_feat], dim=1)
        output = self.fusion(combined)
        
        return output

def train_av_model():
    # Training code similar to other models
    # Save as: av_model_student.pt
    pass
```

### After Training:
1. Download `av_model_student.pt`
2. Send to Person 1 (Pranay)
3. Create Pull Request with agent code

---

# 📋 WORKFLOW SUMMARY

## Step 1: Everyone Sets Up (Day 1)
- [ ] Fork repository
- [ ] Clone to local machine
- [ ] Create personal branch
- [ ] Set up Kaggle account

## Step 2: Parallel Training (Days 2-5)
| Person | Models to Train | Dataset |
|--------|----------------|---------|
| Person 1 | BG-Model | Original DFDC |
| Person 2 | CM-Model, RR-Model | Compressed + Re-recorded |
| Person 3 | LL-Model, TM-Model | Low-light + Original |
| Person 4 | AV-Model | Original with audio |

## Step 3: Code Development (Days 2-5)
| Person | Code to Write |
|--------|--------------|
| Person 1 | Web backend, frontend |
| Person 2 | Dataset creation scripts |
| Person 3 | Dataset creation scripts |
| Person 4 | LangGraph agent, nodes |

## Step 4: Integration (Days 6-7)
- Person 1 collects all model weights
- Person 1 integrates agent code
- Person 1 tests complete system
- Everyone reviews and fixes bugs

## Step 5: Deployment (Day 8)
- Docker containerization
- Final testing
- Documentation
- Submission

---

# 🔄 HOW TO SUBMIT YOUR WORK

## For Each Team Member:

### 1. Commit Your Changes
```bash
git add .
git commit -m "Person X: Added [model/feature] - [description]"
```

### 2. Push to Your Fork
```bash
git push origin your-branch-name
```

### 3. Create Pull Request
1. Go to your fork on GitHub
2. Click "Pull Request"
3. Select base: `Pranay22077/deepfake-agentic` branch: `main`
4. Add description of what you did
5. Submit PR

### 4. Share Model Weights
- Upload `.pt` files to Google Drive
- Share link with Person 1 (Pranay)
- Or attach to Pull Request if small enough

---

# 📞 COMMUNICATION

## Daily Standup Questions:
1. What did you complete yesterday?
2. What will you work on today?
3. Any blockers?

## Shared Resources:
- **Dataset**: DFDC on Kaggle (everyone uses same)
- **Code**: GitHub repository
- **Models**: Google Drive shared folder
- **Communication**: WhatsApp/Discord group

---

# ✅ CHECKLIST

## Person 1 (Pranay):
- [ ] Train BG-Model
- [ ] Build backend API
- [ ] Build frontend
- [ ] Integrate all models
- [ ] Deploy with Docker

## Person 2:
- [ ] Create compression dataset
- [ ] Train CM-Model
- [ ] Create re-recording dataset
- [ ] Train RR-Model
- [ ] Submit PR

## Person 3:
- [ ] Create low-light dataset
- [ ] Train LL-Model
- [ ] Train TM-Model
- [ ] Submit PR

## Person 4:
- [ ] Implement agent state
- [ ] Implement agent nodes
- [ ] Build LangGraph agent
- [ ] Train AV-Model
- [ ] Submit PR

---

**🎯 GOAL: Build a complete agentic deepfake detection system with multiple specialized models, intelligent routing, and explainable results!**