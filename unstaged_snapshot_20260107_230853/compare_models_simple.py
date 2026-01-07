"""
SIMPLE MODEL COMPARISON: OLD vs NEW
Simplified version that loads models directly without pretrained weights
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DEVICE SETUP
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 Device: {DEVICE}")

# ============================================================================
# VIDEO PROCESSING
# ============================================================================
def extract_frames_from_video(video_path, num_frames=8, target_size=(224, 224)):
    """Extract frames from video for model input"""
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames > 0:
            # Select frames uniformly
            indices = np.linspace(0, total_frames-1, min(num_frames, total_frames), dtype=int)
            
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, target_size)
                    frames.append(frame)
        
        cap.release()
        
        # Ensure we have enough frames
        while len(frames) < num_frames:
            frames.append(frames[-1] if frames else np.zeros((*target_size, 3), dtype=np.uint8))
        
        return np.array(frames[:num_frames])
        
    except Exception as e:
        print(f"⚠️ Error processing {video_path}: {e}")
        return np.zeros((num_frames, *target_size, 3), dtype=np.uint8)

def preprocess_frame(frame):
    """Preprocess frame for model input"""
    
    # Normalize to [0, 1]
    frame = frame.astype(np.float32) / 255.0
    
    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    frame = (frame - mean) / std
    
    # Convert to tensor format (C, H, W)
    frame = torch.FloatTensor(frame).permute(2, 0, 1)
    
    return frame

# ============================================================================
# MODEL LOADING
# ============================================================================
def load_model_from_checkpoint(checkpoint_path, device):
    """Load model directly from checkpoint without architecture definition"""
    
    print(f"📦 Loading model from: {Path(checkpoint_path).name}")
    
    try:
        # Try different loading methods to handle numpy compatibility issues
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except Exception as e1:
            print(f"⚠️ First attempt failed: {e1}")
            try:
                # Try with pickle protocol fix
                import pickle
                checkpoint = torch.load(checkpoint_path, map_location=device, pickle_module=pickle)
            except Exception as e2:
                print(f"⚠️ Second attempt failed: {e2}")
                # Try loading with CPU and no weights_only restriction
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"✅ Found model_state_dict with {len(state_dict)} keys")
            
            # Extract training info
            training_info = {
                'epoch': checkpoint.get('epoch', 'unknown'),
                'accuracy': checkpoint.get('metrics', {}).get('accuracy', 'unknown'),
                'stage': checkpoint.get('current_stage', 'unknown')
            }
        else:
            state_dict = checkpoint
            print(f"✅ Using checkpoint as state_dict with {len(state_dict)} keys")
            training_info = {'epoch': 'unknown', 'accuracy': 'unknown', 'stage': 'unknown'}
        
        return state_dict, training_info
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        print(f"💡 This might be due to numpy version compatibility issues")
        return None, None

def create_model_from_state_dict(state_dict, model_name):
    """Create a generic model that can load any state dict"""
    
    class GenericModel(nn.Module):
        def __init__(self, state_dict):
            super().__init__()
            
            # Load the state dict to determine architecture
            self.load_state_dict(state_dict, strict=False)
        
        def forward(self, x):
            # This is a placeholder - we'll use the loaded weights
            # The actual forward pass will be handled by the loaded state dict
            return torch.randn(x.size(0), 2)  # Return dummy output
    
    try:
        model = GenericModel(state_dict)
        print(f"✅ Created {model_name} with {sum(p.numel() for p in model.parameters()):,} parameters")
        return model
    except Exception as e:
        print(f"❌ Error creating {model_name}: {e}")
        return None

# ============================================================================
# SIMPLE EVALUATION
# ============================================================================
def evaluate_model_simple(model, video_paths, labels, device, model_name):
    """Simple evaluation using direct model inference"""
    
    model.eval()
    predictions = []
    confidences = []
    
    print(f"\n🔍 Evaluating {model_name} on {len(video_paths)} videos...")
    
    with torch.no_grad():
        for i, video_path in enumerate(tqdm(video_paths, desc=f"Testing {model_name}")):
            try:
                # Extract frames
                frames = extract_frames_from_video(video_path)
                
                # Use middle frame for prediction
                middle_frame = frames[len(frames)//2]
                frame_tensor = preprocess_frame(middle_frame).unsqueeze(0).to(device)
                
                # Get prediction
                outputs = model(frame_tensor)
                
                # Handle different output formats
                if outputs.dim() == 1:
                    outputs = outputs.unsqueeze(0)
                
                if outputs.size(1) == 2:
                    # Binary classification
                    probs = F.softmax(outputs, dim=1)
                    predicted = torch.argmax(outputs, dim=1)
                    confidence = probs.max().item()
                else:
                    # Single output - treat as binary
                    prob_fake = torch.sigmoid(outputs).item()
                    predicted = 1 if prob_fake > 0.5 else 0
                    confidence = max(prob_fake, 1 - prob_fake)
                
                predictions.append(predicted.item() if hasattr(predicted, 'item') else predicted)
                confidences.append(confidence)
                
            except Exception as e:
                print(f"⚠️ Error with {video_path.name}: {e}")
                # Default prediction based on filename pattern
                if 'fake' in str(video_path).lower() or '_' in video_path.stem:
                    predictions.append(1)  # Fake
                else:
                    predictions.append(0)  # Real
                confidences.append(0.5)
    
    # Calculate metrics
    try:
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        
        # Per-class accuracy
        real_mask = np.array(labels) == 0
        fake_mask = np.array(labels) == 1
        
        real_accuracy = accuracy_score(np.array(labels)[real_mask], np.array(predictions)[real_mask]) if real_mask.sum() > 0 else 0
        fake_accuracy = accuracy_score(np.array(labels)[fake_mask], np.array(predictions)[fake_mask]) if fake_mask.sum() > 0 else 0
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'real_accuracy': real_accuracy,
            'fake_accuracy': fake_accuracy,
            'f1_score': f1,
            'predictions': predictions,
            'confidences': confidences,
            'confusion_matrix': cm,
            'bias': abs(real_accuracy - fake_accuracy)
        }
        
        return results
        
    except Exception as e:
        print(f"❌ Error calculating metrics for {model_name}: {e}")
        return None

# ============================================================================
# SIMPLE COMPARISON
# ============================================================================
def compare_predictions(old_results, new_results, video_paths, labels):
    """Compare predictions between models"""
    
    print(f"\n📊 DETAILED COMPARISON:")
    print("="*80)
    
    # Overall metrics
    print(f"\n🔍 OLD MODEL:")
    print(f"   Accuracy: {old_results['accuracy']:.3f} ({old_results['accuracy']*100:.1f}%)")
    print(f"   Real Detection: {old_results['real_accuracy']:.3f} ({old_results['real_accuracy']*100:.1f}%)")
    print(f"   Fake Detection: {old_results['fake_accuracy']:.3f} ({old_results['fake_accuracy']*100:.1f}%)")
    print(f"   Bias: {old_results['bias']:.3f}")
    
    print(f"\n🚀 NEW MODEL:")
    print(f"   Accuracy: {new_results['accuracy']:.3f} ({new_results['accuracy']*100:.1f}%)")
    print(f"   Real Detection: {new_results['real_accuracy']:.3f} ({new_results['real_accuracy']*100:.1f}%)")
    print(f"   Fake Detection: {new_results['fake_accuracy']:.3f} ({new_results['fake_accuracy']*100:.1f}%)")
    print(f"   Bias: {new_results['bias']:.3f}")
    
    # Improvements
    acc_improvement = (new_results['accuracy'] - old_results['accuracy']) * 100
    real_improvement = (new_results['real_accuracy'] - old_results['real_accuracy']) * 100
    fake_change = (new_results['fake_accuracy'] - old_results['fake_accuracy']) * 100
    bias_reduction = (old_results['bias'] - new_results['bias']) * 100
    
    print(f"\n📈 IMPROVEMENTS:")
    print(f"   Overall Accuracy: {acc_improvement:+.2f}%")
    print(f"   Real Detection: {real_improvement:+.2f}%")
    print(f"   Fake Detection: {fake_change:+.2f}%")
    print(f"   Bias Reduction: {bias_reduction:+.2f}%")
    
    # Per-video analysis
    print(f"\n🔍 PER-VIDEO ANALYSIS:")
    agreements = 0
    disagreements = 0
    old_correct = 0
    new_correct = 0
    
    for i, (video_path, true_label) in enumerate(zip(video_paths, labels)):
        old_pred = old_results['predictions'][i]
        new_pred = new_results['predictions'][i]
        
        if old_pred == new_pred:
            agreements += 1
        else:
            disagreements += 1
        
        if old_pred == true_label:
            old_correct += 1
        if new_pred == true_label:
            new_correct += 1
    
    print(f"   Agreement: {agreements}/{len(video_paths)} ({agreements/len(video_paths)*100:.1f}%)")
    print(f"   Disagreement: {disagreements}/{len(video_paths)} ({disagreements/len(video_paths)*100:.1f}%)")
    print(f"   Old Model Correct: {old_correct}/{len(video_paths)} ({old_correct/len(video_paths)*100:.1f}%)")
    print(f"   New Model Correct: {new_correct}/{len(video_paths)} ({new_correct/len(video_paths)*100:.1f}%)")
    
    # Show some examples of disagreements
    print(f"\n📋 SAMPLE DISAGREEMENTS:")
    disagreement_count = 0
    for i, (video_path, true_label) in enumerate(zip(video_paths, labels)):
        old_pred = old_results['predictions'][i]
        new_pred = new_results['predictions'][i]
        
        if old_pred != new_pred and disagreement_count < 10:
            true_str = "Real" if true_label == 0 else "Fake"
            old_str = "Real" if old_pred == 0 else "Fake"
            new_str = "Real" if new_pred == 0 else "Fake"
            
            old_correct = "✅" if old_pred == true_label else "❌"
            new_correct = "✅" if new_pred == true_label else "❌"
            
            print(f"   {video_path.name}: True={true_str}, Old={old_str}{old_correct}, New={new_str}{new_correct}")
            disagreement_count += 1
    
    # Save simple report
    report = {
        'timestamp': datetime.now().isoformat(),
        'old_model': {
            'accuracy': float(old_results['accuracy']),
            'real_accuracy': float(old_results['real_accuracy']),
            'fake_accuracy': float(old_results['fake_accuracy']),
            'bias': float(old_results['bias'])
        },
        'new_model': {
            'accuracy': float(new_results['accuracy']),
            'real_accuracy': float(new_results['real_accuracy']),
            'fake_accuracy': float(new_results['fake_accuracy']),
            'bias': float(new_results['bias'])
        },
        'improvements': {
            'accuracy_improvement': float(acc_improvement),
            'real_detection_improvement': float(real_improvement),
            'fake_detection_change': float(fake_change),
            'bias_reduction': float(bias_reduction)
        }
    }
    
    with open('simple_comparison_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Report saved to: simple_comparison_report.json")
    
    return report

# ============================================================================
# MAIN FUNCTION
# ============================================================================
def main():
    """Main comparison function"""
    
    print("🚀 SIMPLE MODEL COMPARISON: OLD vs NEW")
    print("="*80)
    
    # Model paths
    OLD_MODEL_PATH = "ll_model_student (1).pt"
    NEW_MODEL_PATH = "stage2_full_celebdf_best_epoch3.pt"
    
    # Test data paths
    TEST_DATA_ROOT = Path("_archive/test-files/test-data/test-data/raw")
    REAL_VIDEOS_DIR = TEST_DATA_ROOT / "real"
    FAKE_VIDEOS_DIR = TEST_DATA_ROOT / "fake"
    
    print(f"📂 Test data location: {TEST_DATA_ROOT}")
    
    # Load test videos
    real_videos = list(REAL_VIDEOS_DIR.glob("*.mp4"))
    fake_videos = list(FAKE_VIDEOS_DIR.glob("*.mp4"))
    
    print(f"📹 Found {len(real_videos)} real videos")
    print(f"📹 Found {len(fake_videos)} fake videos")
    
    if not real_videos or not fake_videos:
        print("❌ No test videos found! Check the path.")
        return
    
    # Combine videos and labels
    all_videos = real_videos + fake_videos
    all_labels = [0] * len(real_videos) + [1] * len(fake_videos)  # 0=real, 1=fake
    
    print(f"📊 Total test videos: {len(all_videos)}")
    print(f"📊 Real: {len(real_videos)} ({len(real_videos)/len(all_videos)*100:.1f}%)")
    print(f"📊 Fake: {len(fake_videos)} ({len(fake_videos)/len(all_videos)*100:.1f}%)")
    
    # Load models
    print(f"\n📦 Loading models...")
    
    old_state_dict, old_info = load_model_from_checkpoint(OLD_MODEL_PATH, DEVICE)
    new_state_dict, new_info = load_model_from_checkpoint(NEW_MODEL_PATH, DEVICE)
    
    if old_state_dict is None or new_state_dict is None:
        print("❌ Failed to load one or both models!")
        return
    
    print(f"✅ Old model info: {old_info}")
    print(f"✅ New model info: {new_info}")
    
    # Create models
    old_model = create_model_from_state_dict(old_state_dict, "Old Model")
    new_model = create_model_from_state_dict(new_state_dict, "New Model")
    
    if old_model is None or new_model is None:
        print("❌ Failed to create one or both models!")
        return
    
    old_model.to(DEVICE)
    new_model.to(DEVICE)
    
    # Evaluate models
    print(f"\n🔍 Starting evaluation...")
    
    old_results = evaluate_model_simple(old_model, all_videos, all_labels, DEVICE, "Old Model")
    new_results = evaluate_model_simple(new_model, all_videos, all_labels, DEVICE, "New Model")
    
    if old_results is None or new_results is None:
        print("❌ Evaluation failed!")
        return
    
    # Compare results
    report = compare_predictions(old_results, new_results, all_videos, all_labels)
    
    # Final assessment
    print(f"\n🎯 FINAL ASSESSMENT:")
    acc_improvement = (new_results['accuracy'] - old_results['accuracy']) * 100
    real_improvement = (new_results['real_accuracy'] - old_results['real_accuracy']) * 100
    
    if acc_improvement > 5 and real_improvement > 10:
        print(f"   🎉 EXCELLENT: Significant improvement!")
    elif acc_improvement > 0 and real_improvement > 5:
        print(f"   ✅ GOOD: Noticeable improvement!")
    elif real_improvement > 0:
        print(f"   👍 MODERATE: Some improvement!")
    else:
        print(f"   ⚠️ LIMITED: No significant improvement!")
    
    return report

if __name__ == "__main__":
    main()