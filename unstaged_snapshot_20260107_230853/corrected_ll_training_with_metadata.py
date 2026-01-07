"""
CORRECTED LL TRAINING WITH PROPER METADATA USAGE
Uses the actual DFDC metadata.json files to get correct real/fake labels
"""
import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as transforms
import cv2
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import gc
import psutil
from tqdm import tqdm
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# ============================================================================
CHECKPOINT_PATH = "/kaggle/input/ll-model-chunk-0/ll_model_chunk_00.pt"  # CHANGE THIS PATH
DATA_DIR = "/kaggle/input/dfdc-10"
OUTPUT_DIR = "/kaggle/working"

# ============================================================================
# SETTINGS
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 4
NUM_WORKERS = 2
PIN_MEMORY = True
PREFETCH_FACTOR = 2

print(f"🚀 CORRECTED LL TRAINING WITH PROPER METADATA")
print(f"📊 Using actual DFDC labels from metadata.json files")
print(f"🔧 Checkpoint: {CHECKPOINT_PATH}")
print(f"📂 Data: {DATA_DIR}")
print(f"💾 Output: {OUTPUT_DIR}")
print(f"🔥 Device: {DEVICE}")
print("="*80)

# ============================================================================
# METADATA INSPECTOR AND LOADER
# ============================================================================
class DFDCMetadataLoader:
    """Loads and processes DFDC metadata.json files correctly"""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        
    def inspect_chunk_metadata(self, chunk_idx):
        """Inspect metadata structure for a chunk"""
        
        print(f"\n🔍 INSPECTING CHUNK {chunk_idx} METADATA")
        print("-" * 50)
        
        # Find chunk directory
        chunk_path = self._find_chunk_directory(chunk_idx)
        if chunk_path is None:
            print(f"❌ Chunk {chunk_idx} directory not found")
            return None, None
        
        print(f"📂 Chunk path: {chunk_path}")
        
        # Load metadata
        metadata_file = chunk_path / "metadata.json"
        if not metadata_file.exists():
            print(f"❌ No metadata.json in {chunk_path}")
            return None, None
        
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            print(f"📋 Loaded {len(metadata)} metadata entries")
            
            # Show sample entries
            sample_keys = list(metadata.keys())[:3]
            print(f"\n📊 SAMPLE ENTRIES:")
            
            for i, key in enumerate(sample_keys):
                entry = metadata[key]
                print(f"   {i+1}. {key}: {entry}")
            
            # Analyze label structure
            label_info = self._analyze_labels(metadata)
            
            return metadata, label_info
            
        except Exception as e:
            print(f"❌ Error loading metadata: {e}")
            return None, None
    
    def _analyze_labels(self, metadata):
        """Analyze the label structure in metadata"""
        
        print(f"\n🏷️ LABEL ANALYSIS:")
        
        # Check different possible label formats
        label_formats = []
        
        sample_entry = next(iter(metadata.values()))
        print(f"   Sample entry type: {type(sample_entry)}")
        print(f"   Sample entry: {sample_entry}")
        
        if isinstance(sample_entry, dict):
            # Entry is a dictionary - check for label fields
            possible_fields = ['label', 'Label', 'LABEL', 'fake', 'real', 'target']
            
            for field in possible_fields:
                if field in sample_entry:
                    print(f"   ✅ Found label field: '{field}'")
                    
                    # Analyze this field across all entries
                    labels = []
                    for entry in metadata.values():
                        if field in entry:
                            labels.append(entry[field])
                    
                    unique_labels = list(set(labels))
                    print(f"   Unique values: {unique_labels}")
                    
                    # Count distribution
                    label_counts = {}
                    for label in labels:
                        label_counts[label] = label_counts.get(label, 0) + 1
                    
                    print(f"   Distribution:")
                    for label, count in label_counts.items():
                        percentage = count / len(labels) * 100
                        print(f"     - {label}: {count} ({percentage:.1f}%)")
                    
                    return {
                        'field': field,
                        'labels': labels,
                        'unique_values': unique_labels,
                        'distribution': label_counts
                    }
            
            # If no standard field found, show all available fields
            all_fields = set()
            for entry in metadata.values():
                if isinstance(entry, dict):
                    all_fields.update(entry.keys())
            
            print(f"   Available fields: {sorted(all_fields)}")
            
        elif isinstance(sample_entry, str):
            # Entry itself might be the label
            print(f"   Entry is string - might be direct label")
            labels = list(metadata.values())
            unique_labels = list(set(labels))
            print(f"   Unique values: {unique_labels}")
            
            label_counts = {}
            for label in labels:
                label_counts[label] = label_counts.get(label, 0) + 1
            
            print(f"   Distribution:")
            for label, count in label_counts.items():
                percentage = count / len(labels) * 100
                print(f"     - {label}: {count} ({percentage:.1f}%)")
            
            return {
                'field': 'direct',
                'labels': labels,
                'unique_values': unique_labels,
                'distribution': label_counts
            }
        
        return None
    
    def _find_chunk_directory(self, chunk_idx):
        """Find the correct chunk directory"""
        
        chunk_name = f"dfdc_train_part_{chunk_idx:02d}"
        chunk_dir = self.data_dir / chunk_name
        
        if chunk_dir.exists():
            # Check for subdirectory
            subdir_name = f"dfdc_train_part_{chunk_idx}"
            subdir_path = chunk_dir / subdir_name
            
            if subdir_path.exists():
                return subdir_path
            else:
                return chunk_dir
        
        return None
    
    def load_chunk_data(self, chunk_idx):
        """Load properly labeled data from a chunk"""
        
        print(f"\n📥 LOADING CHUNK {chunk_idx} WITH CORRECT LABELS")
        
        # First inspect the metadata
        metadata, label_info = self.inspect_chunk_metadata(chunk_idx)
        
        if metadata is None or label_info is None:
            print(f"❌ Cannot load chunk {chunk_idx} - metadata issues")
            return []
        
        # Find chunk directory and video files
        chunk_path = self._find_chunk_directory(chunk_idx)
        video_files = []
        
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            video_files.extend(list(chunk_path.glob(f"*{ext}")))
        
        print(f"🎬 Found {len(video_files)} video files")
        
        # Create samples with correct labels
        samples = []
        label_field = label_info['field']
        
        real_count = 0
        fake_count = 0
        missing_metadata = 0
        
        for video_file in video_files:
            video_name = video_file.stem
            
            if video_name in metadata:
                # Get label based on the identified structure
                if label_field == 'direct':
                    label_value = metadata[video_name]
                else:
                    label_value = metadata[video_name].get(label_field)
                
                # Convert label to binary (0=real, 1=fake)
                if label_value in ['REAL', 'real', 0, '0']:
                    label = 0  # Real
                    real_count += 1
                elif label_value in ['FAKE', 'fake', 1, '1']:
                    label = 1  # Fake
                    fake_count += 1
                else:
                    print(f"⚠️ Unknown label '{label_value}' for {video_name}")
                    continue
                
                samples.append({
                    'video_path': str(video_file),
                    'label': label,
                    'video_name': video_name,
                    'original_label': label_value
                })
            else:
                missing_metadata += 1
        
        print(f"\n📊 CHUNK {chunk_idx} LABEL DISTRIBUTION:")
        print(f"   ✅ Real videos: {real_count}")
        print(f"   ✅ Fake videos: {fake_count}")
        print(f"   ⚠️ Missing metadata: {missing_metadata}")
        print(f"   📈 Total samples: {len(samples)}")
        
        if real_count > 0 and fake_count > 0:
            balance_ratio = min(real_count, fake_count) / max(real_count, fake_count)
            print(f"   ⚖️ Balance ratio: {balance_ratio:.3f} (1.0 = perfect balance)")
        
        return samples

# ============================================================================
# LOW-LIGHT ANALYSIS MODULE
# ============================================================================
class LowLightAnalysisModule(nn.Module):
    """Module for analyzing low-light conditions and artifacts"""
    
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Luminance analysis
        self.luminance_analyzer = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=1)
        )
        
        # Noise pattern detector
        self.noise_detector = nn.Sequential(
            nn.Conv2d(in_channels, 12, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(12, 24, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(24, 12, kernel_size=1)
        )
        
        # Shadow/highlight inconsistency detector
        self.shadow_detector = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=1)
        )
    
    def forward(self, x):
        # Luminance analysis
        lum_features = self.luminance_analyzer(x)
        lum_features = F.adaptive_avg_pool2d(lum_features, (7, 7))
        
        # Noise analysis
        noise_features = self.noise_detector(x)
        noise_features = F.adaptive_avg_pool2d(noise_features, (7, 7))
        
        # Shadow analysis
        shadow_features = self.shadow_detector(x)
        shadow_features = F.adaptive_avg_pool2d(shadow_features, (7, 7))
        
        # Combine features
        combined = torch.cat([lum_features, noise_features, shadow_features], dim=1)
        return combined

# ============================================================================
# SPECIALIST MODEL
# ============================================================================
class LLSpecialistModel(nn.Module):
    """Low-Light specialist model"""
    
    def __init__(self, num_classes=2):
        super().__init__()
        
        from torchvision.models import efficientnet_b4
        print(f"🔄 Loading EfficientNet-B4 for LL model...")
        
        self.backbone = efficientnet_b4(weights='IMAGENET1K_V1')
        self.backbone.classifier = nn.Identity()
        backbone_features = 1792
        
        # Specialist module
        self.specialist_module = LowLightAnalysisModule()
        specialist_features = 36 * 7 * 7  # (16+12+8) * 7 * 7
        
        # Classifier
        total_features = backbone_features + specialist_features
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(total_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
        print(f"✅ LL model ready!")
    
    def forward(self, x):
        # Extract backbone features
        backbone_features = self.backbone.features(x)
        backbone_features = self.backbone.avgpool(backbone_features)
        backbone_features = torch.flatten(backbone_features, 1)
        
        # Extract specialist features
        specialist_features = self.specialist_module(x)
        specialist_features = torch.flatten(specialist_features, 1)
        
        # Combine features
        combined_features = torch.cat([backbone_features, specialist_features], dim=1)
        
        # Final classification
        output = self.classifier(combined_features)
        return output

# ============================================================================
# CORRECTED DATASET
# ============================================================================
class CorrectedDFDCDataset(Dataset):
    """Dataset using correct metadata labels"""
    
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
        
        print(f"📊 Dataset created with {len(samples)} samples")
        
        # Analyze distribution
        real_count = sum(1 for s in samples if s['label'] == 0)
        fake_count = sum(1 for s in samples if s['label'] == 1)
        
        print(f"   Real: {real_count} ({real_count/len(samples)*100:.1f}%)")
        print(f"   Fake: {fake_count} ({fake_count/len(samples)*100:.1f}%)")
    
    def _extract_frames(self, video_path, num_frames=8):
        """Extract frames from video"""
        
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames > 0:
                indices = np.linspace(0, total_frames-1, min(num_frames, total_frames), dtype=int)
                
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    
                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = self._preprocess_for_lowlight(frame)
                        frame = cv2.resize(frame, (224, 224))
                        frames.append(frame)
            
            cap.release()
            
            # Ensure we have enough frames
            while len(frames) < num_frames:
                frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
            
            return np.array(frames[:num_frames])
            
        except Exception as e:
            print(f"⚠️ Error extracting frames from {video_path}: {e}")
            return np.zeros((num_frames, 224, 224, 3), dtype=np.uint8)
    
    def _preprocess_for_lowlight(self, frame):
        """Enhanced preprocessing for low-light analysis"""
        
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Enhanced CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l_enhanced = clahe.apply(l)
        
        # Reconstruct RGB
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        return enhanced_rgb
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Extract frames
        frames = self._extract_frames(sample['video_path'])
        
        # Use middle frame
        frame = frames[len(frames)//2]
        
        if self.transform:
            frame = self.transform(frame)
        else:
            frame = torch.FloatTensor(frame).permute(2, 0, 1) / 255.0
        
        label = torch.LongTensor([sample['label']])
        
        return frame, label

# ============================================================================
# CORRECTED TRAINER
# ============================================================================
class CorrectedLLTrainer:
    """Trainer using correct metadata labels"""
    
    def __init__(self, data_dir, output_dir, checkpoint_path=None):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"🚀 Initializing CORRECTED LL trainer...")
        
        # Initialize model
        self.model = LLSpecialistModel(num_classes=2)
        self.model.to(DEVICE)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=3)
        
        # Metadata loader
        self.metadata_loader = DFDCMetadataLoader(data_dir)
        
        # Training state
        self.current_chunk = 0
        self.total_chunks = 10
        self.training_history = []
        
        # Load checkpoint if provided
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        
        print(f"✅ CORRECTED LL trainer ready!")
    
    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint"""
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return
        
        print(f"📥 Loading checkpoint: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            
            if 'model_state_dict' in checkpoint:
                # Try to load compatible weights
                old_state_dict = checkpoint['model_state_dict']
                new_state_dict = self.model.state_dict()
                
                transferred = 0
                for name, param in old_state_dict.items():
                    if name in new_state_dict and param.shape == new_state_dict[name].shape:
                        new_state_dict[name] = param
                        transferred += 1
                
                self.model.load_state_dict(new_state_dict)
                print(f"✅ Transferred {transferred} compatible layers")
            
            # Load training state
            self.current_chunk = checkpoint.get('current_chunk', 1)
            self.training_history = checkpoint.get('training_history', [])
            
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            print("🔄 Starting fresh training...")
    
    def train_on_chunk(self, chunk_idx):
        """Train on a specific chunk with correct labels"""
        
        print(f"\n🎯 TRAINING ON CHUNK {chunk_idx} WITH CORRECT LABELS")
        print("="*60)
        
        # Load chunk data with correct labels
        samples = self.metadata_loader.load_chunk_data(chunk_idx)
        
        if len(samples) == 0:
            print(f"⚠️ No valid samples in chunk {chunk_idx}")
            return
        
        # Create dataset and dataloader
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset = CorrectedDFDCDataset(samples, transform)
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY
        )
        
        # Training loop
        self.model.train()
        
        for epoch in range(5):  # 5 epochs per chunk
            print(f"\n📈 Epoch {epoch+1}/5 for chunk {chunk_idx}")
            
            epoch_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            # Class-specific tracking
            real_correct = 0
            real_total = 0
            fake_correct = 0
            fake_total = 0
            
            pbar = tqdm(dataloader, desc=f"Training LL")
            
            for batch_idx, (frames, labels) in enumerate(pbar):
                try:
                    frames = frames.to(DEVICE, non_blocking=True)
                    labels = labels.squeeze().to(DEVICE, non_blocking=True)
                    
                    # Forward pass
                    self.optimizer.zero_grad()
                    outputs = self.model(frames)
                    loss = self.criterion(outputs, labels)
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    
                    # Track metrics
                    epoch_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    correct_predictions += (predicted == labels).sum().item()
                    total_predictions += labels.size(0)
                    
                    # Class-specific accuracy
                    for i in range(len(labels)):
                        true_label = labels[i].item()
                        pred_label = predicted[i].item()
                        
                        if true_label == 0:  # Real
                            real_total += 1
                            if pred_label == true_label:
                                real_correct += 1
                        else:  # Fake
                            fake_total += 1
                            if pred_label == true_label:
                                fake_correct += 1
                    
                    # Update progress
                    current_acc = correct_predictions / total_predictions * 100
                    real_acc = (real_correct / real_total * 100) if real_total > 0 else 0
                    fake_acc = (fake_correct / fake_total * 100) if fake_total > 0 else 0
                    
                    pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{current_acc:.1f}%',
                        'Real': f'{real_acc:.1f}%',
                        'Fake': f'{fake_acc:.1f}%'
                    })
                    
                except Exception as e:
                    print(f"⚠️ Batch error: {e}")
                    continue
            
            # Epoch summary
            avg_loss = epoch_loss / len(dataloader)
            accuracy = correct_predictions / total_predictions
            real_accuracy = real_correct / real_total if real_total > 0 else 0
            fake_accuracy = fake_correct / fake_total if fake_total > 0 else 0
            
            print(f"✅ Epoch {epoch+1} completed:")
            print(f"   📉 Loss: {avg_loss:.4f}")
            print(f"   🎯 Accuracy: {accuracy*100:.2f}%")
            print(f"   🔴 Real: {real_accuracy*100:.2f}% ({real_correct}/{real_total})")
            print(f"   🟡 Fake: {fake_accuracy*100:.2f}% ({fake_correct}/{fake_total})")
            
            self.scheduler.step()
        
        # Save checkpoint
        final_metrics = {
            'accuracy': accuracy,
            'real_accuracy': real_accuracy,
            'fake_accuracy': fake_accuracy,
            'avg_loss': avg_loss,
            'chunk_idx': chunk_idx,
            'samples_processed': len(samples)
        }
        
        self.training_history.append(final_metrics)
        self._save_checkpoint(chunk_idx, final_metrics)
        
        print(f"\n🎉 Chunk {chunk_idx} training completed!")
        print(f"📊 Final metrics: Acc={accuracy*100:.2f}%, Real={real_accuracy*100:.2f}%, Fake={fake_accuracy*100:.2f}%")
    
    def _save_checkpoint(self, chunk_idx, metrics):
        """Save checkpoint"""
        
        checkpoint_path = self.output_dir / f"corrected_ll_model_chunk_{chunk_idx:02d}.pt"
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'current_chunk': chunk_idx,
            'training_history': self.training_history,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'model_type': 'corrected_ll'
        }, checkpoint_path)
        
        print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def continue_training(self):
        """Continue training with correct labels"""
        
        print(f"\n🚀 STARTING CORRECTED LL TRAINING")
        print(f"📊 Using proper DFDC metadata labels")
        print(f"🔧 Starting from chunk {self.current_chunk}")
        print("="*80)
        
        for chunk_idx in range(self.current_chunk, self.total_chunks):
            try:
                start_time = time.time()
                self.train_on_chunk(chunk_idx)
                end_time = time.time()
                
                self.current_chunk = chunk_idx + 1
                
                # Progress summary
                progress = (chunk_idx + 1) / self.total_chunks * 100
                time_taken = end_time - start_time
                
                print(f"\n📈 PROGRESS: {progress:.1f}% ({chunk_idx+1}/10 chunks)")
                print(f"⏱️ Chunk {chunk_idx} took: {time_taken/60:.1f} minutes")
                
            except Exception as e:
                print(f"❌ Error training on chunk {chunk_idx}: {e}")
                continue
        
        # Save final model
        final_model_path = self.output_dir / "corrected_ll_model_final.pt"
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'training_history': self.training_history,
            'model_type': 'corrected_ll',
            'total_chunks_processed': self.current_chunk,
            'final_metrics': self.training_history[-1] if self.training_history else {}
        }, final_model_path)
        
        print(f"\n🎉 CORRECTED LL TRAINING COMPLETED!")
        print(f"💾 Final model saved: {final_model_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main function"""
    
    print("🚀 CORRECTED LL TRAINING WITH PROPER METADATA")
    print("="*80)
    print("📊 This version uses the actual DFDC metadata.json files")
    print("🏷️ Labels are extracted correctly from metadata")
    print("✅ No more random label assignment!")
    print("="*80)
    
    # Initialize trainer
    trainer = CorrectedLLTrainer(DATA_DIR, OUTPUT_DIR, CHECKPOINT_PATH)
    
    # Start training
    trainer.continue_training()
    
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()

# Execute immediately
print("🚀 STARTING CORRECTED TRAINING...")
main()