"""
MINIMAL INTERCEPTOR LL-MODEL
Lightweight version for testing and development
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

class MinimalLLModel(nn.Module):
    """Minimal version of the LL-Model for testing"""
    
    def __init__(self, num_classes=2):
        super().__init__()
        
        # Use smaller backbone
        self.backbone = mobilenet_v3_small(weights='IMAGENET1K_V1')
        self.backbone.classifier = nn.Identity()
        
        # Simple classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(576, 256),  # MobileNetV3-Small features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

# Create and test
if __name__ == "__main__":
    model = MinimalLLModel()
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(f"Minimal model works! Output: {output.shape}")
    
    # Save minimal model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': 'minimal_ll_model',
        'note': 'Lightweight version for testing'
    }, 'minimal_ll_model.pt')
    print("✅ Saved minimal model as minimal_ll_model.pt")
