#!/usr/bin/env python3
"""
Model Pruning Script
Implements structured pruning to reduce model size and latency
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import argparse
import json
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.student import create_student_model

class StructuredPruner:
    """Structured pruning implementation"""
    
    def __init__(self, model):
        self.model = model
        self.original_params = sum(p.numel() for p in model.parameters())
    
    def prune_conv_layers(self, prune_ratio=0.2):
        """Prune convolutional layers using structured pruning"""
        pruned_layers = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                # L2 norm based channel pruning
                prune.ln_structured(
                    module, 
                    name='weight', 
                    amount=prune_ratio, 
                    n=2, 
                    dim=0  # Prune output channels
                )
                pruned_layers.append(name)
                print(f"Pruned {name}: {prune_ratio*100:.1f}% of channels")
        
        return pruned_layers
    
    def prune_linear_layers(self, prune_ratio=0.3):
        """Prune linear layers using unstructured pruning"""
        pruned_layers = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Magnitude-based unstructured pruning
                prune.l1_unstructured(
                    module,
                    name='weight',
                    amount=prune_ratio
                )
                pruned_layers.append(name)
                print(f"Pruned {name}: {prune_ratio*100:.1f}% of weights")
        
        return pruned_layers
    
    def remove_pruning_reparameterization(self):
        """Remove pruning reparameterization to make pruning permanent"""
        for module in self.model.modules():
            try:
                prune.remove(module, 'weight')
            except:
                pass
            try:
                prune.remove(module, 'bias')
            except:
                pass
    
    def calculate_sparsity(self):
        """Calculate overall model sparsity"""
        total_params = 0
        zero_params = 0
        
        for param in self.model.parameters():
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
        
        sparsity = zero_params / total_params
        return sparsity
    
    def get_model_size(self):
        """Get model size in MB"""
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        size_mb = (param_size + buffer_size) / (1024 * 1024)
        return size_mb

def magnitude_based_pruning(model, target_sparsity=0.5):
    """Global magnitude-based pruning"""
    parameters_to_prune = []
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))
    
    # Global magnitude pruning
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=target_sparsity
    )
    
    return parameters_to_prune

def channel_pruning(model, prune_ratio=0.2):
    """Channel-wise structured pruning for conv layers"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.out_channels > 1:
            # Calculate channel importance (L2 norm of filters)
            weight = module.weight.data
            channel_norms = torch.norm(weight.view(weight.size(0), -1), dim=1)
            
            # Determine channels to prune
            num_channels_to_prune = int(module.out_channels * prune_ratio)
            if num_channels_to_prune > 0:
                _, indices_to_prune = torch.topk(
                    channel_norms, 
                    num_channels_to_prune, 
                    largest=False
                )
                
                # Create mask
                mask = torch.ones_like(channel_norms)
                mask[indices_to_prune] = 0
                
                # Apply mask to weights
                module.weight.data *= mask.view(-1, 1, 1, 1)
                
                if module.bias is not None:
                    module.bias.data *= mask
                
                print(f"Pruned {num_channels_to_prune} channels from {name}")

def main():
    parser = argparse.ArgumentParser(description='Prune Student Model')
    parser.add_argument('--model', required=True, help='Path to model checkpoint')
    parser.add_argument('--out', required=True, help='Output path for pruned model')
    parser.add_argument('--method', choices=['magnitude', 'structured', 'channel'], 
                       default='structured', help='Pruning method')
    parser.add_argument('--ratio', type=float, default=0.3, help='Pruning ratio')
    parser.add_argument('--target_sparsity', type=float, default=0.5, 
                       help='Target sparsity for magnitude pruning')
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.model}")
    
    # Load model
    model = create_student_model()
    checkpoint = torch.load(args.model, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Initialize pruner
    pruner = StructuredPruner(model)
    
    print(f"Original model size: {pruner.get_model_size():.2f} MB")
    print(f"Original parameters: {pruner.original_params:,}")
    
    # Apply pruning based on method
    if args.method == 'magnitude':
        print(f"Applying magnitude-based pruning (sparsity: {args.target_sparsity})")
        magnitude_based_pruning(model, args.target_sparsity)
        
    elif args.method == 'structured':
        print(f"Applying structured pruning (ratio: {args.ratio})")
        pruner.prune_conv_layers(args.ratio)
        pruner.prune_linear_layers(args.ratio)
        
    elif args.method == 'channel':
        print(f"Applying channel pruning (ratio: {args.ratio})")
        channel_pruning(model, args.ratio)
    
    # Calculate results
    sparsity = pruner.calculate_sparsity()
    new_size = pruner.get_model_size()
    remaining_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nPruning Results:")
    print(f"Sparsity: {sparsity:.3f}")
    print(f"New model size: {new_size:.2f} MB")
    print(f"Size reduction: {(1 - new_size/pruner.get_model_size())*100:.1f}%")
    print(f"Remaining parameters: {remaining_params:,}")
    
    # Remove pruning reparameterization for permanent pruning
    if args.method in ['magnitude', 'structured']:
        pruner.remove_pruning_reparameterization()
        print("Removed pruning reparameterization")
    
    # Save pruned model
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    # Update checkpoint with pruning info
    checkpoint['pruning_info'] = {
        'method': args.method,
        'ratio': args.ratio,
        'target_sparsity': args.target_sparsity,
        'final_sparsity': sparsity,
        'original_size_mb': pruner.get_model_size(),
        'pruned_size_mb': new_size,
        'original_params': pruner.original_params,
        'remaining_params': remaining_params
    }
    
    torch.save(checkpoint, args.out)
    
    # Save pruning report
    report_path = args.out.replace('.pt', '_pruning_report.json')
    with open(report_path, 'w') as f:
        json.dump(checkpoint['pruning_info'], f, indent=2)
    
    print(f"\nPruned model saved to: {args.out}")
    print(f"Pruning report saved to: {report_path}")
    
    # Test model forward pass
    print("\nTesting pruned model...")
    model.eval()
    with torch.no_grad():
        # Test single frame
        dummy_video = torch.randn(1, 3, 224, 224)
        dummy_audio = torch.randn(1, 16000 * 3)
        
        try:
            output = model(dummy_video, dummy_audio)
            print(f"Forward pass successful! Output shape: {output.shape}")
        except Exception as e:
            print(f"Forward pass failed: {e}")

if __name__ == "__main__":
    main()