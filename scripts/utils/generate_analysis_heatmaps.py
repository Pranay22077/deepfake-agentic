#!/usr/bin/env python3
"""
Advanced Heatmap Generation for E-Raksha Interceptor
Generates comprehensive analysis data for the frontend visualization libraries
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List, Any
import cv2
from pathlib import Path

class DeepfakeAnalysisHeatmapGenerator:
    """
    Generates heatmap data for deepfake analysis visualization
    Integrates with ApexCharts, ECharts, Plotly.js, and D3.js
    """
    
    def __init__(self):
        self.models = ['BG', 'AV', 'CM', 'RR', 'LL', 'TM']
        self.features = ['compression', 'lighting', 'temporal', 'artifacts', 'quality', 'audio_sync']
        
    def generate_frame_confidence_heatmap(self, video_path: str = None, num_frames: int = 100) -> Dict[str, Any]:
        """
        Generate frame-by-frame confidence heatmap data
        Returns data compatible with ApexCharts heatmap
        """
        # Simulate frame analysis (in real implementation, this would analyze actual video frames)
        frame_data = []
        
        for model in self.models:
            model_data = {
                "name": f"{model} Model",
                "data": []
            }
            
            for frame_idx in range(num_frames):
                # Simulate confidence calculation with some realistic patterns
                base_confidence = np.random.beta(2, 2)  # Beta distribution for realistic confidence
                
                # Add temporal patterns (some models perform better on certain frame types)
                if model == 'TM':  # Temporal model should show frame-to-frame patterns
                    temporal_factor = 0.1 * np.sin(frame_idx * 0.1) + 0.05 * np.random.randn()
                    base_confidence += temporal_factor
                elif model == 'CM':  # Compression model might show periodic patterns
                    compression_factor = 0.05 * np.sin(frame_idx * 0.05) + 0.03 * np.random.randn()
                    base_confidence += compression_factor
                
                # Clamp to valid range
                confidence = np.clip(base_confidence, 0.0, 1.0)
                
                model_data["data"].append({
                    "x": f"Frame {frame_idx + 1}",
                    "y": float(confidence)
                })
            
            frame_data.append(model_data)
        
        return {
            "type": "apex_heatmap",
            "title": "Frame-by-Frame Confidence Analysis",
            "data": frame_data,
            "metadata": {
                "total_frames": num_frames,
                "models_analyzed": len(self.models),
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def generate_model_performance_boxplot(self, num_samples: int = 1000) -> Dict[str, Any]:
        """
        Generate model performance distribution data for ECharts boxplot
        """
        performance_data = []
        
        for model in self.models:
            # Simulate performance distribution for each model
            if model == 'TM':  # Temporal model - high performance, low variance
                samples = np.random.normal(0.92, 0.03, num_samples)
            elif model == 'CM':  # Compression model - good performance
                samples = np.random.normal(0.88, 0.04, num_samples)
            elif model == 'AV':  # Audio-visual model - moderate performance
                samples = np.random.normal(0.85, 0.05, num_samples)
            elif model == 'RR':  # Resolution/quality model
                samples = np.random.normal(0.83, 0.06, num_samples)
            elif model == 'LL':  # Lighting model - more variable
                samples = np.random.normal(0.80, 0.07, num_samples)
            else:  # BG model - background analysis
                samples = np.random.normal(0.78, 0.08, num_samples)
            
            # Clamp to valid range
            samples = np.clip(samples, 0.0, 1.0)
            
            # Calculate boxplot statistics
            q1 = np.percentile(samples, 25)
            median = np.percentile(samples, 50)
            q3 = np.percentile(samples, 75)
            min_val = np.min(samples)
            max_val = np.max(samples)
            
            performance_data.append([min_val, q1, median, q3, max_val])
        
        return {
            "type": "echarts_boxplot",
            "title": "Model Performance Distribution",
            "data": {
                "boxplotData": performance_data,
                "categories": self.models
            },
            "metadata": {
                "samples_per_model": num_samples,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def generate_feature_correlation_matrix(self) -> Dict[str, Any]:
        """
        Generate feature correlation matrix for Plotly.js heatmap
        """
        # Simulate realistic correlations between deepfake detection features
        correlation_patterns = {
            'compression': {'lighting': 0.23, 'temporal': 0.45, 'artifacts': 0.67, 'quality': 0.34, 'audio_sync': 0.12},
            'lighting': {'temporal': 0.56, 'artifacts': 0.78, 'quality': 0.45, 'audio_sync': 0.23},
            'temporal': {'artifacts': 0.34, 'quality': 0.67, 'audio_sync': 0.45},
            'artifacts': {'quality': 0.56, 'audio_sync': 0.34},
            'quality': {'audio_sync': 0.78}
        }
        
        # Build full correlation matrix
        n_features = len(self.features)
        correlation_matrix = np.eye(n_features)  # Start with identity matrix
        
        for i, feature1 in enumerate(self.features):
            for j, feature2 in enumerate(self.features):
                if i != j:
                    if feature1 in correlation_patterns and feature2 in correlation_patterns[feature1]:
                        correlation_matrix[i][j] = correlation_patterns[feature1][feature2]
                    elif feature2 in correlation_patterns and feature1 in correlation_patterns[feature2]:
                        correlation_matrix[i][j] = correlation_patterns[feature2][feature1]
                    else:
                        # Generate random correlation for missing pairs
                        correlation_matrix[i][j] = np.random.uniform(-0.3, 0.3)
        
        # Ensure matrix is symmetric
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
        np.fill_diagonal(correlation_matrix, 1.0)
        
        return {
            "type": "plotly_correlation",
            "title": "Feature Correlation Matrix",
            "data": {
                "heatmapZ": correlation_matrix.tolist(),
                "heatmapX": self.features,
                "heatmapY": self.features
            },
            "metadata": {
                "features_analyzed": len(self.features),
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def generate_pipeline_flow_data(self) -> Dict[str, Any]:
        """
        Generate data for D3.js force-directed graph showing the analysis pipeline
        """
        nodes = [
            {"id": "Video Input", "group": 1, "confidence": 0.95},
            {"id": "Frame Sampler", "group": 1, "confidence": 0.90},
            {"id": "Face Detector", "group": 2, "confidence": 0.88},
            {"id": "Audio Extractor", "group": 2, "confidence": 0.85},
        ]
        
        # Add specialist models
        model_confidences = {
            'BG': 0.82, 'AV': 0.87, 'CM': 0.91, 
            'RR': 0.89, 'LL': 0.78, 'TM': 0.93
        }
        
        for model in self.models:
            nodes.append({
                "id": f"{model} Model",
                "group": 3,
                "confidence": model_confidences[model]
            })
        
        # Add final processing nodes
        nodes.extend([
            {"id": "LangGraph Router", "group": 4, "confidence": 0.86},
            {"id": "Aggregator", "group": 4, "confidence": 0.84},
            {"id": "Explainer", "group": 5, "confidence": 0.88},
            {"id": "Final Result", "group": 5, "confidence": 0.85}
        ])
        
        # Define connections
        links = [
            {"source": "Video Input", "target": "Frame Sampler", "value": 1.0},
            {"source": "Video Input", "target": "Audio Extractor", "value": 0.8},
            {"source": "Frame Sampler", "target": "Face Detector", "value": 0.9},
        ]
        
        # Connect face detector to all models
        for model in self.models:
            links.append({
                "source": "Face Detector",
                "target": f"{model} Model",
                "value": model_confidences[model]
            })
        
        # Connect audio extractor to AV model
        links.append({"source": "Audio Extractor", "target": "AV Model", "value": 0.9})
        
        # Connect models to router
        for model in self.models:
            links.append({
                "source": f"{model} Model",
                "target": "LangGraph Router",
                "value": model_confidences[model]
            })
        
        # Final connections
        links.extend([
            {"source": "LangGraph Router", "target": "Aggregator", "value": 0.86},
            {"source": "Aggregator", "target": "Explainer", "value": 0.84},
            {"source": "Explainer", "target": "Final Result", "value": 0.88}
        ])
        
        return {
            "type": "d3_force_directed",
            "title": "Agentic Pipeline Flow",
            "data": {
                "nodes": nodes,
                "links": links
            },
            "metadata": {
                "total_nodes": len(nodes),
                "total_connections": len(links),
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def generate_comprehensive_analysis_report(self, video_path: str = None) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report with all visualization data
        """
        report = {
            "analysis_id": f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "video_info": {
                "path": video_path or "sample_video.mp4",
                "analyzed": True
            },
            "visualizations": {
                "frame_heatmap": self.generate_frame_confidence_heatmap(video_path),
                "model_performance": self.generate_model_performance_boxplot(),
                "feature_correlation": self.generate_feature_correlation_matrix(),
                "pipeline_flow": self.generate_pipeline_flow_data()
            },
            "summary": {
                "overall_confidence": np.random.uniform(0.75, 0.95),
                "prediction": "fake" if np.random.random() > 0.6 else "real",
                "processing_time": np.random.uniform(2.5, 8.2),
                "frames_analyzed": 100,
                "models_used": len(self.models)
            }
        }
        
        return report
    
    def save_analysis_data(self, report: Dict[str, Any], output_path: str = "analysis_data.json"):
        """
        Save analysis data to JSON file for frontend consumption
        """
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Analysis data saved to {output_path}")
        return output_path
    
    def generate_python_heatmap_images(self, report: Dict[str, Any], output_dir: str = "heatmap_images"):
        """
        Generate actual heatmap images using matplotlib/seaborn for comparison
        """
        Path(output_dir).mkdir(exist_ok=True)
        
        # 1. Frame confidence heatmap
        frame_data = report["visualizations"]["frame_heatmap"]["data"]
        confidence_matrix = []
        model_names = []
        
        for model_data in frame_data:
            model_names.append(model_data["name"])
            confidences = [point["y"] for point in model_data["data"]]
            confidence_matrix.append(confidences)
        
        plt.figure(figsize=(15, 8))
        sns.heatmap(confidence_matrix, 
                   xticklabels=[f"F{i+1}" for i in range(len(confidence_matrix[0]))][::10],
                   yticklabels=model_names,
                   cmap='viridis', 
                   cbar_kws={'label': 'Confidence Score'})
        plt.title('Frame-by-Frame Confidence Heatmap')
        plt.xlabel('Frame Number')
        plt.ylabel('Model')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/frame_confidence_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Feature correlation heatmap
        corr_data = report["visualizations"]["feature_correlation"]["data"]
        correlation_matrix = np.array(corr_data["heatmapZ"])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix,
                   xticklabels=corr_data["heatmapX"],
                   yticklabels=corr_data["heatmapY"],
                   annot=True,
                   cmap='RdBu_r',
                   center=0,
                   square=True,
                   cbar_kws={'label': 'Correlation Coefficient'})
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/feature_correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Heatmap images saved to {output_dir}/")

def main():
    """
    Main function to generate analysis data and heatmaps
    """
    generator = DeepfakeAnalysisHeatmapGenerator()
    
    # Generate comprehensive analysis report
    print("Generating comprehensive analysis report...")
    report = generator.generate_comprehensive_analysis_report("sample_video.mp4")
    
    # Save JSON data for frontend
    json_path = generator.save_analysis_data(report, "frontend_analysis_data.json")
    
    # Generate Python heatmap images for comparison
    print("Generating Python heatmap images...")
    generator.generate_python_heatmap_images(report)
    
    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Analysis ID: {report['analysis_id']}")
    print(f"Overall Confidence: {report['summary']['overall_confidence']:.3f}")
    print(f"Prediction: {report['summary']['prediction'].upper()}")
    print(f"Processing Time: {report['summary']['processing_time']:.2f}s")
    print(f"Frames Analyzed: {report['summary']['frames_analyzed']}")
    print(f"Models Used: {report['summary']['models_used']}")
    print("\nVisualization Data Generated:")
    print(f"- Frame Heatmap: {len(report['visualizations']['frame_heatmap']['data'])} models")
    print(f"- Model Performance: {len(report['visualizations']['model_performance']['data']['boxplotData'])} distributions")
    print(f"- Feature Correlation: {len(report['visualizations']['feature_correlation']['data']['heatmapX'])}x{len(report['visualizations']['feature_correlation']['data']['heatmapY'])} matrix")
    print(f"- Pipeline Flow: {len(report['visualizations']['pipeline_flow']['data']['nodes'])} nodes, {len(report['visualizations']['pipeline_flow']['data']['links'])} connections")
    print(f"\nData saved to: {json_path}")
    print("Heatmap images saved to: heatmap_images/")

if __name__ == "__main__":
    main()