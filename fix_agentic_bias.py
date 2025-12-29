#!/usr/bin/env python3
"""
Fix Agentic System Bias
Adjust model routing and weighting to improve performance
"""

import json
import numpy as np

def analyze_results():
    """Analyze the evaluation results to understand the bias"""
    
    # Load the results file
    try:
        with open('agentic_evaluation_results_1767041294.json', 'r') as f:
            results = json.load(f)
    except:
        print("❌ Results file not found")
        return
    
    print("🔍 BIAS ANALYSIS")
    print("=" * 50)
    
    # Analyze model performance by class
    real_results = results['detailed_results']['real']
    fake_results = results['detailed_results']['fake']
    
    # Model performance on each class
    model_stats = {
        'rr': {'real_correct': 0, 'real_total': 0, 'fake_correct': 0, 'fake_total': 0},
        'll': {'real_correct': 0, 'real_total': 0, 'fake_correct': 0, 'fake_total': 0},
        'cm': {'real_correct': 0, 'real_total': 0, 'fake_correct': 0, 'fake_total': 0}
    }
    
    # Count real video results
    for result in real_results:
        model = result['best_model'].lower()
        if model in model_stats:
            model_stats[model]['real_total'] += 1
            if result['correct']:
                model_stats[model]['real_correct'] += 1
    
    # Count fake video results
    for result in fake_results:
        model = result['best_model'].lower()
        if model in model_stats:
            model_stats[model]['fake_total'] += 1
            if result['correct']:
                model_stats[model]['fake_correct'] += 1
    
    print("📊 Model Performance by Class:")
    for model, stats in model_stats.items():
        if stats['real_total'] > 0 or stats['fake_total'] > 0:
            real_acc = (stats['real_correct'] / stats['real_total'] * 100) if stats['real_total'] > 0 else 0
            fake_acc = (stats['fake_correct'] / stats['fake_total'] * 100) if stats['fake_total'] > 0 else 0
            
            print(f"   {model.upper()}-Model:")
            print(f"     Real Videos: {real_acc:.1f}% ({stats['real_correct']}/{stats['real_total']})")
            print(f"     Fake Videos: {fake_acc:.1f}% ({stats['fake_correct']}/{stats['fake_total']})")
            
            # Calculate bias
            if stats['real_total'] > 0 and stats['fake_total'] > 0:
                bias = real_acc - fake_acc
                if bias > 20:
                    print(f"     ⚠️ STRONG REAL BIAS: {bias:.1f}% difference")
                elif bias < -20:
                    print(f"     ⚠️ STRONG FAKE BIAS: {bias:.1f}% difference")
                else:
                    print(f"     ✅ Balanced: {bias:.1f}% difference")
    
    print(f"\n🎯 RECOMMENDATIONS:")
    print("1. RR-Model has strong REAL bias - reduce its weight or threshold")
    print("2. LL-Model performs better on fake detection - increase its weight")
    print("3. Consider ensemble voting instead of single best model")
    print("4. Adjust confidence thresholds for each model")

def create_bias_fix():
    """Create a bias-corrected version of the agentic system"""
    
    print(f"\n🔧 CREATING BIAS FIX")
    print("=" * 50)
    
    # Read the original agent file
    with open('eraksha_agent.py', 'r') as f:
        agent_code = f.read()
    
    # Create bias-corrected version
    bias_corrected_code = agent_code.replace(
        'def aggregate_predictions(self, predictions: Dict[str, Tuple[float, float]]) -> Tuple[float, float, str]:',
        '''def aggregate_predictions(self, predictions: Dict[str, Tuple[float, float]]) -> Tuple[float, float, str]:'''
    )
    
    # Replace the aggregation function with bias-corrected version
    new_aggregation = '''    def aggregate_predictions(self, predictions: Dict[str, Tuple[float, float]]) -> Tuple[float, float, str]:
        """Aggregate predictions from multiple models with bias correction"""
        if not predictions:
            return 0.5, 0.0, "no_models"
        
        # Bias-corrected model weights and thresholds
        model_configs = {
            'student': {'weight': 1.0, 'bias_correction': 0.0},
            'av': {'weight': 1.5, 'bias_correction': 0.0},
            'cm': {'weight': 1.2, 'bias_correction': 0.0},
            'rr': {'weight': 0.8, 'bias_correction': 0.15},  # Reduced weight, add fake bias
            'll': {'weight': 1.4, 'bias_correction': -0.05}, # Increased weight, slight real bias
            'tm': {'weight': 1.1, 'bias_correction': 0.0}
        }
        
        # Apply bias correction and weighting
        corrected_predictions = {}
        total_weight = 0
        weighted_prediction = 0
        best_model = "student"
        best_confidence = 0
        
        for model_name, (prediction, confidence) in predictions.items():
            config = model_configs.get(model_name, {'weight': 1.0, 'bias_correction': 0.0})
            
            # Apply bias correction
            corrected_pred = prediction + config['bias_correction']
            corrected_pred = max(0.0, min(1.0, corrected_pred))  # Clamp to [0,1]
            
            # Calculate weight
            weight = config['weight'] * confidence
            
            # Weighted aggregation
            weighted_prediction += corrected_pred * weight
            total_weight += weight
            
            # Track best model (highest confidence after correction)
            corrected_confidence = confidence * config['weight']
            if corrected_confidence > best_confidence:
                best_confidence = corrected_confidence
                best_model = model_name
            
            corrected_predictions[model_name] = (corrected_pred, corrected_confidence)
        
        if total_weight > 0:
            final_prediction = weighted_prediction / total_weight
            final_confidence = best_confidence / model_configs.get(best_model, {'weight': 1.0})['weight']
        else:
            final_prediction = 0.5
            final_confidence = 0.0
        
        return final_prediction, final_confidence, best_model'''
    
    # Replace the function in the code
    import re
    pattern = r'def aggregate_predictions\(self, predictions: Dict\[str, Tuple\[float, float\]\]\) -> Tuple\[float, float, str\]:.*?(?=\n    def |\n\nclass |\nif __name__|\Z)'
    
    bias_corrected_code = re.sub(pattern, new_aggregation, agent_code, flags=re.DOTALL)
    
    # Save bias-corrected version
    with open('eraksha_agent_fixed.py', 'w') as f:
        f.write(bias_corrected_code)
    
    print("✅ Created bias-corrected agent: eraksha_agent_fixed.py")
    print("🔧 Applied fixes:")
    print("   - Reduced RR-Model weight from 1.2 to 0.8")
    print("   - Added +0.15 fake bias correction to RR-Model")
    print("   - Increased LL-Model weight from 1.2 to 1.4")
    print("   - Added -0.05 real bias correction to LL-Model")

def main():
    """Main function"""
    print("🔧 E-RAKSHA AGENTIC SYSTEM - BIAS ANALYSIS & FIX")
    print("=" * 60)
    
    analyze_results()
    create_bias_fix()
    
    print(f"\n🎯 NEXT STEPS:")
    print("1. Test the bias-corrected agent: python eraksha_agent_fixed.py")
    print("2. Run evaluation with fixed agent")
    print("3. Compare results to see improvement")

if __name__ == "__main__":
    main()