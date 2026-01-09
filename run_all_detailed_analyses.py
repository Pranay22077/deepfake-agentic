#!/usr/bin/env python3
"""
E-Raksha Master Analysis Controller

Comprehensive analysis orchestrator for all specialist models.
Runs detailed performance analysis and generates automated training roadmaps.

Features:
- Multi-model performance analysis
- Bias detection and correction recommendations
- Automated training roadmap generation
- Performance optimization insights
- Systematic improvement planning

Author: E-Raksha Team
"""

import sys
import os
from pathlib import Path
import json
import time
from datetime import datetime

# Import analysis modules
sys.path.insert(0, str(Path(__file__).parent))

def run_comprehensive_analysis():
    """
    Execute comprehensive analysis across all specialist models.
    
    Returns:
        dict: Complete analysis results and recommendations
    """
    print("=" * 80)
    print("E-RAKSHA COMPREHENSIVE MODEL ANALYSIS SUITE")
    print("=" * 80)
    print(f"Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Analysis pipeline configuration
    analyses_to_run = [
        {
            'name': 'Comprehensive Multi-Model Analysis',
            'script': 'detailed_model_analysis.py',
            'description': 'Analyzes all models together with ensemble recommendations'
        },
        {
            'name': 'BG Model Detailed Analysis',
            'script': 'test_bg_model_detailed.py',
            'description': 'Background/Lighting specialist - lighting artifacts, shadow detection'
        },
        {
            'name': 'CM Model Detailed Analysis',
            'script': 'test_cm_model_detailed.py',
            'description': 'Compression specialist - DCT analysis, quantization artifacts'
        }
    ]
    
    results_summary = {
        'analysis_timestamp': datetime.now().isoformat(),
        'analyses_completed': [],
        'analyses_failed': [],
        'overall_recommendations': []
    }
    
    for analysis in analyses_to_run:
        print(f"\n{'='*60}")
        print(f"RUNNING: {analysis['name']}")
        print(f"Description: {analysis['description']}")
        print(f"{'='*60}")
        
        try:
            start_time = time.time()
            
            # Run the analysis script
            exit_code = os.system(f"python {analysis['script']}")
            
            execution_time = time.time() - start_time
            
            if exit_code == 0:
                print(f"\n✅ {analysis['name']} completed successfully in {execution_time:.1f}s")
                results_summary['analyses_completed'].append({
                    'name': analysis['name'],
                    'script': analysis['script'],
                    'execution_time': execution_time,
                    'status': 'success'
                })
            else:
                print(f"\n❌ {analysis['name']} failed with exit code {exit_code}")
                results_summary['analyses_failed'].append({
                    'name': analysis['name'],
                    'script': analysis['script'],
                    'exit_code': exit_code,
                    'status': 'failed'
                })
                
        except Exception as e:
            print(f"\n❌ {analysis['name']} failed with exception: {e}")
            results_summary['analyses_failed'].append({
                'name': analysis['name'],
                'script': analysis['script'],
                'error': str(e),
                'status': 'error'
            })
    
    # Generate overall summary
    generate_overall_summary(results_summary)
    
    return results_summary

def generate_overall_summary(results_summary):
    """Generate overall analysis summary and recommendations"""
    print(f"\n{'='*80}")
    print("OVERALL ANALYSIS SUMMARY")
    print(f"{'='*80}")
    
    completed = len(results_summary['analyses_completed'])
    failed = len(results_summary['analyses_failed'])
    total = completed + failed
    
    print(f"\n📊 EXECUTION SUMMARY:")
    print(f"   Total Analyses: {total}")
    print(f"   Completed Successfully: {completed}")
    print(f"   Failed: {failed}")
    print(f"   Success Rate: {completed/total*100:.1f}%" if total > 0 else "   Success Rate: 0%")
    
    if results_summary['analyses_completed']:
        print(f"\n✅ COMPLETED ANALYSES:")
        for analysis in results_summary['analyses_completed']:
            print(f"   • {analysis['name']} ({analysis['execution_time']:.1f}s)")
    
    if results_summary['analyses_failed']:
        print(f"\n❌ FAILED ANALYSES:")
        for analysis in results_summary['analyses_failed']:
            error_msg = analysis.get('error', f"Exit code: {analysis.get('exit_code', 'unknown')}")
            print(f"   • {analysis['name']} - {error_msg}")
    
    # Load and summarize individual analysis results
    analysis_files = [
        ('comprehensive_model_analysis.json', 'Multi-Model Analysis'),
        ('bg_model_detailed_analysis.json', 'BG Model Analysis'),
        ('cm_model_detailed_analysis.json', 'CM Model Analysis')
    ]
    
    model_performances = {}
    critical_issues = []
    training_priorities = []
    
    print(f"\n📈 MODEL PERFORMANCE SUMMARY:")
    
    for file_path, analysis_name in analysis_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                if 'comprehensive_model_analysis.json' in file_path:
                    # Multi-model analysis
                    for model_key, model_data in data.items():
                        if isinstance(model_data, dict) and 'summary' in model_data:
                            accuracy = model_data['summary'].get('overall_accuracy', 0)
                            model_performances[model_key] = accuracy
                            print(f"   {model_data.get('model_name', model_key)}: {accuracy:.1%}")
                            
                            # Check for critical issues
                            if accuracy < 0.6:
                                critical_issues.append(f"{model_data.get('model_name', model_key)} has poor accuracy ({accuracy:.1%})")
                            
                            bias = model_data['summary'].get('bias', 0)
                            if abs(bias) > 0.3:
                                critical_issues.append(f"{model_data.get('model_name', model_key)} has high bias ({bias:+.1%})")
                
                else:
                    # Individual model analysis
                    if 'recommendations' in data and 'priority' in data['recommendations']:
                        priority = data['recommendations']['priority']
                        model_name = data.get('model_info', {}).get('name', analysis_name)
                        training_priorities.append((model_name, priority))
                        
                        if priority == 'HIGH':
                            critical_issues.append(f"{model_name} needs urgent training improvements")
                            
            except Exception as e:
                print(f"   ❌ Failed to load {analysis_name}: {e}")
    
    # Generate overall recommendations
    overall_recommendations = []
    
    if model_performances:
        # Find best and worst performing models
        best_model = max(model_performances.items(), key=lambda x: x[1])
        worst_model = min(model_performances.items(), key=lambda x: x[1])
        
        overall_recommendations.extend([
            f"PRIORITIZE: {best_model[0]} is the best performer ({best_model[1]:.1%}) - use as ensemble leader",
            f"URGENT FIX: {worst_model[0]} needs immediate attention ({worst_model[1]:.1%})",
        ])
        
        # Check ensemble readiness
        good_models = [k for k, v in model_performances.items() if v > 0.65]
        if len(good_models) >= 3:
            overall_recommendations.append(f"ENSEMBLE READY: {len(good_models)} models suitable for production ensemble")
        else:
            overall_recommendations.append(f"ENSEMBLE NOT READY: Only {len(good_models)} models performing adequately")
    
    if critical_issues:
        print(f"\n🚨 CRITICAL ISSUES:")
        for issue in critical_issues[:5]:  # Show top 5 issues
            print(f"   • {issue}")
    
    if training_priorities:
        high_priority = [tp for tp in training_priorities if tp[1] == 'HIGH']
        if high_priority:
            print(f"\n🔥 HIGH PRIORITY TRAINING NEEDED:")
            for model_name, _ in high_priority:
                print(f"   • {model_name}")
    
    print(f"\n🎯 OVERALL RECOMMENDATIONS:")
    for i, rec in enumerate(overall_recommendations, 1):
        print(f"   {i}. {rec}")
    
    # Additional system-wide recommendations
    system_recommendations = [
        "Implement model-specific training pipelines based on individual analyses",
        "Focus on dataset diversity for models with high bias",
        "Consider architecture improvements for low-confidence models",
        "Establish continuous monitoring for model performance degradation",
        "Create specialized datasets for each model's weaknesses"
    ]
    
    print(f"\n📋 SYSTEM-WIDE RECOMMENDATIONS:")
    for i, rec in enumerate(system_recommendations, 1):
        print(f"   {i}. {rec}")
    
    # Save overall summary
    results_summary['overall_recommendations'] = overall_recommendations + system_recommendations
    results_summary['model_performances'] = model_performances
    results_summary['critical_issues'] = critical_issues
    results_summary['training_priorities'] = training_priorities
    
    with open('overall_analysis_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print(f"\n💾 Overall summary saved to 'overall_analysis_summary.json'")
    
    # Generate training roadmap
    generate_training_roadmap(results_summary)

def generate_training_roadmap(results_summary):
    """Generate a comprehensive training roadmap"""
    print(f"\n{'='*80}")
    print("TRAINING ROADMAP GENERATION")
    print(f"{'='*80}")
    
    roadmap = {
        'roadmap_version': '1.0',
        'created_at': datetime.now().isoformat(),
        'phases': []
    }
    
    # Phase 1: Critical Issues (0-2 weeks)
    phase1 = {
        'phase': 1,
        'name': 'Critical Issues Resolution',
        'duration': '0-2 weeks',
        'priority': 'URGENT',
        'tasks': []
    }
    
    if results_summary.get('critical_issues'):
        phase1['tasks'].extend([
            'Fix broken models (TM model completely broken)',
            'Address models with accuracy < 60%',
            'Correct high bias models (>30% bias)',
            'Implement emergency model fixes'
        ])
    
    # Phase 2: Model-Specific Improvements (2-6 weeks)
    phase2 = {
        'phase': 2,
        'name': 'Model-Specific Improvements',
        'duration': '2-6 weeks',
        'priority': 'HIGH',
        'tasks': [
            'Implement BG model lighting detection improvements',
            'Enhance CM model compression artifact detection',
            'Improve AV model lip-sync analysis',
            'Upgrade RR model resolution artifact detection',
            'Enhance LL model low-light processing'
        ]
    }
    
    # Phase 3: Dataset Enhancement (4-8 weeks)
    phase3 = {
        'phase': 3,
        'name': 'Dataset Enhancement',
        'duration': '4-8 weeks',
        'priority': 'MEDIUM',
        'tasks': [
            'Collect model-specific training data',
            'Create specialized datasets for each model weakness',
            'Implement advanced data augmentation',
            'Balance real/fake samples for biased models',
            'Add challenging edge cases'
        ]
    }
    
    # Phase 4: Architecture Optimization (6-10 weeks)
    phase4 = {
        'phase': 4,
        'name': 'Architecture Optimization',
        'duration': '6-10 weeks',
        'priority': 'MEDIUM',
        'tasks': [
            'Implement attention mechanisms',
            'Add multi-scale processing',
            'Enhance specialist modules',
            'Optimize ensemble weights',
            'Add cross-model feature sharing'
        ]
    }
    
    # Phase 5: Production Optimization (8-12 weeks)
    phase5 = {
        'phase': 5,
        'name': 'Production Optimization',
        'duration': '8-12 weeks',
        'priority': 'LOW',
        'tasks': [
            'Model compression and optimization',
            'Inference speed improvements',
            'Memory usage optimization',
            'Deployment pipeline automation',
            'Continuous monitoring setup'
        ]
    }
    
    roadmap['phases'] = [phase1, phase2, phase3, phase4, phase5]
    
    # Print roadmap
    print(f"\n📅 TRAINING ROADMAP:")
    for phase in roadmap['phases']:
        print(f"\n   PHASE {phase['phase']}: {phase['name']}")
        print(f"   Duration: {phase['duration']} | Priority: {phase['priority']}")
        print(f"   Tasks:")
        for task in phase['tasks']:
            print(f"     • {task}")
    
    # Save roadmap
    with open('training_roadmap.json', 'w') as f:
        json.dump(roadmap, f, indent=2, default=str)
    
    print(f"\n💾 Training roadmap saved to 'training_roadmap.json'")
    
    # Generate next steps
    print(f"\n🚀 IMMEDIATE NEXT STEPS:")
    print(f"   1. Review individual model analysis files for detailed recommendations")
    print(f"   2. Start with Phase 1 critical issues (broken models)")
    print(f"   3. Implement model-specific improvements based on analysis")
    print(f"   4. Set up continuous monitoring for model performance")
    print(f"   5. Begin collecting specialized training data")

def main():
    """Main execution function"""
    print("🤖 Starting comprehensive deepfake detection model analysis...")
    
    start_time = time.time()
    results = run_comprehensive_analysis()
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Total execution time: {total_time:.1f} seconds")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n📁 GENERATED FILES:")
    output_files = [
        'comprehensive_model_analysis.json',
        'bg_model_detailed_analysis.json', 
        'cm_model_detailed_analysis.json',
        'overall_analysis_summary.json',
        'training_roadmap.json'
    ]
    
    for file_path in output_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"   ✅ {file_path} ({file_size:,} bytes)")
        else:
            print(f"   ❌ {file_path} (not generated)")
    
    print(f"\n🎯 SUMMARY:")
    completed = len(results['analyses_completed'])
    total = completed + len(results['analyses_failed'])
    print(f"   Analyses completed: {completed}/{total}")
    
    if results['critical_issues']:
        print(f"   Critical issues found: {len(results['critical_issues'])}")
    
    print(f"\n📖 NEXT STEPS:")
    print(f"   1. Review 'overall_analysis_summary.json' for high-level insights")
    print(f"   2. Check individual model analysis files for detailed recommendations")
    print(f"   3. Follow 'training_roadmap.json' for systematic improvements")
    print(f"   4. Start with critical issues (Phase 1) immediately")

if __name__ == "__main__":
    main()