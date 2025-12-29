# ============================================
# PERSON 3: INTEGRATION WITH MAIN SYSTEM
# ============================================

import os
import sys
from person_3 import load_ll_model, load_tm_model, TMModel, LLModel
import torch
import torch.nn.functional as F

def integrate_with_agent_system():
    """
    Integration function for the LangGraph agent system
    This shows how Person 3's models integrate with Person 1's system
    """
    
    print("🔗 Person 3 Integration with E-Raksha System")
    print("=" * 60)
    
    # Model paths
    ll_model_path = "ll_model_student (1).pt"
    tm_model_path = "tm_model_student.pt"
    
    # Check if models exist
    ll_exists = os.path.exists(ll_model_path)
    tm_exists = os.path.exists(tm_model_path)
    
    print(f"📱 LL-Model (Low-Light): {'✅ Found' if ll_exists else '❌ Missing'}")
    print(f"⏰ TM-Model (Temporal): {'✅ Found' if tm_exists else '❌ Missing'}")
    
    if not (ll_exists and tm_exists):
        print("\n⚠️ Some model files are missing!")
        print("Make sure you have:")
        print(f"   - {ll_model_path}")
        print(f"   - {tm_model_path}")
        return False
    
    # Test model loading
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️ Device: {device}")
    
    try:
        # Load models
        ll_model = load_ll_model(ll_model_path, device)
        tm_model = load_tm_model(tm_model_path, device)
        
        print("✅ Both models loaded successfully!")
        
        # Test inference with dummy data
        print("\n🧪 Testing inference with dummy face data...")
        
        # Simulate face tensors (batch_size=1, frames=10, channels=3, height=224, width=224)
        dummy_faces_ll = torch.randn(1, 10, 3, 224, 224).to(device)
        dummy_faces_tm = torch.randn(1, 15, 3, 224, 224).to(device)  # More frames for temporal
        
        with torch.no_grad():
            # LL-Model inference
            ll_logits = ll_model(dummy_faces_ll)
            ll_probs = F.softmax(ll_logits, dim=1)
            ll_prediction = ll_probs[0, 1].item()
            
            # TM-Model inference
            tm_logits = tm_model(dummy_faces_tm)
            tm_probs = F.softmax(tm_logits, dim=1)
            tm_prediction = tm_probs[0, 1].item()
        
        print(f"📱 LL-Model prediction: {ll_prediction:.3f} ({'FAKE' if ll_prediction > 0.5 else 'REAL'})")
        print(f"⏰ TM-Model prediction: {tm_prediction:.3f} ({'FAKE' if tm_prediction > 0.5 else 'REAL'})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during integration test: {e}")
        return False

def create_agent_integration_code():
    """
    Generate the code that Person 1 needs to integrate Person 3's models
    """
    
    integration_code = '''
# ============================================
# INTEGRATION CODE FOR PERSON 1 (PRANAY)
# Add this to your agent system
# ============================================

# In src/agent/nodes.py, add these imports:
from person_3 import load_ll_model, load_tm_model
import torch.nn.functional as F

# Add this function to select Person 3's models:
def select_person3_specialist(metadata: dict) -> str:
    """Select between LL-Model and TM-Model based on video characteristics"""
    brightness = metadata.get('avg_brightness', 128)
    
    if brightness < 50:  # Dark video
        return 'll_model'
    else:
        return 'tm_model'

# Modify the domain_inference_node function:
def domain_inference_node(state: AgentState) -> AgentState:
    """Run specialist model inference"""
    specialist = select_specialist(state['metadata'])
    
    # Check if it's a Person 3 model
    if specialist in ['ll_model', 'tm_model']:
        return person3_inference_node(state, specialist)
    else:
        # Handle other specialists (Person 2, Person 4)
        return original_domain_inference_node(state)

def person3_inference_node(state: AgentState, model_type: str) -> AgentState:
    """Handle Person 3's LL-Model and TM-Model inference"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if model_type == 'll_model':
        model = load_ll_model("ll_model_student (1).pt", device)
        max_frames = 10
    else:  # tm_model
        model = load_tm_model("tm_model_student.pt", device)
        max_frames = 15
    
    # Use existing face extraction from state
    frames = torch.stack(state['frames'][:max_frames]).to(device)
    
    with torch.no_grad():
        logits = model(frames.unsqueeze(0))
        probs = F.softmax(logits, dim=1)
        prediction = probs[0, 1].item()
    
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    state['specialist_prediction'] = prediction
    state['specialist_confidence'] = confidence
    state['selected_specialist'] = f'Person3_{model_type}'
    
    print(f'[PERSON3] {model_type}: Prediction: {prediction:.3f}, Confidence: {confidence:.3f}')
    
    return state
'''
    
    print("\n📋 Integration Code for Person 1:")
    print("=" * 60)
    print(integration_code)
    
    # Save to file
    with open("person3_integration_code.txt", "w") as f:
        f.write(integration_code)
    
    print("💾 Integration code saved to: person3_integration_code.txt")

def model_summary():
    """Display summary of Person 3's contribution"""
    
    print("\n📊 Person 3 Model Summary")
    print("=" * 60)
    
    # Check model files
    models = {
        "LL-Model (Low-Light)": "ll_model_student (1).pt",
        "TM-Model (Temporal)": "tm_model_student.pt"
    }
    
    for name, path in models.items():
        exists = os.path.exists(path)
        size = os.path.getsize(path) / (1024*1024) if exists else 0
        
        print(f"🔹 {name}")
        print(f"   File: {path}")
        print(f"   Status: {'✅ Ready' if exists else '❌ Missing'}")
        if exists:
            print(f"   Size: {size:.1f} MB")
        print()
    
    print("🎯 Specializations:")
    print("   📱 LL-Model: Optimized for low-light/dark videos")
    print("   ⏰ TM-Model: Detects temporal inconsistencies with LSTM")
    print()
    print("🤖 Auto-selection Logic:")
    print("   • Brightness < 50: Use LL-Model")
    print("   • Brightness ≥ 50: Use TM-Model")
    print()
    print("🔗 Integration Status:")
    print("   • Models: Ready for integration")
    print("   • API: person_3.py provides all functions")
    print("   • Testing: Basic functionality verified")

if __name__ == "__main__":
    # Run integration test
    success = integrate_with_agent_system()
    
    if success:
        # Generate integration code
        create_agent_integration_code()
        
        # Show summary
        model_summary()
        
        print("\n🎉 Person 3 Integration Complete!")
        print("✅ Models are ready for the main system")
        print("📤 Send the following files to Person 1 (Pranay):")
        print("   • person_3.py")
        print("   • ll_model_student (1).pt")
        print("   • tm_model_student.pt")
        print("   • person3_integration_code.txt")
    else:
        print("\n❌ Integration test failed!")
        print("Please check your model files and try again.")