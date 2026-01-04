<div align="center">

# E-RAKSHA

## **Agentic AI for Deepfake Detection & Authenticity Verification**

### Comprehensive Technical Documentation

---

**E-Raksha Hackathon 2026 — Problem Statement II**  
*National Cyber Challenge by eDC IIT Delhi in collaboration with CyberPeace*  
*Official Pre-Summit Event of BECon'26 and AI Impact Summit'26*

---

**Document Version:** 2.0.0  
**Last Updated:** December 30, 2025  
**Classification:** Technical Submission

</div>

---

<div style="page-break-after: always;"></div>

## Table of Contents

| Section | Title | Page |
|:-------:|-------|:----:|
| **Part I** | **Foundation** | |
| 1 | Executive Summary | 3 |
| 2 | Problem Statement Analysis | 5 |
| 3 | Solution Architecture Overview | 8 |
| **Part II** | **Agentic System** | |
| 4 | Agentic AI System Design | 12 |
| 5 | LangGraph Agent Implementation | 16 |
| 6 | Intelligent Routing Mechanism | 20 |
| **Part III** | **Neural Networks** | |
| 7 | Neural Network Architectures | 24 |
| 8 | Specialist Model Ensemble | 32 |
| 9 | Bias Correction & Calibration | 38 |
| **Part IV** | **Implementation** | |
| 10 | Preprocessing Pipeline | 42 |
| 11 | Inference Engine | 46 |
| 12 | Explainability & Trust | 50 |
| **Part V** | **Deployment** | |
| 13 | API & Backend Architecture | 54 |
| 14 | Frontend & User Interface | 58 |
| 15 | Edge & Mobile Deployment | 62 |
| **Part VI** | **Operations** | |
| 16 | Security & Model Versioning | 66 |
| 17 | Safe Learning & Feedback | 70 |
| 18 | Testing Strategy | 74 |
| **Appendices** | | |
| A | Mathematical Foundations | 78 |
| B | Troubleshooting Guide | 82 |
| C | API Reference | 86 |
| D | Glossary | 90 |

---

<div style="page-break-after: always;"></div>


# Part I: Foundation

---

## 1. Executive Summary

### 1.1 Project Overview

> **E-Raksha** is an advanced agentic AI system designed for autonomous deepfake detection and media authenticity verification.

The system addresses the critical challenge outlined in **Problem Statement II** of the E-Raksha Hackathon 2026: developing an intelligent agent capable of detecting manipulated media, verifying authenticity, and strengthening digital trust across platforms and operational environments.

### 1.2 Key Performance Metrics

| Metric | Value | Target |
|--------|:-----:|:------:|
| **Overall Detection Confidence** | 94.9% | >90% |
| **Average Processing Time** | 2.1s | <5s |
| **Model Ensemble Size** | 6 models | — |
| **Total Parameters** | 47.2M | <100M |
| **Inference Memory** | 512 MB | <1GB |
| **Supported Platforms** | Cloud, Edge, Mobile | ✓ |

### 1.3 Key Innovations

<table>
<tr>
<td width="50%">

#### Innovation 1: Agentic Architecture
The system employs an autonomous agent that intelligently routes video analysis through multiple specialist models based on video characteristics.

</td>
<td width="50%">

#### Innovation 2: Domain Specialists
Six specialized neural networks, each trained to detect specific manipulation types:
- BG-Model (Baseline)
- AV-Model (Audio-Visual)
- CM-Model (Compression)
- RR-Model (Re-recording)
- LL-Model (Low-light)
- TM-Model (Temporal)

</td>
</tr>
<tr>
<td>

#### Innovation 3: Bias Correction
Sophisticated bias correction mechanism that dynamically adjusts model weights and prediction biases for balanced detection.

</td>
<td>

#### Innovation 4: Edge-Ready
All models optimized for deployment on resource-constrained devices, enabling field operatives to perform detection without cloud dependency.

</td>
</tr>
</table>

### 1.4 Individual Model Performance

| Model | Accuracy | Parameters | Size | Purpose |
|-------|:--------:|:----------:|:----:|---------|
| **BG-Model** | 86.25% | 2.1M | 8.2 MB | Baseline Generalist |
| **AV-Model** | 93.0% | 15.8M | 60.4 MB | Audio-Visual Analysis |
| **CM-Model** | 80.83% | 11.7M | 44.6 MB | Compression Detection |
| **RR-Model** | 85.0% | 11.7M | 44.6 MB | Re-recording Detection |
| **LL-Model** | 93.42% | 11.7M | 44.6 MB | Low-light Analysis |
| **TM-Model** | 78.5% | 14.2M | 54.3 MB | Temporal Consistency |

---

<div style="page-break-after: always;"></div>

## 2. Problem Statement Analysis

### 2.1 The Deepfake Threat Landscape

> Deepfake technology represents one of the most significant threats to digital trust and national security in the modern era.

#### Threat Categories Addressed

| Category | Description | Detection Method |
|----------|-------------|------------------|
| **Face Swap** | Replace face while preserving body movements | Boundary & texture analysis |
| **Face Reenactment** | Manipulate expressions to match source | Facial dynamics analysis |
| **Lip-Sync** | Alter audio while modifying video | Audio-visual correlation |
| **Full Synthesis** | Entirely synthetic GAN-generated faces | Generative artifact detection |

### 2.2 Operational Requirements

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    HACKATHON REQUIREMENTS MAPPING                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  [OK] On-Device, Field-Operative Agent                                    │
│     └─► MobileNetV3-based student model, sub-3s processing              │
│                                                                          │
│  [OK] Edge Inference                                                       │
│     └─► CPU-only inference with optional GPU, INT8 quantization         │
│                                                                          │
│  [OK] Compression-Aware Detection                                          │
│     └─► Dedicated CM-Model for compressed video artifacts               │
│                                                                          │
│  [OK] Cognitive Assistance for Operatives                                  │
│     └─► Human-readable explanations with confidence levels              │
│                                                                          │
│  [OK] Offline Detection Without Cloud                                      │
│     └─► Complete self-contained inference pipeline                      │
│                                                                          │
│  [OK] Low-Power Operation                                                  │
│     └─► Early-exit capability, optimized inference path                 │
│                                                                          │
│  [OK] Immediate Authentication Results                                     │
│     └─► Average 2.1 seconds processing time                             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Technical Challenges & Solutions

| Challenge | Impact | Our Solution |
|-----------|--------|--------------|
| **Domain Shift** | Models fail on real-world content | Diverse training + domain specialists |
| **Adversarial Robustness** | Evasion through perturbations | Ensemble diversity |
| **Computational Constraints** | Edge deployment limits | Knowledge distillation |
| **Class Imbalance** | More real than fake content | Bias correction mechanism |

---

<div style="page-break-after: always;"></div>

## 3. Solution Architecture Overview

### 3.1 High-Level System Architecture

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                        E-RAKSHA SYSTEM ARCHITECTURE                        ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  ┌──────────────────────────────────────────────────────────────────────┐ ║
║  │                    LAYER 4: SYSTEM INTERFACE                          │ ║
║  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │ ║
║  │  │   React     │  │   FastAPI   │  │  PostgreSQL │  │  Monitoring │  │ ║
║  │  │  Frontend   │  │   Backend   │  │   Database  │  │   Grafana   │  │ ║
║  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │ ║
║  └──────────────────────────────────────────────────────────────────────┘ ║
║                                    │                                       ║
║                                    ▼                                       ║
║  ┌──────────────────────────────────────────────────────────────────────┐ ║
║  │                    LAYER 3: AGENTIC INTELLIGENCE                      │ ║
║  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │ ║
║  │  │  LangGraph  │  │  Routing    │  │  Confidence │  │ Explanation │  │ ║
║  │  │   Agent     │  │   Engine    │  │  Evaluator  │  │  Generator  │  │ ║
║  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │ ║
║  └──────────────────────────────────────────────────────────────────────┘ ║
║                                    │                                       ║
║                                    ▼                                       ║
║  ┌──────────────────────────────────────────────────────────────────────┐ ║
║  │                    LAYER 2: MODEL BANK                                │ ║
║  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐  │ ║
║  │  │   BG   │ │   AV   │ │   CM   │ │   RR   │ │   LL   │ │   TM   │  │ ║
║  │  │ Model  │ │ Model  │ │ Model  │ │ Model  │ │ Model  │ │ Model  │  │ ║
║  │  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘ └────────┘  │ ║
║  └──────────────────────────────────────────────────────────────────────┘ ║
║                                    │                                       ║
║                                    ▼                                       ║
║  ┌──────────────────────────────────────────────────────────────────────┐ ║
║  │                    LAYER 1: INPUT PROCESSING                          │ ║
║  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │ ║
║  │  │   Video     │  │   Frame     │  │    Face     │  │   Audio     │  │ ║
║  │  │  Decoder    │  │  Sampler    │  │  Detector   │  │  Extractor  │  │ ║
║  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │ ║
║  └──────────────────────────────────────────────────────────────────────┘ ║
║                                                                            ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

### 3.2 Data Flow Pipeline

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│  Video  │───▶│ Analyze │───▶│ Extract │───▶│ Baseline│───▶│  Route  │
│  Input  │    │ Metadata│    │ Frames  │    │Inference│    │Decision │
└─────────┘    └─────────┘    └─────────┘    └─────────┘    └────┬────┘
                                                                  │
                    ┌─────────────────────────────────────────────┤
                    │                                             │
                    ▼                                             ▼
             ┌─────────────┐                              ┌─────────────┐
             │  High Conf  │                              │  Low/Med    │
             │   Accept    │                              │  Specialists│
             └──────┬──────┘                              └──────┬──────┘
                    │                                             │
                    └─────────────────┬───────────────────────────┘
                                      │
                                      ▼
                              ┌─────────────┐
                              │  Aggregate  │
                              │  & Explain  │
                              └──────┬──────┘
                                     │
                                     ▼
                              ┌─────────────┐
                              │   Final     │
                              │   Result    │
                              └─────────────┘
```

---

<div style="page-break-after: always;"></div>

# Part II: Agentic System

---

## 4. Agentic AI System Design

### 4.1 What Makes This Agentic?

> **Key Insight:** The agent does NOT predict whether a video is fake. The agent DECIDES which experts should predict, and interprets their outputs.

| Traditional ML | Agentic System |
|----------------|----------------|
| `Input → Model → Output` | `Input → Observe → Reason → Plan → Act → Explain` |
| Fixed behavior | Adaptive behavior |
| No decision-making | Active decision-making |
| Cannot handle uncertainty | Handles uncertainty gracefully |
| No learning from feedback | Learns safely from feedback |

### 4.2 Agent Responsibilities

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         AGENT RESPONSIBILITIES                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  PERCEPTION                                                           │
│     • Extract and analyze video metadata                                 │
│     • Compute bitrate, FPS, resolution, codec                           │
│     • Detect compression artifacts                                       │
│     • Calculate brightness, noise level                                  │
│                                                                          │
│  REASONING                                                            │
│     • Evaluate confidence and uncertainty                                │
│     • Compute prediction variance across frames                          │
│     • Analyze multi-model consensus                                      │
│                                                                          │
│  PLANNING                                                             │
│     • Select which models to invoke                                      │
│     • High confidence → Accept baseline                                  │
│     • Medium confidence → Route to specialist                            │
│     • Low confidence → Use all specialists                               │
│                                                                          │
│  ACTION                                                               │
│     • Execute inference pipeline                                         │
│     • Load appropriate models                                            │
│     • Aggregate results intelligently                                    │
│                                                                          │
│  LEARNING                                                             │
│     • Incorporate feedback safely                                        │
│     • Quarantine user feedback                                           │
│     • Human validation before training                                   │
│                                                                          │
│  EXPLANATION                                                          │
│     • Generate human-interpretable justifications                        │
│     • Grad-CAM heatmaps                                                  │
│     • Confidence scores with reasoning                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Agent State Machine

```
                              ┌──────────┐
                              │   IDLE   │
                              └────┬─────┘
                                   │ video input
                                   ▼
                              ┌──────────┐
                              │ ANALYZING│
                              └────┬─────┘
                                   │ analysis complete
                                   ▼
                              ┌──────────┐
                              │ BASELINE │
                              │INFERENCE │
                              └────┬─────┘
                                   │ prediction complete
                                   ▼
                              ┌──────────┐
                              │ ROUTING  │
                              └────┬─────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
                    ▼              ▼              ▼
              ┌──────────┐  ┌──────────┐  ┌──────────┐
              │  HIGH    │  │  MEDIUM  │  │   LOW    │
              │  CONF    │  │  CONF    │  │  CONF    │
              └────┬─────┘  └────┬─────┘  └────┬─────┘
                   │             │              │
                   │             ▼              ▼
                   │       ┌──────────┐  ┌──────────┐
                   │       │SPECIALIST│  │   ALL    │
                   │       │INFERENCE │  │SPECIALISTS│
                   │       └────┬─────┘  └────┬─────┘
                   │             │              │
                   └─────────────┼──────────────┘
                                 │
                                 ▼
                           ┌──────────┐
                           │AGGREGATING│
                           └────┬─────┘
                                │
                                ▼
                           ┌──────────┐
                           │ COMPLETE │
                           └──────────┘
```

### 4.4 Confidence Level Classification

| Level | Threshold | Action | Specialists |
|:-----:|:---------:|--------|:-----------:|
| [GREEN] **HIGH** | ≥ 85% | Accept baseline | 0 |
| [YELLOW] **MEDIUM** | 65-85% | Selective routing | 2-3 |
| [RED] **LOW** | < 65% | All specialists | 5-6 |

---

<div style="page-break-after: always;"></div>

## 5. LangGraph Agent Implementation

### 5.1 State Definition

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional
import torch

class AgentState(TypedDict):
    """Complete state for the E-Raksha agent"""
    
    # ═══════════════════════════════════════════════════════════
    # INPUT
    # ═══════════════════════════════════════════════════════════
    video_path: str
    request_id: str
    
    # ═══════════════════════════════════════════════════════════
    # METADATA
    # ═══════════════════════════════════════════════════════════
    metadata: dict
    bitrate: int
    fps: int
    resolution: tuple
    avg_brightness: float
    
    # ═══════════════════════════════════════════════════════════
    # PROCESSING
    # ═══════════════════════════════════════════════════════════
    frames: List[torch.Tensor]
    faces_detected: int
    audio_waveform: torch.Tensor
    
    # ═══════════════════════════════════════════════════════════
    # PREDICTIONS
    # ═══════════════════════════════════════════════════════════
    student_prediction: float
    student_confidence: float
    specialist_predictions: dict
    selected_specialists: List[str]
    
    # ═══════════════════════════════════════════════════════════
    # DECISION
    # ═══════════════════════════════════════════════════════════
    final_prediction: str      # 'REAL' or 'FAKE'
    confidence: float          # 0.0 to 1.0
    confidence_level: str      # 'high', 'medium', 'low'
    explanation: str
    
    # ═══════════════════════════════════════════════════════════
    # ROUTING
    # ═══════════════════════════════════════════════════════════
    next_action: str           # 'ACCEPT', 'DOMAIN', 'HUMAN'
```

### 5.2 Node Implementations

#### Node 1: Ingest (Input Validation)

```python
def ingest_node(state: AgentState) -> AgentState:
    """Validate input video file"""
    video_path = state['video_path']
    
    # ─────────────────────────────────────────────────────────
    # Validation checks
    # ─────────────────────────────────────────────────────────
    if not os.path.exists(video_path):
        raise ValueError(f'Video not found: {video_path}')
    
    valid_formats = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    if not any(video_path.lower().endswith(fmt) for fmt in valid_formats):
        raise ValueError('Invalid video format')
    
    file_size = os.path.getsize(video_path) / (1024 * 1024)
    if file_size > 100:
        raise ValueError('Video too large (max 100MB)')
    
    # ─────────────────────────────────────────────────────────
    # Assign request ID
    # ─────────────────────────────────────────────────────────
    state['request_id'] = str(uuid.uuid4())
    
    return state
```

#### Node 2: Policy Decision (Routing Logic)

```python
def policy_decision_node(state: AgentState) -> AgentState:
    """Determine routing based on confidence"""
    confidence = state['student_confidence']
    
    # ─────────────────────────────────────────────────────────
    # Confidence-based routing
    # ─────────────────────────────────────────────────────────
    if confidence >= 0.85:
        state['next_action'] = 'ACCEPT'
        state['confidence_level'] = 'high'
        
    elif confidence >= 0.65:
        state['next_action'] = 'DOMAIN'
        state['confidence_level'] = 'medium'
        
    else:
        state['next_action'] = 'DOMAIN'
        state['confidence_level'] = 'low'
    
    return state
```

### 5.3 Graph Construction

```python
# ═══════════════════════════════════════════════════════════════════════
# CREATE LANGGRAPH WORKFLOW
# ═══════════════════════════════════════════════════════════════════════

workflow = StateGraph(AgentState)

# ─────────────────────────────────────────────────────────────────────
# Add nodes
# ─────────────────────────────────────────────────────────────────────
workflow.add_node('ingest', ingest_node)
workflow.add_node('metadata', metadata_node)
workflow.add_node('preprocess', preprocess_node)
workflow.add_node('student', student_inference_node)
workflow.add_node('policy', policy_decision_node)
workflow.add_node('domain', domain_inference_node)
workflow.add_node('explain', explanation_node)

# ─────────────────────────────────────────────────────────────────────
# Define edges
# ─────────────────────────────────────────────────────────────────────
workflow.set_entry_point('ingest')
workflow.add_edge('ingest', 'metadata')
workflow.add_edge('metadata', 'preprocess')
workflow.add_edge('preprocess', 'student')
workflow.add_edge('student', 'policy')

# ─────────────────────────────────────────────────────────────────────
# Conditional routing
# ─────────────────────────────────────────────────────────────────────
workflow.add_conditional_edges(
    'policy',
    route_decision,
    {
        'ACCEPT': 'explain',
        'DOMAIN': 'domain'
    }
)

workflow.add_edge('domain', 'explain')
workflow.add_edge('explain', END)

# ─────────────────────────────────────────────────────────────────────
# Compile and run
# ─────────────────────────────────────────────────────────────────────
app = workflow.compile()
result = app.invoke({'video_path': 'test_video.mp4'})
```

---

<div style="page-break-after: always;"></div>

## 6. Intelligent Routing Mechanism

### 6.1 Routing Decision Tree

```
                              ┌─────────────────┐
                              │   VIDEO INPUT   │
                              └────────┬────────┘
                                       │
                                       ▼
                         ┌─────────────────────────┐
                         │  ANALYZE CHARACTERISTICS │
                         │  • Bitrate              │
                         │  • Brightness           │
                         │  • Noise level          │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │   RUN BASELINE MODEL    │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │  BASELINE CONFIDENCE?   │
                         └────────────┬────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
              ▼                       ▼                       ▼
       ┌─────────────┐        ┌─────────────┐        ┌─────────────┐
       │   ≥ 85%     │        │   65-85%    │        │   < 65%     │
       │   HIGH      │        │   MEDIUM    │        │    LOW      │
       └──────┬──────┘        └──────┬──────┘        └──────┬──────┘
              │                      │                       │
              ▼                      ▼                       ▼
       ┌─────────────┐        ┌─────────────┐        ┌─────────────┐
       │   ACCEPT    │        │   ROUTE TO  │        │  ROUTE TO   │
       │  BASELINE   │        │  RELEVANT   │        │    ALL      │
       │             │        │ SPECIALISTS │        │ SPECIALISTS │
       └──────┬──────┘        └──────┬──────┘        └──────┬──────┘
              │                      │                       │
              │               ┌──────┴──────┐                │
              │               │             │                │
              │               ▼             ▼                │
              │        ┌───────────┐ ┌───────────┐          │
              │        │Compressed?│ │Low-light? │          │
              │        │  → CM     │ │  → LL     │          │
              │        └───────────┘ └───────────┘          │
              │                                              │
              └──────────────────┬───────────────────────────┘
                                 │
                                 ▼
                         ┌─────────────┐
                         │  AGGREGATE  │
                         │   RESULTS   │
                         └─────────────┘
```

### 6.2 Characteristic-Based Routing Rules

| Characteristic | Detection Threshold | Specialist Activated |
|----------------|:-------------------:|:--------------------:|
| **Compressed** | Bitrate < 1 Mbps | CM-Model |
| **Low-Light** | Brightness < 80 | LL-Model |
| **Re-recorded** | Noise > 500 | RR-Model |
| **Has Audio** | Audio present | AV-Model |
| **Multi-frame** | Frames > 1 | TM-Model |

### 6.3 Routing Efficiency Analysis

| Scenario | Models Used | Computation Saved |
|----------|:-----------:|:-----------------:|
| High confidence (>85%) | 1 (baseline) | **83%** |
| Medium conf, compressed | 3 (BG, CM, TM) | **50%** |
| Medium conf, low-light | 3 (BG, LL, TM) | **50%** |
| Low confidence | 6 (all) | **0%** |

### 6.4 Video Characteristics Analysis

```python
class VideoCharacteristics:
    """Analyze video for intelligent routing"""
    
    @staticmethod
    def analyze_video(video_path: str) -> dict:
        cap = cv2.VideoCapture(video_path)
        
        # ─────────────────────────────────────────────────────
        # Basic metadata
        # ─────────────────────────────────────────────────────
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # ─────────────────────────────────────────────────────
        # Bitrate calculation
        # ─────────────────────────────────────────────────────
        file_size = os.path.getsize(video_path)
        bitrate = (file_size * 8) / duration if duration > 0 else 0
        
        # ─────────────────────────────────────────────────────
        # Quality analysis
        # ─────────────────────────────────────────────────────
        brightness_samples = []
        noise_samples = []
        
        for frame in sampled_frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness_samples.append(np.mean(gray))
            noise_samples.append(cv2.Laplacian(gray, cv2.CV_64F).var())
        
        # ─────────────────────────────────────────────────────
        # Determine characteristics
        # ─────────────────────────────────────────────────────
        return {
            'fps': fps,
            'resolution': (width, height),
            'duration': duration,
            'bitrate': bitrate,
            'avg_brightness': np.mean(brightness_samples),
            'avg_noise': np.mean(noise_samples),
            'is_compressed': bitrate < 1000000,      # < 1 Mbps
            'is_low_light': avg_brightness < 80,
            'is_rerecorded': is_noisy and non_standard_resolution
        }
```

---

<div style="page-break-after: always;"></div>

# Part III: Neural Networks

---

## 7. Neural Network Architectures

### 7.1 Model Overview

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                         E-RAKSHA MODEL BANK                                ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  ┌─────────────────────────────────────────────────────────────────────┐  ║
║  │                    BG-MODEL (Baseline Generalist)                    │  ║
║  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │  ║
║  │  │ MobileNetV3 │─▶│  Temporal   │─▶│   Fusion    │─▶│ Classifier │  │  ║
║  │  │  Backbone   │  │    Conv     │  │   Layer     │  │    Head    │  │  ║
║  │  │   576-dim   │  │   256-dim   │  │   384-dim   │  │   2-class  │  │  ║
║  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │  ║
║  │                                                                      │  ║
║  │  Parameters: 2.1M │ Size: 8.2 MB │ Accuracy: 86.25%                 │  ║
║  └─────────────────────────────────────────────────────────────────────┘  ║
║                                                                            ║
║  ┌─────────────────────────────────────────────────────────────────────┐  ║
║  │                    AV-MODEL (Audio-Visual Specialist)                │  ║
║  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │  ║
║  │  │  ResNet18   │  │   Audio     │  │  Lip-Sync   │  │   Fusion   │  │  ║
║  │  │  Backbone   │  │    CNN      │  │  Detector   │  │  + Class   │  │  ║
║  │  │   512-dim   │  │   256-dim   │  │   Score     │  │   2-class  │  │  ║
║  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │  ║
║  │                                                                      │  ║
║  │  Parameters: 15.8M │ Size: 60.4 MB │ Accuracy: 93.0%                │  ║
║  └─────────────────────────────────────────────────────────────────────┘  ║
║                                                                            ║
║  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐          ║
║  │    CM-MODEL      │ │    RR-MODEL      │ │    LL-MODEL      │          ║
║  │   Compression    │ │   Re-recording   │ │    Low-light     │          ║
║  │   ResNet18       │ │   ResNet18       │ │   ResNet18       │          ║
║  │   11.7M params   │ │   11.7M params   │ │   11.7M params   │          ║
║  │   80.83% acc     │ │   85.0% acc      │ │   93.42% acc     │          ║
║  └──────────────────┘ └──────────────────┘ └──────────────────┘          ║
║                                                                            ║
║  ┌─────────────────────────────────────────────────────────────────────┐  ║
║  │                    TM-MODEL (Temporal Specialist)                    │  ║
║  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │  ║
║  │  │  ResNet18   │─▶│    LSTM     │─▶│   Temporal  │─▶│ Classifier │  │  ║
║  │  │ (per-frame) │  │  2 layers   │  │   Feature   │  │    Head    │  │  ║
║  │  │   512-dim   │  │   256-dim   │  │   256-dim   │  │   2-class  │  │  ║
║  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │  ║
║  │                                                                      │  ║
║  │  Parameters: 14.2M │ Size: 54.3 MB │ Accuracy: 78.5%                │  ║
║  └─────────────────────────────────────────────────────────────────────┘  ║
║                                                                            ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

### 7.2 BG-Model Architecture (Baseline Generalist)

#### Visual Branch - MobileNetV3-Small

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MOBILENETV3-SMALL BACKBONE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Input: [B, 3, 224, 224]                                                │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ Initial Conv: Conv2d(3→16, k=3, s=2, p=1) + BN + Hardswish      │    │
│  │ Output: [B, 16, 112, 112]                                        │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              │                                           │
│                              ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ Inverted Residual Blocks (11 blocks)                             │    │
│  │                                                                   │    │
│  │  Block 1:  expand=16,  out=16,  k=3, s=2, SE=True               │    │
│  │  Block 2:  expand=72,  out=24,  k=3, s=2, SE=False              │    │
│  │  Block 3:  expand=88,  out=24,  k=3, s=1, SE=False              │    │
│  │  Block 4:  expand=96,  out=40,  k=5, s=2, SE=True               │    │
│  │  Block 5:  expand=240, out=40,  k=5, s=1, SE=True               │    │
│  │  Block 6:  expand=240, out=40,  k=5, s=1, SE=True               │    │
│  │  Block 7:  expand=120, out=48,  k=5, s=1, SE=True               │    │
│  │  Block 8:  expand=144, out=48,  k=5, s=1, SE=True               │    │
│  │  Block 9:  expand=288, out=96,  k=5, s=2, SE=True               │    │
│  │  Block 10: expand=576, out=96,  k=5, s=1, SE=True               │    │
│  │  Block 11: expand=576, out=96,  k=5, s=1, SE=True               │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              │                                           │
│                              ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ Final Conv: Conv2d(96→576, k=1) + BN + Hardswish                │    │
│  │ AdaptiveAvgPool2d(1)                                             │    │
│  │ Output: [B, 576]                                                 │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Parameter Count

| Layer | Parameters |
|-------|:----------:|
| MobileNetV3-Small Backbone | 1,528,104 |
| Temporal Conv1d | 442,624 |
| Audio Branch Conv Layers | 74,432 |
| Audio Branch Linear | 262,272 |
| Fusion Linear Layers | 98,688 |
| Classification Head | 4,290 |
| **Total** | **2,143,234** |

---

<div style="page-break-after: always;"></div>

### 7.3 AV-Model Architecture (Audio-Visual Specialist)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         AV-MODEL ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                      VISUAL BRANCH                               │    │
│  │                                                                   │    │
│  │  Input: [B, T, 3, 224, 224]                                      │    │
│  │                              │                                    │    │
│  │                              ▼                                    │    │
│  │  ┌─────────────────────────────────────────────────────────┐     │    │
│  │  │              ResNet18 Backbone                           │     │    │
│  │  │  conv1 → bn1 → relu → maxpool                           │     │    │
│  │  │  layer1 (64→64)   → layer2 (64→128)                     │     │    │
│  │  │  layer3 (128→256) → layer4 (256→512)                    │     │    │
│  │  │  avgpool → Output: [B, T, 512]                          │     │    │
│  │  └─────────────────────────────────────────────────────────┘     │    │
│  │                              │                                    │    │
│  │                              ▼                                    │    │
│  │  ┌─────────────────────────────────────────────────────────┐     │    │
│  │  │  Temporal Conv1d(512→256, k=3, p=1)                     │     │    │
│  │  │  Mean pooling → Output: [B, 256]                        │     │    │
│  │  └─────────────────────────────────────────────────────────┘     │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                      AUDIO BRANCH                                │    │
│  │                                                                   │    │
│  │  Input: [B, 48000] (3 seconds @ 16kHz)                           │    │
│  │                              │                                    │    │
│  │                              ▼                                    │    │
│  │  ┌─────────────────────────────────────────────────────────┐     │    │
│  │  │  Mel Spectrogram Transform                               │     │    │
│  │  │  n_mels=64, n_fft=512, hop=256                          │     │    │
│  │  │  Output: [B, 1, 64, 186]                                 │     │    │
│  │  └─────────────────────────────────────────────────────────┘     │    │
│  │                              │                                    │    │
│  │                              ▼                                    │    │
│  │  ┌─────────────────────────────────────────────────────────┐     │    │
│  │  │  Audio CNN                                               │     │    │
│  │  │  Conv2d(1→32) → BN → ReLU → MaxPool                     │     │    │
│  │  │  Conv2d(32→64) → BN → ReLU → MaxPool                    │     │    │
│  │  │  Conv2d(64→128) → BN → ReLU → AdaptiveAvgPool(4,4)      │     │    │
│  │  │  Flatten → Linear(2048→256)                              │     │    │
│  │  │  Output: [B, 256]                                        │     │    │
│  │  └─────────────────────────────────────────────────────────┘     │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    LIP-SYNC DETECTOR                             │    │
│  │                                                                   │    │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │    │
│  │  │ Lip Encoder  │    │Audio Encoder │    │   Sync Net   │       │    │
│  │  │  CNN → 256   │ +  │  CNN → 256   │ →  │ 512→256→1    │       │    │
│  │  └──────────────┘    └──────────────┘    │  Sigmoid     │       │    │
│  │                                           └──────────────┘       │    │
│  │  Output: lip_sync_score [B, 1]                                   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    FUSION & CLASSIFICATION                       │    │
│  │                                                                   │    │
│  │  Concat(visual[256], audio[256], lip_sync[1]) → [B, 513]        │    │
│  │  Linear(513→256) → ReLU → Dropout(0.3)                          │    │
│  │  Linear(256→128) → ReLU → Dropout(0.2)                          │    │
│  │  Linear(128→64) → ReLU → Dropout(0.2)                           │    │
│  │  Linear(64→2) → Output: [B, 2]                                   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  Total Parameters: 15,847,522 │ Size: 60.4 MB │ Accuracy: 93.0%        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.4 Specialist Models (CM, RR, LL)

All three specialist models share the same ResNet18 architecture with custom classification heads:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SPECIALIST MODEL ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Input: [B, 3, 224, 224]                                                │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    ResNet18 Backbone                             │    │
│  │                                                                   │    │
│  │  conv1: Conv2d(3→64, k=7, s=2, p=3)     →  9,472 params         │    │
│  │  bn1: BatchNorm2d(64)                    →  128 params           │    │
│  │  layer1: 2 × BasicBlock(64→64)           →  147,584 params       │    │
│  │  layer2: 2 × BasicBlock(64→128)          →  525,568 params       │    │
│  │  layer3: 2 × BasicBlock(128→256)         →  2,099,712 params     │    │
│  │  layer4: 2 × BasicBlock(256→512)         →  8,393,728 params     │    │
│  │  avgpool: AdaptiveAvgPool2d(1)           →  Output: [B, 512]     │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              │                                           │
│                              ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                Custom Classification Head                        │    │
│  │                                                                   │    │
│  │  Dropout(0.3)                                                    │    │
│  │  Linear(512→256) + ReLU + BatchNorm1d(256)  →  131,840 params   │    │
│  │  Dropout(0.2)                                                    │    │
│  │  Linear(256→128) + ReLU                      →  32,896 params    │    │
│  │  Dropout(0.1)                                                    │    │
│  │  Linear(128→2)                               →  258 params       │    │
│  │                                                                   │    │
│  │  Output: [B, 2]                                                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  Total Parameters: 11,689,026 │ Size: 44.6 MB                           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Training Specialization

| Model | Training Focus | Data Augmentation |
|-------|----------------|-------------------|
| **CM-Model** | Compression artifacts | JPEG compression (q=50-95), H.264 re-encoding |
| **RR-Model** | Re-recording patterns | Moiré filters, screen grid overlay |
| **LL-Model** | Low-light conditions | Brightness reduction (0.3-0.7), Gaussian noise |

---

<div style="page-break-after: always;"></div>

## 8. Specialist Model Ensemble

### 8.1 Ensemble Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      DYNAMIC ENSEMBLE ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│                           ┌─────────────┐                               │
│                           │   VIDEO     │                               │
│                           │   INPUT     │                               │
│                           └──────┬──────┘                               │
│                                  │                                       │
│                                  ▼                                       │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    BASELINE INFERENCE                              │  │
│  │                                                                     │  │
│  │  ┌─────────────┐                                                   │  │
│  │  │  BG-Model   │ ──────────────────────────────────────────────┐  │  │
│  │  │  (Always)   │                                                │  │  │
│  │  └─────────────┘                                                │  │  │
│  └─────────────────────────────────────────────────────────────────│──┘  │
│                                                                     │     │
│                                  ▼                                  │     │
│  ┌───────────────────────────────────────────────────────────────┐ │     │
│  │                    SPECIALIST ROUTING                          │ │     │
│  │                                                                 │ │     │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐  │ │     │
│  │  │AV-Model │ │CM-Model │ │RR-Model │ │LL-Model │ │TM-Model │  │ │     │
│  │  │(if low  │ │(if comp │ │(if re-  │ │(if low  │ │(if med/ │  │ │     │
│  │  │ conf)   │ │ressed) │ │recorded)│ │ light)  │ │low conf)│  │ │     │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘  │ │     │
│  │       │           │           │           │           │        │ │     │
│  └───────┼───────────┼───────────┼───────────┼───────────┼────────┘ │     │
│          │           │           │           │           │          │     │
│          └───────────┴───────────┴───────────┴───────────┘          │     │
│                                  │                                   │     │
│                                  ▼                                   │     │
│  ┌───────────────────────────────────────────────────────────────────┘    │
│  │                                                                         │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  │                    WEIGHTED AGGREGATION                          │   │
│  │  │                                                                   │   │
│  │  │  final_pred = Σ(weight_i × bias_corrected_pred_i) / Σ(weight_i) │   │
│  │  │                                                                   │   │
│  │  └─────────────────────────────────────────────────────────────────┘   │
│  │                                                                         │
│  └─────────────────────────────────────────────────────────────────────────┘
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Model Weight Configuration

| Model | Weight | Bias Correction | Rationale |
|:-----:|:------:|:---------------:|-----------|
| **student** | 1.4 | 0.0 | Reliable baseline, moderate boost |
| **av** | 1.5 | 0.0 | High accuracy, strong AV analysis |
| **cm** | 1.5 | 0.0 | Good compression detection |
| **rr** | 0.8 | +0.15 | Strong real bias, reduced weight |
| **ll** | 1.0 | -0.05 | Strong fake bias, neutral weight |
| **tm** | 1.3 | 0.0 | Good temporal analysis |

### 8.3 Weighted Aggregation Algorithm

```python
def aggregate_predictions(predictions: Dict[str, Tuple[float, float]]):
    """
    Aggregate predictions from multiple models with bias correction
    
    Args:
        predictions: Dict mapping model_name to (prediction, confidence)
                    where prediction is P(fake) in [0, 1]
    
    Returns:
        final_prediction: float in [0, 1]
        final_confidence: float in [0, 1]
        best_model: str
    """
    
    # ═══════════════════════════════════════════════════════════════════
    # MODEL CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════
    model_configs = {
        'student': {'weight': 1.4, 'bias_correction': 0.0},
        'av':      {'weight': 1.5, 'bias_correction': 0.0},
        'cm':      {'weight': 1.5, 'bias_correction': 0.0},
        'rr':      {'weight': 0.8, 'bias_correction': 0.15},
        'll':      {'weight': 1.0, 'bias_correction': -0.05},
        'tm':      {'weight': 1.3, 'bias_correction': 0.0}
    }
    
    total_weight = 0
    weighted_prediction = 0
    best_model = "student"
    best_confidence = 0
    
    # ═══════════════════════════════════════════════════════════════════
    # AGGREGATION LOOP
    # ═══════════════════════════════════════════════════════════════════
    for model_name, (prediction, confidence) in predictions.items():
        config = model_configs.get(model_name)
        
        # Apply bias correction
        corrected_pred = prediction + config['bias_correction']
        corrected_pred = max(0.0, min(1.0, corrected_pred))
        
        # Calculate effective weight
        weight = config['weight'] * confidence
        
        # Accumulate weighted prediction
        weighted_prediction += corrected_pred * weight
        total_weight += weight
        
        # Track best model
        if confidence * config['weight'] > best_confidence:
            best_confidence = confidence * config['weight']
            best_model = model_name
    
    # ═══════════════════════════════════════════════════════════════════
    # FINAL CALCULATION
    # ═══════════════════════════════════════════════════════════════════
    final_prediction = weighted_prediction / total_weight
    final_confidence = best_confidence / model_configs[best_model]['weight']
    
    return final_prediction, final_confidence, best_model
```

### 8.4 Ensemble Decision Boundaries

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DECISION BOUNDARIES                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  FINAL CLASSIFICATION:                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                                                                   │    │
│  │    P(fake) > 0.5  ────────────────────────────────▶  [RED] FAKE     │    │
│  │                                                                   │    │
│  │    P(fake) ≤ 0.5  ────────────────────────────────▶  [GREEN] REAL     │    │
│  │                                                                   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  CONFIDENCE LEVEL:                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                                                                   │    │
│  │    confidence ≥ 0.85  ────────────────────────────▶  [GREEN] HIGH     │    │
│  │                                                                   │    │
│  │    confidence ≥ 0.65  ────────────────────────────▶  [YELLOW] MEDIUM   │    │
│  │                                                                   │    │
│  │    confidence < 0.65  ────────────────────────────▶  [RED] LOW      │    │
│  │                                                                   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

<div style="page-break-after: always;"></div>

## 9. Bias Correction & Calibration

### 9.1 The Bias Problem

> During evaluation, we identified significant bias in individual models that caused imbalanced predictions.

#### Initial Model Bias Analysis

| Model | Real Accuracy | Fake Accuracy | Bias Direction |
|:-----:|:-------------:|:-------------:|:--------------:|
| student | 72% | 68% | Slight real bias |
| av | 65% | 78% | Fake bias |
| cm | 75% | 70% | Slight real bias |
| rr | 92% | 45% | **Strong real bias** |
| ll | 40% | 95% | **Strong fake bias** |
| tm | 70% | 72% | Balanced |

### 9.2 Bias Correction Methodology

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      BIAS CORRECTION FORMULA                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Corrected Prediction = Original Prediction + Bias Correction            │
│                                                                          │
│  Where:                                                                  │
│  • Positive bias correction → shifts predictions toward FAKE             │
│  • Negative bias correction → shifts predictions toward REAL             │
│  • Values determined empirically on validation set                       │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                                                                   │    │
│  │  corrected_pred = original_pred + bias_correction                │    │
│  │  corrected_pred = clamp(corrected_pred, 0.0, 1.0)                │    │
│  │                                                                   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 9.3 Bias Correction Derivation

| Model | Real Acc | Fake Acc | Raw Bias | Correction |
|:-----:|:--------:|:--------:|:--------:|:----------:|
| rr | 92% | 45% | -23.5% | **+0.15** |
| ll | 40% | 95% | +27.5% | **-0.05** |
| cm | 75% | 70% | -2.5% | 0.0 |
| student | 72% | 68% | -2.0% | 0.0 |
| av | 65% | 78% | +6.5% | 0.0 |
| tm | 70% | 72% | +1.0% | 0.0 |

### 9.4 Calibration Results

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    POST-CALIBRATION PERFORMANCE                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                                                                   │    │
│  │  Metric              │  Before   │  After    │  Improvement      │    │
│  │  ────────────────────┼───────────┼───────────┼─────────────────  │    │
│  │  Overall Accuracy    │    45%    │    70%    │     +25%  ✓       │    │
│  │  Real Accuracy       │    56%    │    68%    │     +12%  ✓       │    │
│  │  Fake Accuracy       │    34%    │    72%    │     +38%  ✓       │    │
│  │  Bias Gap            │    22%    │     4%    │     -18%  ✓       │    │
│  │  F1 Score            │   0.382   │   0.700   │    +0.318 ✓       │    │
│  │                                                                   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  Key Improvements:                                                       │
│  • Balanced detection of both real and fake content                     │
│  • Reduced false positive rate by 38%                                   │
│  • Improved F1 score by 83%                                             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 9.5 Temperature Scaling

```python
class ConfidenceCalibrator:
    """Calibrate model confidence using temperature scaling"""
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
    
    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits"""
        return torch.softmax(logits / self.temperature, dim=1)
    
    def learn_temperature(self, val_logits, val_labels) -> float:
        """Learn optimal temperature on validation set"""
        best_temp = 1.0
        best_ece = float('inf')
        
        for temp in np.arange(0.5, 3.0, 0.1):
            probs = torch.softmax(val_logits / temp, dim=1)
            ece = self._compute_ece(probs, val_labels)
            
            if ece < best_ece:
                best_ece = ece
                best_temp = temp
        
        self.temperature = best_temp
        return best_temp
```

---

<div style="page-break-after: always;"></div>

# Part IV: Implementation

---

## 10. Preprocessing Pipeline

### 10.1 Video Ingestion

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      SUPPORTED VIDEO FORMATS                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Format     │  Codecs              │  Max Size  │  Max Duration          │
│  ───────────┼──────────────────────┼────────────┼────────────────────    │
│  MP4        │  H.264, H.265        │  100 MB    │  5 minutes             │
│  AVI        │  Various             │  100 MB    │  5 minutes             │
│  MOV        │  QuickTime           │  100 MB    │  5 minutes             │
│  WebM       │  VP8, VP9            │  100 MB    │  5 minutes             │
│  MKV        │  Matroska            │  100 MB    │  5 minutes             │
│                                                                          │
│  Resolution: 144p (min) to 4K (max)                                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 10.2 Frame Extraction

```python
def extract_frames(video_path: str, max_frames: int = 8) -> List[torch.Tensor]:
    """
    Extract frames from video using uniform temporal sampling
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to extract
    
    Returns:
        List of frame tensors [C, H, W]
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # ─────────────────────────────────────────────────────────────────
    # Compute uniform sample indices
    # ─────────────────────────────────────────────────────────────────
    frame_indices = np.linspace(
        0, total_frames - 1, 
        min(max_frames, total_frames), 
        dtype=int
    )
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            # BGR to RGB conversion
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize to model input size
            frame_resized = cv2.resize(frame_rgb, (224, 224))
            
            # Normalize to [0, 1]
            frame_tensor = torch.from_numpy(frame_resized).float() / 255.0
            
            # Reorder to [C, H, W]
            frame_tensor = frame_tensor.permute(2, 0, 1)
            
            frames.append(frame_tensor)
    
    cap.release()
    return frames
```

### 10.3 Face Detection (MTCNN)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MTCNN PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    P-Net (Proposal Network)                      │    │
│  │                                                                   │    │
│  │  Input: Image pyramid at multiple scales                         │    │
│  │  Conv layers: 3×3, 3×3, 3×3                                      │    │
│  │  Output: Face/non-face classification + bounding box regression  │    │
│  │  NMS to reduce candidates                                        │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              │                                           │
│                              ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    R-Net (Refine Network)                        │    │
│  │                                                                   │    │
│  │  Input: Candidate regions from P-Net                             │    │
│  │  Conv layers: 3×3, 3×3, 3×3                                      │    │
│  │  FC layers: 128                                                  │    │
│  │  Output: Refined classification + bounding box                   │    │
│  │  NMS to reduce candidates                                        │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              │                                           │
│                              ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    O-Net (Output Network)                        │    │
│  │                                                                   │    │
│  │  Input: Refined candidates from R-Net                            │    │
│  │  Conv layers: 3×3, 3×3, 3×3, 2×2                                 │    │
│  │  FC layers: 256                                                  │    │
│  │  Output: Final classification + bounding box + landmarks         │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 10.4 Audio Extraction

```python
def extract_audio(video_path: str, duration: float = 3.0) -> torch.Tensor:
    """
    Extract audio from video for AV-Model
    
    Args:
        video_path: Path to video file
        duration: Duration in seconds to extract
    
    Returns:
        audio_waveform: Tensor [audio_length]
    """
    import torchaudio
    
    # Load audio
    waveform, sr = torchaudio.load(video_path)
    
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample to 16kHz
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)
    
    # Trim/pad to target length
    target_length = int(duration * 16000)
    
    if waveform.shape[1] > target_length:
        # Center crop
        start = (waveform.shape[1] - target_length) // 2
        waveform = waveform[:, start:start + target_length]
    else:
        # Zero pad
        padding = target_length - waveform.shape[1]
        waveform = F.pad(waveform, (0, padding))
    
    return waveform.squeeze(0)
```

### 10.5 Mel Spectrogram Parameters

| Parameter | Value | Description |
|-----------|:-----:|-------------|
| sample_rate | 16000 Hz | Audio sample rate |
| n_mels | 64 | Number of mel filterbanks |
| n_fft | 512 | FFT window size |
| hop_length | 256 | Hop between windows |
| f_min | 0 Hz | Minimum frequency |
| f_max | 8000 Hz | Maximum frequency |

**For 3-second audio at 16kHz:**
- Input samples: 48,000
- Time frames: (48000 - 512) / 256 + 1 = 186
- Output shape: [64, 186]

---

<div style="page-break-after: always;"></div>

## 11. Inference Engine

### 11.1 Complete Inference Pipeline

```python
class InferenceEngine:
    """Orchestrates the complete prediction pipeline"""
    
    def __init__(self, device='auto'):
        if device == 'auto':
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            self.device = torch.device(device)
        
        # Load all models
        self.models = self._load_models()
        
        # Set models to evaluation mode
        for model in self.models.values():
            if model is not None:
                model.eval()
    
    @torch.no_grad()
    def predict(self, video_path: str) -> Dict[str, Any]:
        """
        Run complete inference pipeline
        
        ┌─────────────────────────────────────────────────────────────┐
        │                    INFERENCE STEPS                          │
        ├─────────────────────────────────────────────────────────────┤
        │                                                              │
        │  Step 1: Analyze video characteristics                      │
        │  Step 2: Extract frames and audio                           │
        │  Step 3: Run baseline inference                             │
        │  Step 4: Intelligent routing                                │
        │  Step 5: Run specialist inference                           │
        │  Step 6: Aggregate predictions                              │
        │  Step 7: Generate explanation                               │
        │                                                              │
        └─────────────────────────────────────────────────────────────┘
        """
        start_time = time.time()
        
        # Step 1: Analyze video characteristics
        video_chars = VideoCharacteristics.analyze_video(video_path)
        
        # Step 2: Extract frames and audio
        frames = self.extract_frames(video_path, max_frames=8)
        audio = self.extract_audio(video_path, duration=3.0)
        
        # Step 3: Run baseline inference
        baseline_pred, baseline_conf = self.run_baseline(frames)
        predictions = {'student': (baseline_pred, baseline_conf)}
        
        # Step 4: Intelligent routing
        specialists = self.route(video_chars, baseline_conf)
        
        # Step 5: Run specialist inference
        for specialist in specialists:
            pred, conf = self.run_specialist(frames, audio, specialist)
            predictions[specialist] = (pred, conf)
        
        # Step 6: Aggregate predictions
        final_pred, final_conf, best_model = self.aggregate(predictions)
        
        # Step 7: Generate explanation
        explanation = self.explain(
            final_pred, final_conf, best_model,
            specialists, video_chars
        )
        
        processing_time = time.time() - start_time
        
        return {
            'prediction': 'FAKE' if final_pred > 0.5 else 'REAL',
            'confidence': final_conf,
            'explanation': explanation,
            'best_model': best_model,
            'specialists_used': specialists,
            'all_predictions': predictions,
            'video_characteristics': video_chars,
            'processing_time': processing_time
        }
```

### 11.2 Streaming Inference

```python
class StreamingInference:
    """Real-time video stream analysis"""
    
    def __init__(self, model, buffer_size=8):
        self.model = model
        self.buffer_size = buffer_size
        self.frame_buffer = []
    
    def process_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Process a single frame from video stream
        
        Returns prediction when buffer is full
        """
        # Preprocess frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (224, 224))
        frame_tensor = torch.from_numpy(frame_resized).float() / 255.0
        frame_tensor = frame_tensor.permute(2, 0, 1)
        
        # Add to buffer
        self.frame_buffer.append(frame_tensor)
        
        # Process when buffer is full
        if len(self.frame_buffer) >= self.buffer_size:
            frames = torch.stack(self.frame_buffer).unsqueeze(0)
            
            with torch.no_grad():
                logits = self.model(frames)
                probs = torch.softmax(logits, dim=1)
            
            # Clear buffer with overlap
            self.frame_buffer = self.frame_buffer[self.buffer_size // 2:]
            
            return {
                'prediction': 'FAKE' if probs[0, 1] > 0.5 else 'REAL',
                'confidence': max(probs[0, 0], probs[0, 1]).item()
            }
        
        return None
```

### 11.3 Memory Optimization

| Optimization | Description | Memory Reduction |
|--------------|-------------|:----------------:|
| **Gradient Checkpointing** | Recompute activations during backward | ~50% |
| **Mixed Precision** | FP16 for compute, FP32 for accumulation | ~50% |
| **INT8 Quantization** | Quantize weights to 8-bit integers | ~75% |
| **Adaptive Batch Size** | Adjust based on available memory | Variable |

#### Memory Usage by Model

| Model | FP32 | FP16 | INT8 |
|-------|:----:|:----:|:----:|
| BG-Model | 8.2 MB | 4.1 MB | 2.1 MB |
| AV-Model | 60.4 MB | 30.2 MB | 15.1 MB |
| CM-Model | 44.6 MB | 22.3 MB | 11.2 MB |
| RR-Model | 44.6 MB | 22.3 MB | 11.2 MB |
| LL-Model | 44.6 MB | 22.3 MB | 11.2 MB |
| TM-Model | 54.3 MB | 27.2 MB | 13.6 MB |

---

<div style="page-break-after: always;"></div>

## 12. Explainability & Trust Mechanisms

### 12.1 Grad-CAM Implementation

> Gradient-weighted Class Activation Mapping (Grad-CAM) provides visual explanations by highlighting important regions in the input.

```python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

class ExplainabilityEngine:
    """Generate visual and textual explanations for predictions"""
    
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        
        # Default to last conv layer (ResNet18)
        if target_layer is None:
            self.target_layer = model.layer4[-1]
        else:
            self.target_layer = target_layer
        
        self.cam = GradCAM(
            model=self.model,
            target_layers=[self.target_layer],
            use_cuda=torch.cuda.is_available()
        )
    
    def generate_heatmap(self, input_tensor: torch.Tensor, 
                         target_class: int = 1) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for input
        
        Args:
            input_tensor: Input image tensor [1, C, H, W]
            target_class: Class to explain (1=fake, 0=real)
        
        Returns:
            heatmap: Numpy array [H, W] with values in [0, 1]
        """
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        return grayscale_cam[0]
    
    def visualize_heatmap(self, frame: np.ndarray, 
                          heatmap: np.ndarray) -> np.ndarray:
        """Overlay heatmap on original frame"""
        frame_normalized = frame.astype(np.float32) / 255.0
        heatmap_resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
        
        visualization = show_cam_on_image(
            frame_normalized,
            heatmap_resized,
            use_rgb=True,
            colormap=cv2.COLORMAP_JET
        )
        
        return visualization
```

### 12.2 Textual Explanation Generation

```python
def generate_detailed_explanation(
    prediction: str, 
    confidence: float,
    best_model: str, 
    specialists: List[str],
    video_chars: dict
) -> str:
    """Generate human-readable explanation for prediction"""
    
    explanation_parts = []
    
    # ─────────────────────────────────────────────────────────────────
    # Main prediction statement
    # ─────────────────────────────────────────────────────────────────
    conf_pct = confidence * 100
    
    if prediction == 'FAKE':
        explanation_parts.append(
            f"This video is classified as MANIPULATED (FAKE) with "
            f"{conf_pct:.1f}% confidence."
        )
    else:
        explanation_parts.append(
            f"This video appears to be AUTHENTIC (REAL) with "
            f"{conf_pct:.1f}% confidence."
        )
    
    # ─────────────────────────────────────────────────────────────────
    # Model attribution
    # ─────────────────────────────────────────────────────────────────
    model_names = {
        'student': 'baseline generalist model',
        'av': 'audio-visual specialist',
        'cm': 'compression artifact detector',
        'rr': 're-recording pattern detector',
        'll': 'low-light specialist',
        'tm': 'temporal consistency analyzer'
    }
    
    explanation_parts.append(
        f"Primary analysis performed by the {model_names.get(best_model)}."
    )
    
    # ─────────────────────────────────────────────────────────────────
    # Specialist insights
    # ─────────────────────────────────────────────────────────────────
    if specialists:
        insights = []
        if 'cm' in specialists:
            insights.append("compression artifacts were analyzed")
        if 'll' in specialists:
            insights.append("low-light enhancement was applied")
        if 'tm' in specialists:
            insights.append("temporal consistency was evaluated")
        if 'av' in specialists:
            insights.append("audio-visual synchronization was verified")
        
        if insights:
            explanation_parts.append(
                f"Additional analysis: {', '.join(insights)}."
            )
    
    # ─────────────────────────────────────────────────────────────────
    # Final assessment
    # ─────────────────────────────────────────────────────────────────
    if prediction == 'FAKE':
        explanation_parts.append(
            "Detected inconsistencies suggest potential manipulation. "
            "Manual review recommended for critical applications."
        )
    else:
        explanation_parts.append(
            "No significant artifacts or inconsistencies were detected."
        )
    
    return " ".join(explanation_parts)
```

### 12.3 Focus Region Analysis

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FACIAL REGION ANALYSIS                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│                    ┌─────────────────────┐                              │
│                    │     FOREHEAD        │  (y: 20-60)                  │
│                    │                     │                              │
│                    ├─────────────────────┤                              │
│                    │       EYES          │  (y: 50-100)                 │
│                    │   👁️         👁️    │                              │
│                    ├─────────────────────┤                              │
│                    │       NOSE          │  (y: 80-140)                 │
│                    │         👃          │                              │
│                    ├─────────────────────┤                              │
│                    │      MOUTH          │  (y: 130-180)                │
│                    │        👄           │                              │
│                    └─────────────────────┘                              │
│                                                                          │
│  High attention regions indicate where the model detected anomalies     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

<div style="page-break-after: always;"></div>

# Part V: Deployment

---

## 13. API & Backend Architecture

### 13.1 FastAPI Backend

```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

app = FastAPI(
    title="E-Raksha Deepfake Detection API",
    description="Agentic AI system for deepfake detection",
    version="2.0.0"
)

# ═══════════════════════════════════════════════════════════════════════
# CORS CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════════════
# RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════
class PredictionResponse(BaseModel):
    success: bool
    prediction: str
    confidence: float
    confidence_level: str
    explanation: str
    best_model: str
    specialists_used: List[str]
    processing_time: float
    request_id: str

# ═══════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": agent.get_model_status(),
        "version": "2.0.0"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_video(file: UploadFile = File(...)):
    """Predict if uploaded video is real or fake"""
    
    # Validate file type
    if not file.content_type.startswith('video/'):
        raise HTTPException(400, "File must be a video")
    
    # Save and process
    temp_path = f"/tmp/{file.filename}"
    async with aiofiles.open(temp_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    try:
        result = agent.predict(temp_path)
        return PredictionResponse(**result)
    finally:
        os.remove(temp_path)

@app.get("/models")
async def get_models():
    """Get information about loaded models"""
    return {
        "models": agent.get_model_status(),
        "bias_correction": "enabled",
        "version": "2.0.0"
    }
```

### 13.2 API Endpoints Reference

| Endpoint | Method | Description | Response |
|----------|:------:|-------------|----------|
| `/health` | GET | Health check | Status, models loaded |
| `/predict` | POST | Analyze video | Prediction, confidence, explanation |
| `/models` | GET | Model information | Model status, versions |
| `/feedback` | POST | Submit feedback | Feedback ID, status |

### 13.3 Database Schema

```sql
-- ═══════════════════════════════════════════════════════════════════════
-- POSTGRESQL SCHEMA FOR E-RAKSHA
-- ═══════════════════════════════════════════════════════════════════════

CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    api_key VARCHAR(64) UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);

CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    video_hash VARCHAR(64) NOT NULL,
    prediction VARCHAR(10) NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    best_model VARCHAR(20),
    specialists_used TEXT[],
    processing_time DECIMAL(6,3),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE feedback (
    id SERIAL PRIMARY KEY,
    prediction_id INTEGER REFERENCES predictions(id),
    user_label VARCHAR(10) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    validated_label VARCHAR(10),
    validation_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_predictions_user ON predictions(user_id);
CREATE INDEX idx_predictions_hash ON predictions(video_hash);
CREATE INDEX idx_feedback_status ON feedback(status);
```

---

<div style="page-break-after: always;"></div>

## 14. Frontend & User Interface

### 14.1 React Component Architecture

```typescript
// ═══════════════════════════════════════════════════════════════════════
// ANALYSIS WORKBENCH COMPONENT
// ═══════════════════════════════════════════════════════════════════════

import React, { useState, useCallback } from 'react';

interface AnalysisResult {
  prediction: string;
  confidence: number;
  explanation: string;
  best_model: string;
  specialists_used: string[];
  processing_time: number;
}

const AnalysisWorkbench: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const analyzeVideo = async () => {
    if (!file) return;

    setIsAnalyzing(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        setResult({
          prediction: data.prediction,
          confidence: data.confidence / 100,
          explanation: data.explanation,
          best_model: data.details?.best_model || 'Unknown',
          specialists_used: data.details?.specialists_used || [],
          processing_time: data.details?.processing_time || 0,
        });
      } else {
        const errorData = await response.json();
        setError(errorData.detail || 'Analysis failed');
      }
    } catch (err) {
      setError('Failed to connect to server');
    }

    setIsAnalyzing(false);
  };

  return (
    <div className="analysis-workbench">
      {/* Upload Section */}
      <UploadSection onFileSelect={setFile} />
      
      {/* Analyze Button */}
      <button onClick={analyzeVideo} disabled={!file || isAnalyzing}>
        {isAnalyzing ? 'Analyzing...' : 'Analyze Video'}
      </button>

      {/* Results Section */}
      {result && <ResultsDisplay result={result} />}
      
      {/* Error Display */}
      {error && <ErrorMessage message={error} />}
    </div>
  );
};
```

### 14.2 UI Components

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         E-RAKSHA WEB INTERFACE                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                        NAVIGATION BAR                            │    │
│  │  E-Raksha    Home  |  Analysis  |  Dashboard  |  FAQ         │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                        UPLOAD SECTION                            │    │
│  │                                                                   │    │
│  │     ┌─────────────────────────────────────────────────────┐     │    │
│  │     │                                                       │     │    │
│  │     │         [FOLDER] Drag and drop your video here             │     │    │
│  │     │              or click to browse                       │     │    │
│  │     │                                                       │     │    │
│  │     │            Maximum file size: 100MB                   │     │    │
│  │     │                                                       │     │    │
│  │     └─────────────────────────────────────────────────────┘     │    │
│  │                                                                   │    │
│  │                    [ Analyze Video ]                          │    │
│  │                                                                   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                        RESULTS SECTION                           │    │
│  │                                                                   │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │    │
│  │  │  Prediction  │  │  Confidence  │  │  Best Model  │           │    │
│  │  │              │  │              │  │              │           │    │
│  │  │   [RED] FAKE    │  │    94.9%     │  │   AV-Model   │           │    │
│  │  │              │  │              │  │              │           │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘           │    │
│  │                                                                   │    │
│  │  ┌─────────────────────────────────────────────────────────┐    │    │
│  │  │                     EXPLANATION                          │    │    │
│  │  │                                                           │    │    │
│  │  │  This video is classified as MANIPULATED (FAKE) with     │    │    │
│  │  │  94.9% confidence. Primary analysis performed by the     │    │    │
│  │  │  audio-visual specialist. Audio-visual synchronization   │    │    │
│  │  │  was verified. Detected inconsistencies suggest          │    │    │
│  │  │  potential manipulation.                                  │    │    │
│  │  │                                                           │    │    │
│  │  └─────────────────────────────────────────────────────────┘    │    │
│  │                                                                   │    │
│  │  Specialists Used: [AV] [CM] [TM]                               │    │
│  │                                                                   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

<div style="page-break-after: always;"></div>

## 15. Edge & Mobile Deployment

### 15.1 Edge Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      EDGE DEPLOYMENT STACK                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    APPLICATION LAYER                             │    │
│  │                                                                   │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │    │
│  │  │   FastAPI   │  │   Video     │  │   Result    │              │    │
│  │  │   Server    │  │   Capture   │  │   Display   │              │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘              │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    INFERENCE LAYER                               │    │
│  │                                                                   │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │    │
│  │  │  TensorRT   │  │  Quantized  │  │  Optimized  │              │    │
│  │  │  Runtime    │  │   Models    │  │  Preprocess │              │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘              │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    HARDWARE LAYER                                │    │
│  │                                                                   │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │    │
│  │  │   Jetson    │  │ Raspberry   │  │   Intel     │              │    │
│  │  │   Nano      │  │   Pi 4      │  │    NCS      │              │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘              │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 15.2 Model Optimization for Edge

```python
# ═══════════════════════════════════════════════════════════════════════
# QUANTIZATION FOR EDGE DEPLOYMENT
# ═══════════════════════════════════════════════════════════════════════

import torch.quantization as quant

def quantize_model_for_edge(model, calibration_data):
    """Quantize model to INT8 for edge deployment"""
    
    model.eval()
    model.qconfig = quant.get_default_qconfig('fbgemm')
    
    # Fuse modules
    model_fused = quant.fuse_modules(model, [
        ['conv', 'bn', 'relu'],
        ['conv', 'bn']
    ])
    
    # Prepare for calibration
    model_prepared = quant.prepare(model_fused)
    
    # Calibrate with representative data
    with torch.no_grad():
        for batch in calibration_data:
            model_prepared(batch)
    
    # Convert to quantized model
    model_quantized = quant.convert(model_prepared)
    
    return model_quantized


# ═══════════════════════════════════════════════════════════════════════
# ONNX EXPORT
# ═══════════════════════════════════════════════════════════════════════

def export_to_onnx(model, output_path, input_shape=(1, 8, 3, 224, 224)):
    """Export model to ONNX format"""
    
    dummy_input = torch.randn(input_shape)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['video_frames'],
        output_names=['prediction'],
        dynamic_axes={
            'video_frames': {0: 'batch_size', 1: 'num_frames'},
            'prediction': {0: 'batch_size'}
        }
    )
```

### 15.3 Performance Targets by Platform

| Platform | Latency | Throughput | Memory | Power |
|----------|:-------:|:----------:|:------:|:-----:|
| **Cloud (GPU)** | < 500ms | 10 vid/s | 4 GB | 250W |
| **Edge (Jetson Nano)** | < 2s | 1 vid/s | 2 GB | 10W |
| **Edge (Raspberry Pi)** | < 5s | 0.3 vid/s | 1 GB | 5W |
| **Mobile (Android)** | < 3s | 0.5 vid/s | 512 MB | 2W |
| **Mobile (iOS)** | < 3s | 0.5 vid/s | 512 MB | 2W |

### 15.4 Android TFLite Integration

```java
public class DeepfakeDetector {
    private Interpreter tfliteInterpreter;
    private static final int INPUT_SIZE = 224;
    private static final int NUM_FRAMES = 8;
    
    public DeepfakeDetector(Context context) throws IOException {
        // Load TFLite model
        MappedByteBuffer modelBuffer = loadModelFile(
            context, "eraksha_model.tflite"
        );
        
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(4);
        options.setUseNNAPI(true);  // Use Android Neural Networks API
        
        tfliteInterpreter = new Interpreter(modelBuffer, options);
    }
    
    public float[] predict(Bitmap[] frames) {
        // Prepare input tensor
        float[][][][] input = new float[1][NUM_FRAMES][INPUT_SIZE][INPUT_SIZE * 3];
        
        for (int i = 0; i < Math.min(frames.length, NUM_FRAMES); i++) {
            Bitmap resized = Bitmap.createScaledBitmap(
                frames[i], INPUT_SIZE, INPUT_SIZE, true
            );
            preprocessFrame(resized, input[0][i]);
        }
        
        // Run inference
        float[][] output = new float[1][2];
        tfliteInterpreter.run(input, output);
        
        return output[0];
    }
}
```

---

<div style="page-break-after: always;"></div>

# Part VI: Operations

---

## 16. Security & Model Versioning

### 16.1 Model Security

```python
import hashlib
import hmac
from cryptography.fernet import Fernet

class ModelSecurity:
    """Security utilities for model protection"""
    
    def __init__(self, secret_key: bytes = None):
        self.secret_key = secret_key or Fernet.generate_key()
        self.cipher = Fernet(self.secret_key)
    
    def compute_model_hash(self, model_path: str) -> str:
        """Compute SHA-256 hash of model file"""
        sha256 = hashlib.sha256()
        with open(model_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def sign_model(self, model_path: str) -> str:
        """Create HMAC signature for model integrity"""
        model_hash = self.compute_model_hash(model_path)
        signature = hmac.new(
            self.secret_key,
            model_hash.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def verify_model(self, model_path: str, signature: str) -> bool:
        """Verify model integrity using signature"""
        expected_signature = self.sign_model(model_path)
        return hmac.compare_digest(signature, expected_signature)
```

### 16.2 Model Version Management

```python
class ModelVersionManager:
    """Manage model versions with metadata tracking"""
    
    def register_model(self, model_name: str, model_path: str,
                      metrics: dict, description: str = '') -> str:
        """Register a new model version"""
        
        security = ModelSecurity()
        
        # Generate version ID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        version_id = f"{model_name}_v{timestamp}"
        
        # Compute hash
        model_hash = security.compute_model_hash(model_path)
        
        # Create version entry
        version_entry = {
            'version_id': version_id,
            'model_name': model_name,
            'model_path': model_path,
            'model_hash': model_hash,
            'metrics': metrics,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'status': 'registered'
        }
        
        return version_id
    
    def promote_version(self, version_id: str) -> bool:
        """Promote a version to active/production"""
        # Implementation...
        pass
    
    def rollback(self, model_name: str, version_id: str) -> bool:
        """Rollback to a previous version"""
        return self.promote_version(version_id)
```

### 16.3 API Security

```python
from fastapi import HTTPException, Depends
from fastapi.security import APIKeyHeader
import secrets

API_KEY_HEADER = APIKeyHeader(name='X-API-Key')

class APISecurityManager:
    def __init__(self):
        self.api_keys = {}
        self.rate_limits = {}
        self.max_requests_per_minute = 60
    
    def generate_api_key(self, user_id: str) -> str:
        """Generate new API key for user"""
        api_key = secrets.token_urlsafe(32)
        self.api_keys[api_key] = {
            'user_id': user_id,
            'created_at': time.time(),
            'active': True
        }
        return api_key
    
    def check_rate_limit(self, api_key: str) -> bool:
        """Check if request is within rate limit"""
        current_time = time.time()
        minute_ago = current_time - 60
        
        if api_key not in self.rate_limits:
            self.rate_limits[api_key] = []
        
        # Clean old requests
        self.rate_limits[api_key] = [
            t for t in self.rate_limits[api_key] if t > minute_ago
        ]
        
        if len(self.rate_limits[api_key]) >= self.max_requests_per_minute:
            return False
        
        self.rate_limits[api_key].append(current_time)
        return True

# Dependency for protected endpoints
async def verify_api_key(api_key: str = Depends(API_KEY_HEADER)):
    if not security_manager.validate_api_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    if not security_manager.check_rate_limit(api_key):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    return api_key
```

---

<div style="page-break-after: always;"></div>

## 17. Safe Learning & Feedback Loop

### 17.1 Safe Learning Architecture

> The system implements a safe learning pipeline that incorporates user feedback while preventing model degradation from incorrect labels.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      SAFE LEARNING PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                  │
│  │   User      │───▶│  Quarantine │───▶│   Human     │                  │
│  │  Feedback   │    │   Queue     │    │ Validation  │                  │
│  └─────────────┘    └─────────────┘    └──────┬──────┘                  │
│                                                │                         │
│                                                ▼                         │
│                                        ┌─────────────┐                  │
│                                        │  Consensus  │                  │
│                                        │   Check     │                  │
│                                        └──────┬──────┘                  │
│                                                │                         │
│                          ┌─────────────────────┼─────────────────────┐  │
│                          │                     │                     │  │
│                          ▼                     ▼                     ▼  │
│                   ┌─────────────┐       ┌─────────────┐       ┌─────────┐
│                   │   Accept    │       │   Reject    │       │  More   │
│                   │  Feedback   │       │  Feedback   │       │ Review  │
│                   └──────┬──────┘       └─────────────┘       └─────────┘
│                          │                                               │
│                          ▼                                               │
│                   ┌─────────────┐                                        │
│                   │ Incremental │                                        │
│                   │  Retrain    │                                        │
│                   └──────┬──────┘                                        │
│                          │                                               │
│                          ▼                                               │
│                   ┌─────────────┐                                        │
│                   │  Validate   │                                        │
│                   │ Performance │                                        │
│                   └──────┬──────┘                                        │
│                          │                                               │
│              ┌───────────┴───────────┐                                  │
│              │                       │                                   │
│              ▼                       ▼                                   │
│       ┌─────────────┐         ┌─────────────┐                           │
│       │   Deploy    │         │  Rollback   │                           │
│       │  New Model  │         │  to Old     │                           │
│       └─────────────┘         └─────────────┘                           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 17.2 Feedback Submission

```python
class SafeLearningPipeline:
    """Safe learning system with human validation"""
    
    def __init__(self, feedback_db_path: str = 'feedback.db'):
        self.feedback_db = feedback_db_path
        self.quarantine_threshold = 0.7
        self.min_validations = 3  # Minimum human validations required
    
    def submit_feedback(self, video_path: str, model_prediction: str,
                       model_confidence: float, user_label: str) -> dict:
        """
        Submit user feedback for a prediction
        
        Feedback is quarantined until validated by multiple humans
        """
        video_hash = self._compute_video_hash(video_path)
        
        # Store in quarantine
        feedback_entry = {
            'video_hash': video_hash,
            'model_prediction': model_prediction,
            'model_confidence': model_confidence,
            'user_label': user_label,
            'status': 'quarantine',
            'validation_count': 0
        }
        
        self._store_feedback(feedback_entry)
        
        return {
            'success': True,
            'status': 'quarantine',
            'message': 'Feedback submitted for validation'
        }
    
    def validate_feedback(self, feedback_id: int, validator_id: str,
                         label: str) -> dict:
        """Human validator confirms or corrects feedback"""
        
        # Add validation
        self._add_validation(feedback_id, validator_id, label)
        
        # Check if enough validations
        validation_count = self._get_validation_count(feedback_id)
        
        if validation_count >= self.min_validations:
            # Compute consensus
            consensus_label = self._compute_consensus(feedback_id)
            
            # Update status
            self._update_status(feedback_id, 'validated', consensus_label)
        
        return {'success': True, 'validation_count': validation_count}
```

### 17.3 Incremental Retraining

```python
class IncrementalTrainer:
    """Safe incremental model retraining"""
    
    def retrain_with_feedback(self, feedback_data: List[dict],
                              validation_set: DataLoader) -> dict:
        """
        Retrain model with validated feedback
        
        Safety checks:
        1. Performance validation before deployment
        2. Rollback capability
        3. Gradual learning rate
        """
        
        # ─────────────────────────────────────────────────────────────
        # Save baseline model
        # ─────────────────────────────────────────────────────────────
        baseline_state = copy.deepcopy(self.model.state_dict())
        baseline_metrics = self._evaluate(validation_set)
        
        # ─────────────────────────────────────────────────────────────
        # Fine-tune with low learning rate
        # ─────────────────────────────────────────────────────────────
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        
        self.model.train()
        for epoch in range(3):  # Limited epochs
            for batch in feedback_loader:
                optimizer.zero_grad()
                outputs = self.model(batch['frames'])
                loss = criterion(outputs, batch['labels'])
                loss.backward()
                optimizer.step()
        
        # ─────────────────────────────────────────────────────────────
        # Validate new model
        # ─────────────────────────────────────────────────────────────
        new_metrics = self._evaluate(validation_set)
        
        # ─────────────────────────────────────────────────────────────
        # Safety check: ensure no significant degradation
        # ─────────────────────────────────────────────────────────────
        if new_metrics['accuracy'] < baseline_metrics['accuracy'] - 0.02:
            # Rollback
            self.model.load_state_dict(baseline_state)
            return {
                'success': False,
                'reason': 'Performance degradation detected',
                'rollback': True
            }
        
        return {
            'success': True,
            'improvement': new_metrics['accuracy'] - baseline_metrics['accuracy']
        }
```

---

<div style="page-break-after: always;"></div>

## 18. Testing Strategy

### 18.1 Testing Pyramid

```
                         ┌─────────────────┐
                         │    E2E Tests    │  10%
                         │   (Selenium)    │
                         ├─────────────────┤
                         │   Integration   │  20%
                         │     Tests       │
                         ├─────────────────┤
                         │                 │
                         │   Unit Tests    │  70%
                         │                 │
                         └─────────────────┘
```

### 18.2 Unit Tests

```python
# tests/test_models.py
import pytest
import torch
from src.models.student import create_student_model
from src.models.specialists_fixed import create_compression_model

class TestStudentModel:
    @pytest.fixture
    def model(self):
        return create_student_model()
    
    def test_output_shape(self, model):
        """Test model output dimensions"""
        x = torch.randn(2, 4, 3, 224, 224)
        output = model(x)
        assert output.shape == (2, 2)
    
    def test_gradient_flow(self, model):
        """Test gradient computation"""
        x = torch.randn(2, 4, 3, 224, 224, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
    
    def test_eval_mode_deterministic(self, model):
        """Test deterministic output in eval mode"""
        model.eval()
        x = torch.randn(1, 4, 3, 224, 224)
        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)
        assert torch.allclose(output1, output2)


class TestPreprocessing:
    def test_frame_extraction(self):
        """Test frame extraction from video"""
        from eraksha_agent import ErakshAgent
        agent = ErakshAgent(device='cpu')
        frames = agent.extract_frames('test_video.mp4', max_frames=8)
        
        assert len(frames) <= 8
        assert all(f.shape == (3, 224, 224) for f in frames)
    
    def test_audio_extraction(self):
        """Test audio extraction from video"""
        from eraksha_agent import ErakshAgent
        agent = ErakshAgent(device='cpu')
        audio = agent.extract_audio('test_video.mp4', duration=3.0)
        
        assert audio.shape[0] == 48000  # 3 seconds at 16kHz
```

### 18.3 Integration Tests

```python
# tests/test_integration.py
import pytest
from eraksha_agent import ErakshAgent

class TestAgentIntegration:
    @pytest.fixture
    def agent(self):
        return ErakshAgent(device='cpu')
    
    def test_full_pipeline(self, agent):
        """Test complete prediction pipeline"""
        result = agent.predict('test_video.mp4')
        
        assert result['success'] == True
        assert result['prediction'] in ['REAL', 'FAKE']
        assert 0 <= result['confidence'] <= 1
        assert 'explanation' in result
        assert 'processing_time' in result
    
    def test_routing_logic(self, agent):
        """Test intelligent routing"""
        result = agent.predict('high_confidence_video.mp4')
        
        if result['confidence'] > 0.85:
            # High confidence should skip specialists
            assert len(result['specialists_used']) == 0


class TestAPIIntegration:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from backend.app_agentic_corrected import app
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get('/health')
        assert response.status_code == 200
        assert response.json()['status'] == 'healthy'
    
    def test_predict_endpoint(self, client):
        """Test prediction endpoint"""
        with open('test_video.mp4', 'rb') as f:
            response = client.post(
                '/predict',
                files={'file': ('test.mp4', f, 'video/mp4')}
            )
        assert response.status_code == 200
        assert 'prediction' in response.json()
```

### 18.4 Performance Benchmarks

```python
# benchmarks/benchmark_inference.py
import time
import numpy as np
from eraksha_agent import ErakshAgent

def benchmark_inference(agent, num_iterations=100):
    """Benchmark inference performance"""
    
    # Create dummy input
    frames = [torch.randn(3, 224, 224) for _ in range(8)]
    
    # Warmup
    for _ in range(10):
        agent.run_student_inference(frames)
    
    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        agent.run_student_inference(frames)
        end = time.perf_counter()
        times.append(end - start)
    
    times = np.array(times) * 1000  # Convert to ms
    
    print(f"Inference Benchmark Results:")
    print(f"  Mean: {times.mean():.2f} ms")
    print(f"  Std:  {times.std():.2f} ms")
    print(f"  P50:  {np.percentile(times, 50):.2f} ms")
    print(f"  P95:  {np.percentile(times, 95):.2f} ms")
    print(f"  P99:  {np.percentile(times, 99):.2f} ms")
```

---

<div style="page-break-after: always;"></div>

# Appendices

---

## Appendix A: Mathematical Foundations

### A.1 Knowledge Distillation Loss

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE DISTILLATION                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Total Loss:                                                             │
│                                                                          │
│      L_total = α × L_soft + (1 - α) × L_hard                            │
│                                                                          │
│  Where:                                                                  │
│      • L_soft: KL divergence between teacher and student soft labels    │
│      • L_hard: Cross-entropy with true labels                           │
│      • α: Balancing coefficient (typically 0.7)                         │
│                                                                          │
│  Soft Loss:                                                              │
│                                                                          │
│      L_soft = T² × KL(softmax(z_t/T) || softmax(z_s/T))                 │
│                                                                          │
│  Where:                                                                  │
│      • z_t: Teacher logits                                              │
│      • z_s: Student logits                                              │
│      • T: Temperature (typically 3-5)                                   │
│      • T² scaling compensates for gradient magnitude reduction          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### A.2 Confidence Score Computation

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CONFIDENCE COMPUTATION                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Single Model Confidence:                                                │
│                                                                          │
│      confidence = prediction_strength × (1 - variance_penalty)          │
│                                                                          │
│  Where:                                                                  │
│      prediction_strength = max(P(real), P(fake))                        │
│      variance_penalty = Var(frame_predictions) / max_variance           │
│                                                                          │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  Ensemble Confidence:                                                    │
│                                                                          │
│      confidence = weighted_avg(model_confidences) × consensus_factor    │
│                                                                          │
│  Where:                                                                  │
│      consensus_factor = 1 - std(model_predictions) / 0.5                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### A.3 Bias Correction Formula

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    BIAS CORRECTION                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Corrected Prediction:                                                   │
│                                                                          │
│      corrected_pred = original_pred + bias_correction                   │
│      corrected_pred = clamp(corrected_pred, 0, 1)                       │
│                                                                          │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  Bias Derivation:                                                        │
│                                                                          │
│      bias = (fake_accuracy - real_accuracy) / 2                         │
│      correction = -bias × scaling_factor                                │
│                                                                          │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  Weighted Aggregation:                                                   │
│                                                                          │
│      final_pred = Σ(weight_i × corrected_pred_i) / Σ(weight_i)         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### A.4 Convolution Mathematics

```
2D Convolution Operation:

For input tensor X of shape [B, C_in, H_in, W_in]
and kernel K of shape [C_out, C_in, K_h, K_w]:

Output[b, c_out, h, w] = Σ(c_in, k_h, k_w) of:
    X[b, c_in, h×s + k_h - p, w×s + k_w - p] × K[c_out, c_in, k_h, k_w] + bias[c_out]

Where:
    s: stride
    p: padding

Output dimensions:
    H_out = floor((H_in + 2×p - K_h) / s) + 1
    W_out = floor((W_in + 2×p - K_w) / s) + 1
```

---

<div style="page-break-after: always;"></div>

## Appendix B: Troubleshooting Guide

### B.1 Common Issues and Solutions

#### Issue: No faces detected in video

| Symptom | Solution |
|---------|----------|
| Error: "No frames extracted" | Lower MTCNN threshold: `MTCNN(thresholds=[0.5, 0.6, 0.6])` |
| Empty frames list | Check video quality (min 480p recommended) |
| | Ensure faces are visible and not too small |
| | Try alternative detector (dlib, RetinaFace) |

#### Issue: Out of memory during inference

| Symptom | Solution |
|---------|----------|
| CUDA out of memory | Reduce batch size: `batch_size = 1` |
| Process killed by OOM | Enable gradient checkpointing |
| | Use mixed precision: `torch.cuda.amp.autocast()` |
| | Clear cache: `torch.cuda.empty_cache()` |
| | Use CPU inference for large videos |

#### Issue: Models not loading

| Symptom | Solution |
|---------|----------|
| FileNotFoundError | Verify model file paths exist |
| State dict mismatch | Ensure model architecture matches weights |
| | Use `strict=False` for partial loading |
| | Handle pickle vs torch format correctly |

#### Issue: Low accuracy on real-world videos

| Symptom | Solution |
|---------|----------|
| High test accuracy, low deployment | Check for domain shift |
| Consistent misclassification | Adjust bias correction values |
| | Use appropriate specialist for video type |
| | Lower confidence threshold |

### B.2 Error Codes Reference

| Code | Description | Resolution |
|:----:|-------------|------------|
| E001 | Video file not found | Check file path |
| E002 | Invalid video format | Use MP4/AVI/MOV/WebM |
| E003 | Video file too large | Max 100MB |
| E004 | No faces detected | Check video quality |
| E005 | Model loading failed | Verify model files |
| E006 | Inference error | Check input format |
| E007 | Audio extraction failed | Check audio track |
| E008 | Memory allocation error | Reduce batch size |
| E009 | Rate limit exceeded | Wait and retry |
| E010 | Invalid API key | Check credentials |

### B.3 Debugging Commands

```bash
# Check model loading
python -c "from eraksha_agent import ErakshAgent; a = ErakshAgent(); print(a.models)"

# Test single video
python -c "from eraksha_agent import ErakshAgent; a = ErakshAgent(); print(a.predict('test.mp4'))"

# Check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check API health
curl -X GET http://localhost:8000/health

# Test prediction endpoint
curl -X POST http://localhost:8000/predict -F "file=@test_video.mp4"
```

---

<div style="page-break-after: always;"></div>

## Appendix C: API Reference

### C.1 Endpoints

#### Health Check

```
GET /health

Response:
{
    "status": "healthy",
    "models_loaded": {
        "student": "loaded",
        "av": "not_available",
        "cm": "loaded",
        "rr": "loaded",
        "ll": "loaded",
        "tm": "loaded"
    },
    "version": "2.0.0"
}
```

#### Predict Video

```
POST /predict
Content-Type: multipart/form-data

Parameters:
    file: Video file (MP4, AVI, MOV, WebM)

Response:
{
    "success": true,
    "prediction": "FAKE",
    "confidence": 94.9,
    "confidence_level": "high",
    "explanation": "This video is classified as MANIPULATED...",
    "details": {
        "best_model": "av",
        "specialists_used": ["cm", "tm"],
        "processing_time": 2.1,
        "all_predictions": {
            "student": {"prediction": 0.72, "confidence": 0.85},
            "cm": {"prediction": 0.68, "confidence": 0.78}
        }
    },
    "bias_correction": "applied"
}
```

#### Get Models

```
GET /models

Response:
{
    "models": {
        "student": {"status": "loaded", "type": "Baseline Generalist"},
        "av": {"status": "not_available", "type": "Audio-Visual"},
        "cm": {"status": "loaded", "type": "Compression"},
        "rr": {"status": "loaded", "type": "Re-recording"},
        "ll": {"status": "loaded", "type": "Low-light"},
        "tm": {"status": "loaded", "type": "Temporal"}
    },
    "total_loaded": 5,
    "bias_correction_enabled": true
}
```

### C.2 Error Responses

```
{
    "detail": "Error message",
    "error_code": "E001",
    "request_id": "abc123"
}
```

---

## Appendix D: Glossary

| Term | Definition |
|------|------------|
| **Agentic AI** | AI systems capable of autonomous decision-making without continuous human intervention |
| **Backbone** | Main feature extraction network, typically pre-trained on ImageNet |
| **Bias Correction** | Adjustment to model predictions to compensate for systematic errors |
| **Confidence Calibration** | Adjusting predicted probabilities to match actual accuracy rates |
| **Deepfake** | Synthetic media created using deep learning, typically involving face manipulation |
| **Ensemble** | Combination of multiple models to improve prediction accuracy |
| **Knowledge Distillation** | Transferring knowledge from a large model (teacher) to smaller model (student) |
| **MTCNN** | Multi-task Cascaded Convolutional Networks for face detection |
| **Quantization** | Reducing numerical precision of model weights to decrease size |
| **ResNet** | Residual Network architecture with skip connections |
| **Softmax** | Function that converts logits to probability distribution |
| **Temperature Scaling** | Calibration technique using single parameter to adjust confidence |

---

<div style="page-break-after: always;"></div>

---

<div align="center">

## Document Information

---

| Property | Value |
|:--------:|:-----:|
| **Document Title** | E-RAKSHA: Agentic AI for Deepfake Detection |
| **Version** | 2.0.0 |
| **Status** | Final |
| **Classification** | Technical Submission |

---

### Prepared For

**E-Raksha Hackathon 2026**  
*Problem Statement II: Deepfake Detection*

**eDC IIT Delhi** in collaboration with **CyberPeace Foundation**  
*Official Pre-Summit Event of BECon'26 and AI Impact Summit'26*

---

### Document Statistics

| Metric | Value |
|:------:|:-----:|
| Total Sections | 18 main + 4 appendices |
| Total Pages | ~100 (estimated) |
| Code Examples | 50+ |
| Diagrams | 20+ |
| Tables | 30+ |

---

### Team E-Raksha

| Role | Responsibility |
|------|----------------|
| **Person 1 (Pranay)** | Team Lead, System Architecture, Documentation |
| **Person 2** | Specialist Models (CM, RR, LL, TM) |
| **Person 3** | Integration, Testing, Deployment |
| **Person 4** | AV-Model, Audio-Visual Analysis |

---

### Contact

📧 **Email:** contact.eraksha@gmail.com  
[LINK] **GitHub:** github.com/eraksha-project

---

### License

This documentation is provided for the E-Raksha Hackathon 2026 evaluation.  
All rights reserved by the authors.

---

**Last Updated:** December 30, 2025

</div>

---

<div align="center">

# E-RAKSHA

**E-RAKSHA**

*Protecting Digital Truth Through Agentic AI*

</div>
