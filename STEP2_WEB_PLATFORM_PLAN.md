# Step 2: Web Platform Development Plan

## Overview
Build a web-first agentic deepfake detection service using your trained model from Step 1.

## Prerequisites (Already Complete)
- Supabase account and project setup
- GitHub repository with secrets
- Vercel and Railway accounts
- Trained baseline model from Step 1

## Step 2 Architecture

```
Frontend (Vercel)          Backend (Railway)           Database (Supabase)
├── React App              ├── FastAPI Server          ├── PostgreSQL
├── Video Upload            ├── Model Inference         ├── File Storage
├── Results Display         ├── Agent Orchestration     └── User Management
└── Real-time Updates       └── Feedback Collection
```

## Phase 2A: Backend API Development

### 2A.1 - FastAPI Backend Structure
```
src/
├── api/
│   ├── app.py              # Main FastAPI application
│   ├── inference.py        # Model inference endpoints
│   ├── upload.py           # File upload handling
│   └── feedback.py         # User feedback collection
├── models/
│   ├── loader.py           # Model loading and caching
│   └── inference.py        # Inference logic
└── db/
    └── supabase_client.py  # Database connection
```

### 2A.2 - Model Integration
- Load your trained `baseline_student.pt` model
- Create inference pipeline for video processing
- Add confidence scoring and thresholding
- Generate explanation heatmaps

### 2A.3 - Agent Workflow (Simplified)
```
Video Upload → Preprocessing → Model Inference → Confidence Check → Results
```

## Phase 2B: Frontend Development

### 2B.1 - React Components
```
src/
├── components/
│   ├── VideoUpload.jsx     # Drag & drop + camera
│   ├── ResultsDisplay.jsx  # Show predictions + heatmaps
│   ├── ProgressBar.jsx     # Real-time processing updates
│   └── FeedbackForm.jsx    # User feedback collection
└── pages/
    ├── Home.jsx            # Main upload page
    └── Results.jsx         # Results visualization
```

### 2B.2 - Key Features
- Video upload (drag & drop or camera capture)
- Real-time processing progress
- Results with confidence scores
- Heatmap overlays showing suspicious regions
- User feedback collection

## Phase 2C: Database Schema

### Supabase Tables
```sql
-- Inference logs
CREATE TABLE inference_logs (
  id SERIAL PRIMARY KEY,
  video_path TEXT NOT NULL,
  result JSONB NOT NULL,
  confidence REAL NOT NULL,
  model_version TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- User feedback
CREATE TABLE feedback_buffer (
  id SERIAL PRIMARY KEY,
  video_path TEXT NOT NULL,
  user_label TEXT CHECK (user_label IN ('real', 'fake', 'unknown')),
  user_confidence REAL,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
```

## Phase 2D: Deployment Pipeline

### 2D.1 - Railway Backend Deployment
- Deploy FastAPI app to Railway
- Configure environment variables
- Set up model file storage

### 2D.2 - Vercel Frontend Deployment
- Deploy React app to Vercel
- Configure API endpoints
- Set up environment variables

## Implementation Steps

### Step 2.1: Setup Local Development
1. Extract your Step 1 outputs
2. Set up local development environment
3. Create project structure

### Step 2.2: Backend Development
1. Create FastAPI application
2. Integrate your trained model
3. Build inference pipeline
4. Add Supabase integration

### Step 2.3: Frontend Development
1. Create React application
2. Build upload interface
3. Create results display
4. Add real-time updates

### Step 2.4: Integration & Testing
1. Connect frontend to backend
2. Test end-to-end workflow
3. Add error handling

### Step 2.5: Deployment
1. Deploy backend to Railway
2. Deploy frontend to Vercel
3. Configure production settings

## Expected Timeline
- **Phase 2A**: Backend (3-4 days)
- **Phase 2B**: Frontend (2-3 days)
- **Phase 2C**: Database (1 day)
- **Phase 2D**: Deployment (1-2 days)

**Total: 7-10 days for complete web platform**

## Success Criteria
- Users can upload videos through web interface
- Model processes videos and returns predictions
- Results display with confidence scores and heatmaps
- Feedback collection system working
- Deployed and accessible via web URL

## Next Actions
1. Run `python verify_step1.py` to verify Step 1 completion
2. Set up local development environment
3. Begin Phase 2A backend development