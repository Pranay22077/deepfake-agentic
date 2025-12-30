# 🚀 Interceptor Deployment Guide

## Architecture Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Frontend      │     │    Backend      │     │  Model Weights  │
│   (Vercel)      │────▶│(Railway/Render) │────▶│ (Hugging Face)  │
│   Free tier     │     │   Free tier     │     │   Free storage  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## Step 1: Upload Model Weights to Hugging Face

### 1.1 Create Hugging Face Account
1. Go to https://huggingface.co/join
2. Create a free account

### 1.2 Create a Model Repository
1. Click "New Model" at https://huggingface.co/new
2. Name it: `interceptor-models`
3. Set visibility: Public (for free hosting)

### 1.3 Upload Model Files
```bash
# Install huggingface CLI
pip install huggingface-hub

# Login
huggingface-cli login

# Upload models
huggingface-cli upload your-username/interceptor-models models/ --repo-type model
```

Or upload via web interface:
1. Go to your repo: https://huggingface.co/your-username/interceptor-models
2. Click "Files and versions" → "Add file" → "Upload files"
3. Upload all `.pt` files from the `models/` folder

### 1.4 Update Backend Code
Edit `backend/model_downloader.py`:
```python
HF_REPO = "your-username/interceptor-models"  # Your actual repo
```

---

## Step 2: Deploy Frontend to Vercel

### 2.1 Push Code to GitHub
```bash
git add .
git commit -m "Prepare for deployment"
git push origin main
```

### 2.2 Deploy to Vercel
1. Go to https://vercel.com
2. Click "Import Project"
3. Select your GitHub repository
4. Configure:
   - Framework: Vite
   - Build Command: `npm run build`
   - Output Directory: `dist`
5. Add Environment Variable:
   ```
   VITE_API_URL=https://your-backend-url.railway.app
   ```
6. Click "Deploy"

### 2.3 Update Frontend API URL
Create `.env.production`:
```
VITE_API_URL=https://your-backend-url.railway.app
```

Update `src/app/pages/AnalysisWorkbench.tsx`:
```typescript
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Then use:
const response = await fetch(`${API_URL}/predict`, {
```

---

## Step 3: Deploy Backend to Railway (Recommended)

### 3.1 Create Railway Account
1. Go to https://railway.app
2. Sign up with GitHub

### 3.2 Deploy
1. Click "New Project" → "Deploy from GitHub repo"
2. Select your repository
3. Railway auto-detects Python
4. Add environment variables if needed:
   ```
   PORT=8000
   HF_REPO=your-username/interceptor-models
   ```
5. Click "Deploy"

### 3.3 Get Your Backend URL
- Railway provides: `https://your-project.up.railway.app`
- Use this URL in your Vercel frontend config

---

## Alternative: Deploy Backend to Render

### 3.1 Create Render Account
1. Go to https://render.com
2. Sign up with GitHub

### 3.2 Deploy
1. Click "New" → "Web Service"
2. Connect your GitHub repo
3. Configure:
   - Name: `interceptor-api`
   - Environment: Python 3
   - Build Command: `pip install -r backend/requirements.txt`
   - Start Command: `cd backend && python app.py`
4. Click "Create Web Service"

---

## Step 4: Configure CORS & Environment

### Backend Environment Variables
```
PORT=8000
HF_REPO=your-username/interceptor-models
ALLOWED_ORIGINS=https://your-frontend.vercel.app
```

### Frontend Environment Variables (Vercel)
```
VITE_API_URL=https://your-backend.railway.app
```

---

## Step 5: Test Deployment

### Test Backend
```bash
curl https://your-backend.railway.app/health
```

Expected response:
```json
{
  "status": "healthy",
  "torch_available": true,
  "models_found": ["bg_model_student.pt", ...]
}
```

### Test Frontend
1. Open https://your-frontend.vercel.app
2. Upload a test video
3. Verify analysis works

---

## Cost Summary (Free Tiers)

| Service | Free Tier |
|---------|-----------|
| **Vercel** | 100GB bandwidth/month |
| **Railway** | $5 credit/month (~500 hours) |
| **Render** | 750 hours/month |
| **Hugging Face** | Unlimited public models |

---

## Troubleshooting

### Backend not starting
- Check logs in Railway/Render dashboard
- Ensure `requirements.txt` is correct
- Verify Python version compatibility

### Models not loading
- Check Hugging Face repo is public
- Verify model file names match
- Check download permissions

### CORS errors
- Add frontend URL to `ALLOWED_ORIGINS`
- Verify backend CORS middleware config

### Slow cold starts
- Railway/Render free tiers sleep after inactivity
- First request may take 30-60 seconds
- Consider upgrading for always-on

---

## Production Checklist

- [ ] Model weights uploaded to Hugging Face
- [ ] Backend deployed and healthy
- [ ] Frontend deployed and connected
- [ ] CORS configured correctly
- [ ] Environment variables set
- [ ] Test video analysis works
- [ ] Custom domain configured (optional)

---

## Quick Commands

```bash
# Push to GitHub
git add . && git commit -m "Deploy" && git push

# Test backend locally
cd backend && python app.py

# Test frontend locally
npm run dev

# Build frontend
npm run build
```

---

**Your Interceptor deployment is ready!** 🛡️
