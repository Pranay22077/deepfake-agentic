#!/bin/bash
# Clear old cached models from deployment

echo "🗑️  Clearing old model files..."

# Remove old models from models directory
rm -f models/baseline_student.pt
rm -f models/av_model_student.pt
rm -f models/cm_model_student.pt
rm -f models/rr_model_student.pt
rm -f models/ll_model_student.pt

# Keep TM model
echo "✅ Kept: models/tm_model_student.pt"

# Remove from root directory if they exist there
rm -f baseline_student.pt
rm -f av_model_student.pt
rm -f cm_model_student.pt
rm -f rr_model_student.pt
rm -f ll_model_student.pt

echo "✅ Old models cleared!"
echo "📝 Next steps:"
echo "   1. Upload new models to Hugging Face"
echo "   2. Redeploy your application"
echo "   3. Models will be downloaded fresh from Hugging Face"
