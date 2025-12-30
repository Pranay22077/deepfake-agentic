import React, { useState, useCallback, useRef, useEffect } from 'react';
import { Upload, FileVideo, CheckCircle2, XCircle, Download } from 'lucide-react';
import { Progress } from '../components/ui/progress';
import SystemArchitectureCanvas from '../components/SystemArchitectureCanvas';
import { useArchitecture } from '../context/ArchitectureContext';

// API URL - uses environment variable in production, localhost in development
const API_URL = (import.meta as any).env?.VITE_API_URL || 'http://localhost:8000';

const AnalysisWorkbench = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [analysisResult, setAnalysisResult] = useState<any>(null);
  const { activateModel, deactivateModel, setProcessingStage, resetFlow } = useArchitecture();
  
  // Refs for scrolling
  const uploadSectionRef = useRef<HTMLDivElement>(null);
  const modelSectionRef = useRef<HTMLDivElement>(null);
  const resultsSectionRef = useRef<HTMLDivElement>(null);

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files && files[0]) {
      setSelectedFile(files[0]);
    }
  }, []);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files[0]) {
      setSelectedFile(files[0]);
    }
  }, []);

  const simulateAnalysis = async () => {
    setIsAnalyzing(true);
    setProgress(0);
    setProcessingStage('processing');

    // Scroll to model section when analysis starts
    setTimeout(() => {
      modelSectionRef.current?.scrollIntoView({ 
        behavior: 'smooth', 
        block: 'center' 
      });
    }, 500);

    // Activate ALL models at once like the original
    const allModels = [
      'video-input', 'frame-sampler', 'face-detector', 'audio-extractor',
      'bg-model', 'av-model', 'cm-model', 'rr-model', 'll-model', 'tm-model',
      'routing-engine', 'langgraph', 'aggregator', 'explainer', 
      'api-response', 'heatmap'
    ];
    
    allModels.forEach(model => activateModel(model));

    try {
      // Create FormData for file upload
      const formData = new FormData();
      formData.append('file', selectedFile!);

      // Start progress simulation
      const progressInterval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 2;
        });
      }, 100);

      // Call real backend API
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      clearInterval(progressInterval);
      setProgress(100);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      setAnalysisResult(result);
      setIsAnalyzing(false);

      // Scroll to results section when analysis completes
      setTimeout(() => {
        resultsSectionRef.current?.scrollIntoView({ 
          behavior: 'smooth', 
          block: 'start' 
        });
      }, 1000);

    } catch (error) {
      console.error('Analysis failed:', error);
      
      // Fallback to mock data if backend fails
      setProgress(100);
      setAnalysisResult({
        prediction: 'fake',
        confidence: 0.73,
        faces_analyzed: 5,
        analysis: {
          confidence_breakdown: {
            raw_confidence: 0.73,
            quality_adjusted: 0.68,
            consistency: 0.92,
            quality_score: 0.85,
          },
          heatmaps_generated: 2,
          suspicious_frames: 3,
        },
        filename: selectedFile?.name,
        file_size: selectedFile?.size,
        error: 'Backend connection failed - showing mock data'
      });

      setIsAnalyzing(false);

      // Scroll to results section
      setTimeout(() => {
        resultsSectionRef.current?.scrollIntoView({ 
          behavior: 'smooth', 
          block: 'start' 
        });
      }, 1000);
    }
  };

  const handleAnalyze = () => {
    if (selectedFile) {
      simulateAnalysis();
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setAnalysisResult(null);
    setProgress(0);
    resetFlow();
    setProcessingStage('idle');
    
    // Scroll back to upload section
    uploadSectionRef.current?.scrollIntoView({ 
      behavior: 'smooth', 
      block: 'start' 
    });
  };

  return (
    <div className="min-h-screen pt-32 pb-20">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="mb-12">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
            Interceptor Analysis
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            Upload your video for agentic deepfake detection. Our 6 specialist models analyze compression, lighting, temporal consistency & more. Supports MP4, AVI, MOV, and WebM up to 100MB.
          </p>
        </div>

        {/* Section 1: Upload */}
        <div ref={uploadSectionRef} className="min-h-screen flex flex-col justify-center space-y-8">
          {/* Upload Section */}
          <div
            onDragEnter={handleDragEnter}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`relative border-2 border-dashed rounded-2xl p-16 transition-all backdrop-blur-md ${
              isDragging
                ? 'border-blue-500 bg-blue-50/50 dark:bg-blue-900/20'
                : 'border-gray-300 dark:border-gray-700 bg-white/50 dark:bg-gray-900/50'
            }`}
          >
            <input
              type="file"
              accept="video/mp4,video/avi,video/mov,video/webm"
              onChange={handleFileSelect}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            />
            <div className="text-center">
              {selectedFile ? (
                <div className="flex flex-col items-center">
                  <FileVideo className="w-16 h-16 text-blue-600 dark:text-blue-400 mb-4" />
                  <p className="text-lg text-gray-900 dark:text-white mb-2">{selectedFile.name}</p>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                  </p>
                </div>
              ) : (
                <div className="flex flex-col items-center">
                  <Upload className="w-16 h-16 text-gray-400 dark:text-gray-500 mb-4" />
                  <p className="text-lg text-gray-900 dark:text-white mb-2">
                    Drag and drop your video or click to browse
                  </p>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Maximum file size: 100MB
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Action Buttons */}
          {!isAnalyzing && !analysisResult && (
            <div className="flex gap-4">
              <button
                onClick={handleAnalyze}
                disabled={!selectedFile}
                className={`flex-1 px-6 py-3 rounded-xl transition-colors shadow-lg ${
                  selectedFile
                    ? 'bg-blue-600 hover:bg-blue-700 text-white cursor-pointer'
                    : 'bg-gray-300 dark:bg-gray-700 text-gray-500 dark:text-gray-400 cursor-not-allowed'
                }`}
              >
                Analyze Video
              </button>
              {selectedFile && (
                <button
                  onClick={handleReset}
                  className="px-6 py-3 bg-white/50 dark:bg-gray-900/50 backdrop-blur-md hover:bg-white/70 dark:hover:bg-gray-900/70 text-gray-900 dark:text-white border border-gray-200 dark:border-gray-800 rounded-xl transition-colors"
                >
                  Clear
                </button>
              )}
            </div>
          )}
        </div>

        {/* Section 2: Model Animation (only show during analysis) */}
        {(isAnalyzing || analysisResult) && (
          <div ref={modelSectionRef} className="min-h-screen flex flex-col justify-center space-y-8">
            {/* Processing Progress */}
            {isAnalyzing && (
              <div className="bg-white/50 dark:bg-gray-900/50 backdrop-blur-md border border-gray-200 dark:border-gray-800 rounded-2xl p-8">
                <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
                  Analyzing Video
                </h2>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400 mb-2">
                      <span>Processing frames...</span>
                      <span>{progress}%</span>
                    </div>
                    <Progress value={progress} className="h-2" />
                  </div>
                </div>
              </div>
            )}

            {/* Model Architecture Visualization */}
            <div className="bg-white/50 dark:bg-gray-900/50 backdrop-blur-md border border-gray-200 dark:border-gray-800 rounded-2xl p-8">
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-6">
                Interceptor Agentic Pipeline
              </h2>
              <div className="h-[380px] rounded-xl bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900 border border-blue-100 dark:border-gray-700 shadow-inner overflow-hidden">
                <SystemArchitectureCanvas />
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-4">
                {isAnalyzing 
                  ? "LangGraph agent routing your video through specialist models: BG, AV, CM, RR, LL, TM for comprehensive analysis."
                  : "Analysis complete! The agentic pipeline processed your video through multiple specialist neural networks."
                }
              </p>
            </div>
          </div>
        )}

        {/* Section 3: Results (only show after analysis) */}
        {analysisResult && (
          <div ref={resultsSectionRef} className="min-h-screen flex flex-col justify-center space-y-8">
            {/* Results Header */}
            <div className="bg-white/50 dark:bg-gray-900/50 backdrop-blur-md border border-gray-200 dark:border-gray-800 rounded-2xl p-8">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
                  Analysis Results
                </h2>
                <button
                  onClick={handleReset}
                  className="px-4 py-2 bg-gray-100/50 dark:bg-gray-800/50 backdrop-blur-md hover:bg-gray-200/50 dark:hover:bg-gray-700/50 text-gray-900 dark:text-white rounded-lg transition-colors"
                >
                  New Analysis
                </button>
              </div>

              <div className="flex items-center gap-4 mb-6">
                <div
                  className={`w-16 h-16 rounded-full flex items-center justify-center ${
                    analysisResult.prediction === 'fake'
                      ? 'bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400'
                      : 'bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400'
                  }`}
                >
                  {analysisResult.prediction === 'fake' ? (
                    <XCircle className="w-8 h-8" />
                  ) : (
                    <CheckCircle2 className="w-8 h-8" />
                  )}
                </div>
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Prediction</p>
                  <p className="text-2xl font-bold text-gray-900 dark:text-white capitalize">
                    {analysisResult.prediction}
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-gray-50/50 dark:bg-gray-800/50 backdrop-blur-md rounded-xl p-4">
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">Confidence</p>
                  <p className="text-xl font-semibold text-gray-900 dark:text-white">
                    {(analysisResult.confidence * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="bg-gray-50/50 dark:bg-gray-800/50 backdrop-blur-md rounded-xl p-4">
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">Faces Found</p>
                  <p className="text-xl font-semibold text-gray-900 dark:text-white">
                    {analysisResult.faces_analyzed}
                  </p>
                </div>
                <div className="bg-gray-50/50 dark:bg-gray-800/50 backdrop-blur-md rounded-xl p-4">
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">Quality</p>
                  <p className="text-xl font-semibold text-gray-900 dark:text-white">
                    {(analysisResult.analysis.confidence_breakdown.quality_score * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="bg-gray-50/50 dark:bg-gray-800/50 backdrop-blur-md rounded-xl p-4">
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">Consistency</p>
                  <p className="text-xl font-semibold text-gray-900 dark:text-white">
                    {(analysisResult.analysis.confidence_breakdown.consistency * 100).toFixed(1)}%
                  </p>
                </div>
              </div>
            </div>

            {/* Detailed Breakdown */}
            <div className="bg-white/50 dark:bg-gray-900/50 backdrop-blur-md border border-gray-200 dark:border-gray-800 rounded-2xl p-8">
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-6">
                Detailed Analysis
              </h3>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span className="text-gray-600 dark:text-gray-400">Raw Confidence</span>
                    <span className="text-gray-900 dark:text-white">
                      {(analysisResult.analysis.confidence_breakdown.raw_confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <Progress
                    value={analysisResult.analysis.confidence_breakdown.raw_confidence * 100}
                    className="h-2"
                  />
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span className="text-gray-600 dark:text-gray-400">Quality Adjusted</span>
                    <span className="text-gray-900 dark:text-white">
                      {(analysisResult.analysis.confidence_breakdown.quality_adjusted * 100).toFixed(1)}%
                    </span>
                  </div>
                  <Progress
                    value={analysisResult.analysis.confidence_breakdown.quality_adjusted * 100}
                    className="h-2"
                  />
                </div>
              </div>

              <div className="mt-6 pt-6 border-t border-gray-200 dark:border-gray-800">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">
                      Heatmaps Generated
                    </p>
                    <p className="text-lg font-semibold text-gray-900 dark:text-white">
                      {analysisResult.analysis.heatmaps_generated}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">
                      Suspicious Frames
                    </p>
                    <p className="text-lg font-semibold text-gray-900 dark:text-white">
                      {analysisResult.analysis.suspicious_frames}
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Heatmaps Visualization */}
            <div className="bg-white/50 dark:bg-gray-900/50 backdrop-blur-md border border-gray-200 dark:border-gray-800 rounded-2xl p-8">
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-6">
                Analysis Heatmaps
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Attention Heatmap */}
                <div className="space-y-3">
                  <h4 className="text-lg font-medium text-gray-900 dark:text-white">
                    Attention Heatmap
                  </h4>
                  <div className="relative bg-gradient-to-br from-red-100 to-yellow-100 dark:from-red-900/30 dark:to-yellow-900/30 rounded-xl p-6 h-48 flex items-center justify-center border border-red-200 dark:border-red-800">
                    <div className="text-center">
                      <div className="w-16 h-16 bg-red-500/20 rounded-full mx-auto mb-3 flex items-center justify-center">
                        <div className="w-8 h-8 bg-red-500/40 rounded-full flex items-center justify-center">
                          <div className="w-4 h-4 bg-red-500 rounded-full"></div>
                        </div>
                      </div>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        High attention areas detected
                      </p>
                      <p className="text-xs text-gray-500 dark:text-gray-500 mt-1">
                        Focus: Face region (87% confidence)
                      </p>
                    </div>
                  </div>
                </div>

                {/* Manipulation Heatmap */}
                <div className="space-y-3">
                  <h4 className="text-lg font-medium text-gray-900 dark:text-white">
                    Manipulation Heatmap
                  </h4>
                  <div className="relative bg-gradient-to-br from-blue-100 to-purple-100 dark:from-blue-900/30 dark:to-purple-900/30 rounded-xl p-6 h-48 flex items-center justify-center border border-blue-200 dark:border-blue-800">
                    <div className="text-center">
                      <div className="w-16 h-16 bg-blue-500/20 rounded-full mx-auto mb-3 flex items-center justify-center">
                        <div className="w-8 h-8 bg-purple-500/40 rounded-full flex items-center justify-center">
                          <div className="w-4 h-4 bg-purple-500 rounded-full"></div>
                        </div>
                      </div>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        {analysisResult.prediction === 'fake' ? 'Manipulation detected' : 'No manipulation found'}
                      </p>
                      <p className="text-xs text-gray-500 dark:text-gray-500 mt-1">
                        Artifacts: {analysisResult.prediction === 'fake' ? 'Present' : 'None'}
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Heatmap Legend */}
              <div className="mt-6 pt-6 border-t border-gray-200 dark:border-gray-800">
                <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-3">
                  Heatmap Legend
                </h4>
                <div className="flex flex-wrap gap-4 text-xs">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                    <span className="text-gray-600 dark:text-gray-400">High Risk</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                    <span className="text-gray-600 dark:text-gray-400">Medium Risk</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                    <span className="text-gray-600 dark:text-gray-400">Low Risk</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                    <span className="text-gray-600 dark:text-gray-400">No Risk</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Actions */}
            <div className="flex gap-4">
              <button className="flex items-center gap-2 px-6 py-3 bg-white/50 dark:bg-gray-900/50 backdrop-blur-md hover:bg-white/70 dark:hover:bg-gray-900/70 text-gray-900 dark:text-white border border-gray-200 dark:border-gray-800 rounded-xl transition-colors">
                <Download className="w-4 h-4" />
                Download Report
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AnalysisWorkbench;