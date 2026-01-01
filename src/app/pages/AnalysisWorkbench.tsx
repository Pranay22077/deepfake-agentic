import React, { useState, useCallback, useEffect, useRef } from 'react';
import { Upload, FileVideo, CheckCircle2, XCircle, Download, FileText, AlertTriangle, Loader2 } from 'lucide-react';
import { Progress } from '../components/ui/progress';
import { useArchitecture } from '../context/ArchitectureContext';
import { useTheme } from '../context/ThemeContext';
import SystemArchitectureCanvas from '../components/SystemArchitectureCanvas';
import LiquidEther from '../components/LiquidEther';
import { saveAnalysis, checkDuplicateFile, type VideoAnalysis } from '../../lib/supabase';

// Backend API URL - Use environment variable or fallback to Vercel deployment
const API_URL = import.meta.env.VITE_API_URL || 'https://deepfake-agentic.vercel.app/api';

const AnalysisWorkbench = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [analysisResult, setAnalysisResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [isDuplicate, setIsDuplicate] = useState(false);
  const [duplicateResult, setDuplicateResult] = useState<any>(null);

  // Refs for scroll targets
  const animationRef = useRef<HTMLDivElement>(null);
  const resultsRef = useRef<HTMLDivElement>(null);

  const {
    state,
    setCurrentPage,
    setProcessingStage,
    activateModel,
    deactivateModel,
    resetFlow,
  } = useArchitecture();

  useEffect(() => {
    setCurrentPage('workbench');
    return () => resetFlow();
  }, [setCurrentPage, resetFlow]);

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

  const analyzeVideo = async () => {
    if (!selectedFile) return;
    
    setIsAnalyzing(true);
    setProgress(0);
    setError(null);
    setIsDuplicate(false);
    setDuplicateResult(null);
    resetFlow();

    try {
      // FIRST: Check if file was previously analyzed
      setProcessingStage('Checking File History');
      setProgress(3);
      
      const existingAnalysis = await checkDuplicateFile(selectedFile.name, selectedFile.size);
      
      if (existingAnalysis) {
        // File was previously analyzed - show results directly
        setIsDuplicate(true);
        setDuplicateResult(existingAnalysis);
        
        // Convert database result to display format
        const displayResult = {
          prediction: existingAnalysis.prediction,
          confidence: existingAnalysis.confidence,
          best_model: existingAnalysis.models_used?.[0]?.toLowerCase() || 'bg-model',
          specialists_used: existingAnalysis.models_used || ['BG-Model'],
          processing_time: existingAnalysis.processing_time || 2.0,
          explanation: existingAnalysis.prediction === 'fake'
            ? `This video is classified as MANIPULATED (FAKE) with ${(existingAnalysis.confidence * 100).toFixed(1)}% confidence. Previously analyzed on ${new Date(existingAnalysis.created_at!).toLocaleDateString()}.`
            : `This video is classified as AUTHENTIC (REAL) with ${(existingAnalysis.confidence * 100).toFixed(1)}% confidence. Previously analyzed on ${new Date(existingAnalysis.created_at!).toLocaleDateString()}.`,
          raw_result: existingAnalysis.analysis_result,
        };
        
        setAnalysisResult(displayResult);
        setProcessingStage('Analysis Complete');
        setProgress(100);
        setIsAnalyzing(false);
        
        // Scroll to results
        setTimeout(() => {
          resultsRef.current?.scrollIntoView({ behavior: 'smooth' });
        }, 500);
        
        return;
      }

      // File is new - proceed with normal analysis
      // Stage 1: Video Upload
      setProcessingStage('Uploading Video');
      activateModel('video-input');
      setProgress(5);

      // Create form data
      const formData = new FormData();
      formData.append('file', selectedFile);

      // Stage 2: Sending to API
      setProcessingStage('Connecting to Server');
      activateModel('frame-sampler');
      setProgress(15);

      // Make API call to Railway backend
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server error: ${response.status}`);
      }

      // Stage 3: Processing
      setProcessingStage('Processing Video');
      activateModel('face-detector');
      activateModel('audio-extractor');
      setProgress(30);

      const result = await response.json();

      // Stage 4: Baseline Analysis
      setProcessingStage('Baseline Analysis');
      activateModel('bg-model');
      setProgress(40);
      await new Promise(resolve => setTimeout(resolve, 300));

      // Stage 5: Routing
      setProcessingStage('Intelligent Routing');
      activateModel('routing-engine');
      activateModel('langgraph');
      setProgress(50);
      await new Promise(resolve => setTimeout(resolve, 300));

      // Stage 6: Specialist Models
      setProcessingStage('Specialist Analysis');
      const modelsUsed = result.models_used || ['BG-Model'];
      for (let i = 0; i < modelsUsed.length; i++) {
        const modelKey = modelsUsed[i].toLowerCase().replace('-model', '-model');
        activateModel(modelKey.replace('-model', '-model'));
        setProgress(55 + (i + 1) * 5);
        await new Promise(resolve => setTimeout(resolve, 200));
      }

      // Stage 7: Aggregation
      setProcessingStage('Result Aggregation');
      activateModel('aggregator');
      setProgress(80);
      await new Promise(resolve => setTimeout(resolve, 300));

      // Stage 8: Explanation
      setProcessingStage('Generating Explanation');
      activateModel('explainer');
      setProgress(90);
      await new Promise(resolve => setTimeout(resolve, 200));

      // Stage 9: Final
      setProcessingStage('Finalizing Results');
      activateModel('api-response');
      activateModel('heatmap');
      setProgress(100);

      // Determine best model from predictions
      const modelPredictions = result.analysis?.model_predictions || {};
      let bestModel = 'bg-model';
      let highestConf = 0;
      Object.entries(modelPredictions).forEach(([model, conf]: [string, any]) => {
        if (conf > highestConf) {
          highestConf = conf;
          bestModel = model.toLowerCase();
        }
      });

      // Generate explanation based on result
      const predictionLabel = result.prediction === 'fake' ? 'MANIPULATED (FAKE)' : 'AUTHENTIC (REAL)';
      const confidencePercent = (result.confidence * 100).toFixed(1);
      const explanation = result.prediction === 'fake'
        ? `This video is classified as ${predictionLabel} with ${confidencePercent}% confidence. Analysis performed by ${result.models_used?.length || 1} specialist model(s). Detected inconsistencies suggest potential manipulation.`
        : `This video is classified as ${predictionLabel} with ${confidencePercent}% confidence. No significant manipulation artifacts detected across ${result.analysis?.frames_analyzed || 30} analyzed frames.`;

      setAnalysisResult({
        prediction: result.prediction,
        confidence: result.confidence,
        best_model: bestModel,
        specialists_used: result.models_used || ['BG-Model'],
        processing_time: result.processing_time || 2.0,
        explanation,
        raw_result: result, // Keep full API response
      });

      setProcessingStage('Analysis Complete');

      // Save to Supabase database
      try {
        const analysisRecord: VideoAnalysis = {
          filename: selectedFile.name,
          file_size: selectedFile.size,
          prediction: result.prediction,
          confidence: result.confidence,
          models_used: result.models_used || ['BG-Model'],
          processing_time: result.processing_time || 2.0,
          analysis_result: result,
          user_ip: 'web-client' // Could be enhanced to get real IP
        };
        
        await saveAnalysis(analysisRecord);
        console.log('Analysis saved to database');
      } catch (dbError) {
        console.error('Failed to save to database:', dbError);
        // Don't show error to user, just log it
      }

      // Scroll to results section when complete
      setTimeout(() => {
        resultsRef.current?.scrollIntoView({
          behavior: 'smooth',
          block: 'start'
        });
      }, 500);

    } catch (err: any) {
      console.error('Analysis error:', err);
      setError(err.message || 'Failed to analyze video. Please try again.');
      setProcessingStage('Error');
      resetFlow();
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleAnalyze = () => {
    if (selectedFile) {
      // Scroll to animation section when analysis starts
      setTimeout(() => {
        animationRef.current?.scrollIntoView({
          behavior: 'smooth',
          block: 'center'
        });
      }, 300);
      analyzeVideo();
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setAnalysisResult(null);
    setProgress(0);
    setError(null);
    setIsDuplicate(false);
    setDuplicateResult(null);
    resetFlow();
  };

  const handleDownloadReport = () => {
    if (!analysisResult) return;
    
    const report = {
      video: selectedFile?.name || 'Unknown',
      timestamp: new Date().toISOString(),
      result: analysisResult.prediction,
      confidence: analysisResult.confidence,
      best_model: analysisResult.best_model,
      specialists_used: analysisResult.specialists_used,
      processing_time: analysisResult.processing_time,
      explanation: analysisResult.explanation
    };
    
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `analysis-report-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen pt-20 sm:pt-28 lg:pt-32 pb-12 sm:pb-16 lg:pb-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-8">
        {/* Header */}
        <div>
          <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 dark:text-white mb-3 sm:mb-4">
            Video Analysis
          </h1>
          <p className="text-sm sm:text-base text-gray-600 dark:text-gray-400">
            Upload your video for instant deepfake detection. Supports MP4, AVI, MOV, and WebM up to 100MB.
          </p>
        </div>

        {/* Upload Section */}
        <div
          onDragEnter={handleDragEnter}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={`relative border-2 border-dashed rounded-xl sm:rounded-2xl p-8 sm:p-12 lg:p-16 transition-all backdrop-blur-md ${
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
                <FileVideo className="w-12 h-12 sm:w-16 sm:h-16 text-blue-600 dark:text-blue-400 mb-3 sm:mb-4" />
                <p className="text-base sm:text-lg text-gray-900 dark:text-white mb-2">{selectedFile.name}</p>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                </p>
              </div>
            ) : (
              <div className="flex flex-col items-center">
                <Upload className="w-12 h-12 sm:w-16 sm:h-16 text-gray-400 dark:text-gray-500 mb-3 sm:mb-4" />
                <p className="text-base sm:text-lg text-gray-900 dark:text-white mb-2">
                  Drag and drop your video or click to browse
                </p>
                <p className="text-xs sm:text-sm text-gray-500 dark:text-gray-400">
                  Maximum file size: 100MB
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Action Buttons */}
        {!isAnalyzing && !analysisResult && (
          <div className="flex flex-col sm:flex-row gap-3 sm:gap-4">
            <button
              onClick={handleAnalyze}
              disabled={!selectedFile}
              className={`flex-1 px-6 py-3 rounded-xl transition-colors shadow-lg text-sm sm:text-base ${
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
                className="px-6 py-3 bg-white/50 dark:bg-gray-900/50 backdrop-blur-md hover:bg-white/70 dark:hover:bg-gray-900/70 text-gray-900 dark:text-white border border-gray-200 dark:border-gray-800 rounded-xl transition-colors text-sm sm:text-base"
              >
                Clear
              </button>
            )}
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="bg-red-50/50 dark:bg-red-900/20 backdrop-blur-md border border-red-200 dark:border-red-800 rounded-xl sm:rounded-2xl p-6 sm:p-8">
            <div className="flex items-start gap-3">
              <AlertTriangle className="w-6 h-6 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
              <div>
                <h3 className="text-lg font-semibold text-red-800 dark:text-red-200 mb-1">Analysis Failed</h3>
                <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
                <button
                  onClick={handleReset}
                  className="mt-3 px-4 py-2 bg-red-100 dark:bg-red-900/30 hover:bg-red-200 dark:hover:bg-red-900/50 text-red-800 dark:text-red-200 rounded-lg transition-colors text-sm"
                >
                  Try Again
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Processing Pipeline Animation - Shown during analysis */}
        {isAnalyzing && (
          <div ref={animationRef} className="space-y-6">
            {/* Processing Progress */}
            <div className="bg-white/50 dark:bg-gray-900/50 backdrop-blur-md border border-gray-200 dark:border-gray-800 rounded-xl sm:rounded-2xl p-6 sm:p-8">
              <h2 className="text-lg sm:text-xl font-semibold text-gray-900 dark:text-white mb-4">
                Analyzing Video
              </h2>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-xs sm:text-sm text-gray-600 dark:text-gray-400 mb-2">
                    <span className="flex items-center gap-2">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      {state.processingStage || 'Processing...'}
                    </span>
                    <span>{progress}%</span>
                  </div>
                  <Progress value={progress} className="h-2" />
                </div>
              </div>
            </div>

            {/* Architecture Visualization */}
            <div className="bg-white/70 dark:bg-gray-900/70 backdrop-blur-md border border-gray-200 dark:border-gray-800 rounded-xl sm:rounded-2xl p-4 sm:p-6 shadow-xl">
              <div className="flex items-center justify-between mb-3 sm:mb-4">
                <h3 className="text-base sm:text-lg font-semibold text-gray-900 dark:text-white">
                  Processing Pipeline
                </h3>
                {state.processingStage !== 'idle' && (
                  <span className="px-2 sm:px-3 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded-full text-xs sm:text-sm font-medium">
                    {state.processingStage}
                  </span>
                )}
              </div>
              <div className="relative h-[250px] sm:h-[320px] md:h-[400px] flex items-center justify-center rounded-lg sm:rounded-xl overflow-hidden border border-gray-200 dark:border-gray-700">
                <SystemArchitectureCanvas />
              </div>
            </div>
          </div>
        )}

        {/* Results Section */}
        {analysisResult && (
          <div ref={resultsRef} className="space-y-6">
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

              {/* Duplicate File Indicator */}
              {isDuplicate && (
                <div className="mb-6 p-4 bg-blue-50/50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-xl">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 bg-blue-100 dark:bg-blue-900/50 rounded-full flex items-center justify-center">
                      <FileVideo className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                    </div>
                    <div>
                      <p className="text-sm font-medium text-blue-900 dark:text-blue-100">
                        Previously Analyzed File
                      </p>
                      <p className="text-xs text-blue-700 dark:text-blue-300">
                        This file was analyzed on {duplicateResult ? new Date(duplicateResult.created_at).toLocaleDateString() : 'a previous date'}. Showing cached results.
                      </p>
                    </div>
                  </div>
                </div>
              )}

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

              <div className="grid grid-cols-3 gap-4 mb-6">
                <div className="bg-gray-50/50 dark:bg-gray-800/50 backdrop-blur-md rounded-xl p-4">
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">Confidence</p>
                  <p className="text-xl font-semibold text-gray-900 dark:text-white">
                    {(analysisResult.confidence * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="bg-gray-50/50 dark:bg-gray-800/50 backdrop-blur-md rounded-xl p-4">
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">Best Model</p>
                  <p className="text-xl font-semibold text-gray-900 dark:text-white">
                    {analysisResult.best_model.toUpperCase()}
                  </p>
                </div>
                <div className="bg-gray-50/50 dark:bg-gray-800/50 backdrop-blur-md rounded-xl p-4">
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">Time</p>
                  <p className="text-xl font-semibold text-gray-900 dark:text-white">
                    {analysisResult.processing_time}s
                  </p>
                </div>
              </div>

              <div className="bg-blue-50/50 dark:bg-blue-900/20 backdrop-blur-md rounded-xl p-4">
                <p className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                  Explanation
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  {analysisResult.explanation}
                </p>
              </div>
            </div>

            <button
              onClick={handleDownloadReport}
              className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-white/50 dark:bg-gray-900/50 backdrop-blur-md hover:bg-white/70 dark:hover:bg-gray-900/70 text-gray-900 dark:text-white border border-gray-200 dark:border-gray-800 rounded-xl transition-colors"
            >
              <Download className="w-4 h-4" />
              Download Report
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default AnalysisWorkbench;