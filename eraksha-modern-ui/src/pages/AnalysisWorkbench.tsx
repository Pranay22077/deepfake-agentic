import React, { useState, useCallback } from 'react';
import { Upload, FileVideo, CheckCircle2, XCircle, Download } from 'lucide-react';
import { Progress } from '../components/ui/progress';

const AnalysisWorkbench = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [analysisResult, setAnalysisResult] = useState<any>(null);

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

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      // Simulate progress
      const progressInterval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 300);

      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });

      clearInterval(progressInterval);
      setProgress(100);

      if (response.ok) {
        const result = await response.json();
        console.log('API Response:', result); // Debug log
        
        // Transform the API response to match our UI format
        setAnalysisResult({
          prediction: result.prediction,
          confidence: result.confidence / 100, // Convert back to decimal
          best_model: result.details?.best_model || 'Unknown',
          specialists_used: result.details?.specialists_used || [],
          explanation: result.explanation,
          filename: result.filename,
          file_size: selectedFile.size,
          processing_time: result.details?.processing_time,
          all_predictions: result.details?.all_predictions,
          video_characteristics: result.details?.video_characteristics,
          bias_correction: result.bias_correction
        });
      } else {
        const errorData = await response.json().catch(() => ({}));
        console.error('API Error:', response.status, errorData);
        throw new Error(`API Error: ${response.status} - ${errorData.detail || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Analysis failed:', error);
      // Show error to user instead of fake results
      setAnalysisResult({
        error: true,
        message: `Analysis failed: ${error.message}. Make sure the backend server is running on http://localhost:8000`,
        prediction: 'error',
        confidence: 0,
        filename: selectedFile.name,
        file_size: selectedFile.size,
      });
    }

    setIsAnalyzing(false);
  };

  const handleReset = () => {
    setSelectedFile(null);
    setAnalysisResult(null);
    setProgress(0);
  };

  return (
    <div className="min-h-screen pt-32 pb-20">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="mb-12">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
            Video Analysis
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            Upload your video for instant deepfake detection. Supports MP4, AVI, MOV, and WebM up to 100MB.
          </p>
        </div>

        {!analysisResult ? (
          <div className="space-y-8">
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

            {/* Processing Progress */}
            {isAnalyzing && (
              <div className="bg-white/50 dark:bg-gray-900/50 backdrop-blur-md border border-gray-200 dark:border-gray-800 rounded-2xl p-8">
                <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
                  Analyzing Video
                </h2>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400 mb-2">
                      <span>Processing with agentic system...</span>
                      <span>{progress}%</span>
                    </div>
                    <Progress value={progress} className="h-2" />
                  </div>
                </div>
              </div>
            )}

            {/* Action Buttons */}
            {!isAnalyzing && (
              <div className="flex gap-4">
                <button
                  onClick={analyzeVideo}
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
        ) : (
          <div className="space-y-8">
            {/* Error Display */}
            {analysisResult.error ? (
              <div className="bg-red-50/50 dark:bg-red-900/20 backdrop-blur-md border border-red-200 dark:border-red-800 rounded-2xl p-8">
                <div className="flex items-center gap-4 mb-4">
                  <div className="w-16 h-16 rounded-full flex items-center justify-center bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400">
                    <XCircle className="w-8 h-8" />
                  </div>
                  <div>
                    <p className="text-sm text-red-600 dark:text-red-400">Error</p>
                    <p className="text-2xl font-bold text-red-900 dark:text-red-100">Analysis Failed</p>
                  </div>
                </div>
                <p className="text-red-800 dark:text-red-200 mb-4">{analysisResult.message}</p>
                <button
                  onClick={handleReset}
                  className="px-4 py-2 bg-red-100/50 dark:bg-red-800/50 backdrop-blur-md hover:bg-red-200/50 dark:hover:bg-red-700/50 text-red-900 dark:text-red-100 rounded-lg transition-colors"
                >
                  Try Again
                </button>
              </div>
            ) : (
              <>
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
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">Best Model</p>
                  <p className="text-xl font-semibold text-gray-900 dark:text-white">
                    {analysisResult.best_model}
                  </p>
                </div>
                <div className="bg-gray-50/50 dark:bg-gray-800/50 backdrop-blur-md rounded-xl p-4">
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">Specialists</p>
                  <p className="text-xl font-semibold text-gray-900 dark:text-white">
                    {analysisResult.specialists_used?.length || 1}
                  </p>
                </div>
                <div className="bg-gray-50/50 dark:bg-gray-800/50 backdrop-blur-md rounded-xl p-4">
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">File Size</p>
                  <p className="text-xl font-semibold text-gray-900 dark:text-white">
                    {((analysisResult.file_size || 0) / (1024 * 1024)).toFixed(1)}MB
                  </p>
                </div>
                {analysisResult.processing_time && (
                  <div className="bg-gray-50/50 dark:bg-gray-800/50 backdrop-blur-md rounded-xl p-4">
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">Processing Time</p>
                    <p className="text-xl font-semibold text-gray-900 dark:text-white">
                      {analysisResult.processing_time}s
                    </p>
                  </div>
                )}
              </div>
            </div>

            {/* Detailed Analysis */}
            <div className="bg-white/50 dark:bg-gray-900/50 backdrop-blur-md border border-gray-200 dark:border-gray-800 rounded-2xl p-8">
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-6">
                Detailed Analysis
              </h3>
              <div className="space-y-4">
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Explanation</p>
                  <p className="text-gray-900 dark:text-white">
                    {analysisResult.explanation || 'Analysis completed using agentic system with multiple specialist models.'}
                  </p>
                  {analysisResult.bias_correction && (
                    <div className="mt-3 px-3 py-1 bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400 rounded-full text-sm inline-block">
                      ✓ Bias correction applied
                    </div>
                  )}
                </div>
                {analysisResult.specialists_used && (
                  <div>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Specialists Used</p>
                    <div className="flex flex-wrap gap-2">
                      {analysisResult.specialists_used.map((specialist: string, index: number) => (
                        <span
                          key={index}
                          className="px-3 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 rounded-full text-sm"
                        >
                          {specialist}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Actions */}
            <div className="flex gap-4">
              <button className="flex items-center gap-2 px-6 py-3 bg-white/50 dark:bg-gray-900/50 backdrop-blur-md hover:bg-white/70 dark:hover:bg-gray-900/70 text-gray-900 dark:text-white border border-gray-200 dark:border-gray-800 rounded-xl transition-colors">
                <Download className="w-4 h-4" />
                Download Report
              </button>
            </div>
            </>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default AnalysisWorkbench;