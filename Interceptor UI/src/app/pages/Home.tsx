import React from 'react';
import { Link } from 'react-router-dom';
import { ShieldCheck, Zap, Eye, ArrowRight, Play } from 'lucide-react';

const Home = () => {
  return (
    <div className="min-h-screen pt-20 sm:pt-28 lg:pt-32 pb-12 sm:pb-16 lg:pb-20">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Hero Section */}
        <div className="text-center mb-12 sm:mb-16 lg:mb-20">
          <div className="inline-flex items-center gap-2 px-3 sm:px-4 py-1.5 sm:py-2 bg-blue-50 dark:bg-blue-900/30 border border-blue-200 dark:border-blue-800 rounded-full mb-4 sm:mb-6">
            <ShieldCheck className="w-3 h-3 sm:w-4 sm:h-4 text-blue-600 dark:text-blue-400" />
            <span className="text-xs sm:text-sm text-blue-700 dark:text-blue-300">
              Advanced AI Detection System
            </span>
          </div>

          <h1 className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-bold text-gray-900 dark:text-white mb-4 sm:mb-6 px-4">
            Intelligent Video
            <br />
            <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              Authenticity Analysis
            </span>
          </h1>

          <p className="text-base sm:text-lg lg:text-xl text-gray-600 dark:text-gray-400 mb-8 sm:mb-10 max-w-3xl mx-auto px-4">
            AI-assisted analysis to detect deepfakes and manipulated videos, with transparent confidence scoring and explanations.
          </p>

          <div className="flex flex-col sm:flex-row gap-3 sm:gap-4 justify-center px-4">
            <Link
              to="/workbench"
              className="group px-6 sm:px-8 py-3 sm:py-4 bg-blue-600 hover:bg-blue-700 text-white rounded-xl transition-all shadow-lg hover:shadow-xl flex items-center justify-center gap-2"
            >
              <Play className="w-4 h-4 sm:w-5 sm:h-5" />
              <span className="text-sm sm:text-base">Start Analysis</span>
              <ArrowRight className="w-4 h-4 sm:w-5 sm:h-5 group-hover:translate-x-1 transition-transform" />
            </Link>
            <Link
              to="/analytics"
              className="px-6 sm:px-8 py-3 sm:py-4 bg-white/70 dark:bg-gray-900/70 backdrop-blur-md hover:bg-white/90 dark:hover:bg-gray-900/90 text-gray-900 dark:text-white border border-gray-200 dark:border-gray-800 rounded-xl transition-all flex items-center justify-center gap-2"
            >
              <span className="text-sm sm:text-base">View Statistics</span>
            </Link>
          </div>
        </div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6 sm:gap-8">
          <div className="bg-white/70 dark:bg-gray-900/70 backdrop-blur-md border border-gray-200 dark:border-gray-800 rounded-xl sm:rounded-2xl p-6 sm:p-8 hover:shadow-xl transition-all">
            <div className="w-10 h-10 sm:w-12 sm:h-12 bg-blue-100 dark:bg-blue-900/30 rounded-lg sm:rounded-xl flex items-center justify-center mb-3 sm:mb-4">
              <ShieldCheck className="w-5 h-5 sm:w-6 sm:h-6 text-blue-600 dark:text-blue-400" />
            </div>
            <h3 className="text-lg sm:text-xl font-bold text-gray-900 dark:text-white mb-2 sm:mb-3">
              Advanced Detection
            </h3>
            <p className="text-sm sm:text-base text-gray-600 dark:text-gray-400">
              6 specialist models (BG, AV, CM, RR, LL, TM) working in harmony for comprehensive analysis
            </p>
          </div>

          <div className="bg-white/70 dark:bg-gray-900/70 backdrop-blur-md border border-gray-200 dark:border-gray-800 rounded-xl sm:rounded-2xl p-6 sm:p-8 hover:shadow-xl transition-all">
            <div className="w-10 h-10 sm:w-12 sm:h-12 bg-purple-100 dark:bg-purple-900/30 rounded-lg sm:rounded-xl flex items-center justify-center mb-3 sm:mb-4">
              <Zap className="w-5 h-5 sm:w-6 sm:h-6 text-purple-600 dark:text-purple-400" />
            </div>
            <h3 className="text-lg sm:text-xl font-bold text-gray-900 dark:text-white mb-2 sm:mb-3">
              Intelligent Routing
            </h3>
            <p className="text-sm sm:text-base text-gray-600 dark:text-gray-400">
              AI-powered routing engine with LangGraph orchestration for optimal model selection
            </p>
          </div>

          <div className="bg-white/70 dark:bg-gray-900/70 backdrop-blur-md border border-gray-200 dark:border-gray-800 rounded-xl sm:rounded-2xl p-6 sm:p-8 hover:shadow-xl transition-all sm:col-span-2 md:col-span-1">
            <div className="w-10 h-10 sm:w-12 sm:h-12 bg-cyan-100 dark:bg-cyan-900/30 rounded-lg sm:rounded-xl flex items-center justify-center mb-3 sm:mb-4">
              <Eye className="w-5 h-5 sm:w-6 sm:h-6 text-cyan-600 dark:text-cyan-400" />
            </div>
            <h3 className="text-lg sm:text-xl font-bold text-gray-900 dark:text-white mb-2 sm:mb-3">
              Visual Insights
            </h3>
            <p className="text-sm sm:text-base text-gray-600 dark:text-gray-400">
              Real-time heatmap generation and detailed explanations for every detection
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;