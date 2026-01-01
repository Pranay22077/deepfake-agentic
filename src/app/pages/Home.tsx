import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { ShieldCheck, Zap, Eye, ArrowRight } from 'lucide-react';
import { getAnalyticsStats } from '../../lib/supabase';

const Home = () => {
  const [stats, setStats] = useState({
    totalAnalyses: 0,
    fakeDetected: 0,
    realDetected: 0,
    recentAnalyses: 0,
    averageConfidence: 0
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const data = await getAnalyticsStats();
        setStats(data);
      } catch (error) {
        console.error('Error fetching stats:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchStats();
  }, []);

  const features = [
    {
      icon: <ShieldCheck className="w-6 h-6" />,
      title: 'Agentic AI Detection',
      description: `6 specialist neural networks with ${loading ? '94.9' : (stats.averageConfidence * 100).toFixed(1)}% overall detection confidence powered by LangGraph`,
    },
    {
      icon: <Eye className="w-6 h-6" />,
      title: 'Intelligent Routing',
      description: 'Smart agent routes videos to specialized models based on compression, lighting & temporal analysis',
    },
    {
      icon: <Zap className="w-6 h-6" />,
      title: 'Real-time Processing',
      description: 'Average 2.1 second processing with Grad-CAM heatmaps for explainable results',
    },
  ];

  return (
    <div className="min-h-screen">
      {/* Hero Section - Full viewport height */}
      <section className="min-h-screen flex items-center justify-center px-4 sm:px-6 lg:px-8">
        <div className="max-w-5xl mx-auto text-center">
          <h1 className="text-5xl md:text-6xl lg:text-7xl font-bold mb-2 text-gray-900 dark:text-white leading-tight">
            Intelligent Video
          </h1>
          <h1 className="text-5xl md:text-6xl lg:text-7xl font-bold mb-6 leading-tight">
            <span className="text-gray-900 dark:text-white">Authenticity </span>
            <span className="text-blue-600 dark:text-blue-400">Analysis</span>
          </h1>
          <p className="text-base md:text-lg text-gray-600 dark:text-gray-300 max-w-3xl mx-auto mb-8">
            AI-assisted analysis to detect deepfakes and manipulated videos, with transparent confidence scoring and explanations.
          </p>
          <div className="flex items-center justify-center gap-4">
            <Link
              to="/workbench"
              className="px-8 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-xl transition-colors flex items-center gap-2 shadow-lg"
            >
              Start Analysis
              <ArrowRight className="w-4 h-4" />
            </Link>
            <Link
              to="/analytics"
              className="px-8 py-3 bg-white/50 dark:bg-gray-900/50 backdrop-blur-md hover:bg-white/70 dark:hover:bg-gray-900/70 text-gray-900 dark:text-white rounded-xl transition-colors shadow-lg border border-gray-200 dark:border-gray-800"
            >
              View Statistics
            </Link>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-6xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <div
                key={index}
                className="bg-white/50 dark:bg-gray-900/50 backdrop-blur-md border border-gray-200 dark:border-gray-800 rounded-2xl p-8 hover:shadow-xl transition-shadow"
              >
                <div className="w-12 h-12 bg-blue-100 dark:bg-blue-900/30 rounded-xl flex items-center justify-center text-blue-600 dark:text-blue-400 mb-4">
                  {feature.icon}
                </div>
                <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                  {feature.title}
                </h3>
                <p className="text-gray-600 dark:text-gray-400">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-6xl mx-auto">
          <div className="bg-white/50 dark:bg-gray-900/50 backdrop-blur-md border border-gray-200 dark:border-gray-800 rounded-2xl p-12">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-8 text-center">
              <div>
                <p className="text-4xl font-bold text-blue-600 dark:text-blue-400 mb-2">
                  {loading ? '...' : stats.totalAnalyses.toLocaleString()}
                </p>
                <p className="text-gray-600 dark:text-gray-400">Videos Analyzed</p>
              </div>
              <div>
                <p className="text-4xl font-bold text-blue-600 dark:text-blue-400 mb-2">
                  {loading ? '94.9' : (stats.averageConfidence * 100).toFixed(1)}%
                </p>
                <p className="text-gray-600 dark:text-gray-400">Detection Confidence</p>
              </div>
              <div>
                <p className="text-4xl font-bold text-blue-600 dark:text-blue-400 mb-2">
                  2.1s
                </p>
                <p className="text-gray-600 dark:text-gray-400">Avg Processing Time</p>
              </div>
              <div>
                <p className="text-4xl font-bold text-blue-600 dark:text-blue-400 mb-2">
                  47.2M
                </p>
                <p className="text-gray-600 dark:text-gray-400">Total Parameters</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-3xl font-bold text-center text-gray-900 dark:text-white mb-12">
            How Interceptor Works
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            {[
              { step: '1', title: 'Upload Video', desc: 'Upload your video file (MP4, AVI, MOV, WebM up to 100MB)' },
              { step: '2', title: 'Video Analysis', desc: 'Extract frames, detect faces, analyze audio & metadata' },
              { step: '3', title: 'Agentic Routing', desc: 'LangGraph agent routes to specialist models (BG, AV, CM, RR, LL, TM)' },
              { step: '4', title: 'Get Results', desc: 'Aggregated prediction with Grad-CAM heatmaps & explanations' },
            ].map((item) => (
              <div key={item.step} className="text-center">
                <div className="w-14 h-14 bg-blue-600 dark:bg-blue-500 text-white rounded-full flex items-center justify-center text-xl font-bold mx-auto mb-4">
                  {item.step}
                </div>
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                  {item.title}
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">{item.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto bg-gradient-to-r from-blue-600 to-blue-700 dark:from-blue-700 dark:to-blue-800 rounded-3xl p-12 text-center shadow-2xl">
          <h2 className="text-3xl font-bold text-white mb-4">
            Ready to Detect Deepfakes?
          </h2>
          <p className="text-blue-100 mb-8 max-w-2xl mx-auto">
            Interceptor uses agentic AI with 6 specialist models for comprehensive deepfake detection. Fast, accurate, and explainable.
          </p>
          <Link
            to="/workbench"
            className="inline-flex items-center gap-2 px-8 py-3 bg-white hover:bg-gray-100 text-blue-600 rounded-xl transition-colors shadow-lg"
          >
            Start Analysis
            <ArrowRight className="w-4 h-4" />
          </Link>
        </div>
      </section>
    </div>
  );
};

export default Home;