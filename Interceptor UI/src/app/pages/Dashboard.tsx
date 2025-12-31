import React from 'react';
import { BarChart3, TrendingUp, Activity, CheckCircle, XCircle, Clock, FileVideo, AlertCircle } from 'lucide-react';

const Dashboard = () => {
  const stats = [
    {
      label: 'Total Videos',
      value: '1,247',
      icon: <FileVideo className="w-5 h-5" />,
      change: '+12%',
    },
    {
      label: 'Detected Fakes',
      value: '342',
      icon: <AlertCircle className="w-5 h-5" />,
      change: '+8%',
    },
    {
      label: 'Avg Processing',
      value: '3.2s',
      icon: <Clock className="w-5 h-5" />,
      change: '-15%',
    },
  ];

  const recentAnalyses = [
    {
      id: '1',
      filename: 'interview_video.mp4',
      timestamp: '2 hours ago',
      prediction: 'real',
      confidence: 0.92,
    },
    {
      id: '2',
      filename: 'news_clip.mov',
      timestamp: '5 hours ago',
      prediction: 'fake',
      confidence: 0.78,
    },
    {
      id: '3',
      filename: 'presentation.webm',
      timestamp: '1 day ago',
      prediction: 'real',
      confidence: 0.85,
    },
    {
      id: '4',
      filename: 'social_media.mp4',
      timestamp: '2 days ago',
      prediction: 'fake',
      confidence: 0.91,
    },
  ];

  return (
    <div className="min-h-screen pt-20 sm:pt-28 lg:pt-32 pb-12 sm:pb-16 lg:pb-20">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="mb-8 sm:mb-12">
          <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 dark:text-white mb-3 sm:mb-4">
            Dashboard
          </h1>
          <p className="text-sm sm:text-base text-gray-600 dark:text-gray-400">
            Track your analysis history and monitor detection performance
          </p>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 sm:gap-6 mb-8 sm:mb-12">
          {stats.map((stat, index) => (
            <div
              key={index}
              className="bg-white/50 dark:bg-gray-900/50 backdrop-blur-md border border-gray-200 dark:border-gray-800 rounded-xl sm:rounded-2xl p-5 sm:p-6"
            >
              <div className="flex items-center justify-between mb-3 sm:mb-4">
                <div className="w-9 h-9 sm:w-10 sm:h-10 bg-blue-100 dark:bg-blue-900/30 rounded-lg sm:rounded-xl flex items-center justify-center text-blue-600 dark:text-blue-400">
                  {stat.icon}
                </div>
                <span className="text-xs sm:text-sm font-semibold text-green-600 dark:text-green-400">
                  {stat.change}
                </span>
              </div>
              <p className="text-xs sm:text-sm text-gray-600 dark:text-gray-400 mb-1">{stat.label}</p>
              <p className="text-2xl sm:text-3xl font-bold text-gray-900 dark:text-white">{stat.value}</p>
            </div>
          ))}
        </div>

        {/* Analysis History */}
        <div className="bg-white/50 dark:bg-gray-900/50 backdrop-blur-md border border-gray-200 dark:border-gray-800 rounded-xl sm:rounded-2xl p-5 sm:p-8">
          <div className="flex items-center justify-between mb-4 sm:mb-6">
            <h2 className="text-xl sm:text-2xl font-bold text-gray-900 dark:text-white">
              Recent Analyses
            </h2>
          </div>

          <div className="overflow-x-auto -mx-5 sm:mx-0">
            <table className="w-full min-w-[600px]">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-800">
                  <th className="text-left py-3 px-4 text-xs sm:text-sm font-semibold text-gray-600 dark:text-gray-400">
                    Filename
                  </th>
                  <th className="text-left py-3 px-4 text-xs sm:text-sm font-semibold text-gray-600 dark:text-gray-400">
                    Time
                  </th>
                  <th className="text-left py-3 px-4 text-xs sm:text-sm font-semibold text-gray-600 dark:text-gray-400">
                    Result
                  </th>
                  <th className="text-left py-3 px-4 text-xs sm:text-sm font-semibold text-gray-600 dark:text-gray-400">
                    Confidence
                  </th>
                  <th className="text-left py-3 px-4 text-xs sm:text-sm font-semibold text-gray-600 dark:text-gray-400">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody>
                {recentAnalyses.map((analysis) => (
                  <tr
                    key={analysis.id}
                    className="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50/50 dark:hover:bg-gray-800/50 transition-colors"
                  >
                    <td className="py-4 px-4">
                      <div className="flex items-center gap-3">
                        <div className="w-8 h-8 bg-blue-100 dark:bg-blue-900/30 rounded-lg flex items-center justify-center">
                          <FileVideo className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                        </div>
                        <span className="text-gray-900 dark:text-white text-sm">
                          {analysis.filename}
                        </span>
                      </div>
                    </td>
                    <td className="py-4 px-4 text-sm text-gray-600 dark:text-gray-400">
                      {analysis.timestamp}
                    </td>
                    <td className="py-4 px-4">
                      <span
                        className={`inline-flex px-3 py-1 rounded-full text-xs font-semibold ${
                          analysis.prediction === 'fake'
                            ? 'bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400'
                            : 'bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400'
                        }`}
                      >
                        {analysis.prediction}
                      </span>
                    </td>
                    <td className="py-4 px-4 text-sm text-gray-900 dark:text-white">
                      {(analysis.confidence * 100).toFixed(1)}%
                    </td>
                    <td className="py-4 px-4">
                      <button className="text-sm text-blue-600 dark:text-blue-400 hover:underline">
                        View
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;