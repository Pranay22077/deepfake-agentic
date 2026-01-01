import React from 'react';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

interface MetadataSummaryProps {
  data: {
    totalVideos: number;
    fakeDetected: number;
    realDetected: number;
    averageConfidence: number;
    averageProcessingTime: number;
    topPerformingModel: string;
    trend: 'up' | 'down' | 'stable';
    trendPercentage: number;
  };
}

const MetadataSummary: React.FC<MetadataSummaryProps> = ({ data }) => {
  const fakePercentage = ((data.fakeDetected / data.totalVideos) * 100).toFixed(1);
  const realPercentage = ((data.realDetected / data.totalVideos) * 100).toFixed(1);

  const getTrendIcon = () => {
    switch (data.trend) {
      case 'up':
        return <TrendingUp className="w-4 h-4 text-green-500" />;
      case 'down':
        return <TrendingDown className="w-4 h-4 text-red-500" />;
      default:
        return <Minus className="w-4 h-4 text-gray-500" />;
    }
  };

  const getTrendColor = () => {
    switch (data.trend) {
      case 'up':
        return 'text-green-600 dark:text-green-400';
      case 'down':
        return 'text-red-600 dark:text-red-400';
      default:
        return 'text-gray-600 dark:text-gray-400';
    }
  };

  return (
    <div className="bg-white/50 dark:bg-gray-900/50 backdrop-blur-md border border-gray-200 dark:border-gray-800 rounded-2xl p-6">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">
        Analysis Summary
      </h3>
      
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {/* Total Videos */}
        <div className="text-center">
          <div className="text-2xl font-bold text-gray-900 dark:text-white">
            {data.totalVideos.toLocaleString()}
          </div>
          <div className="text-sm text-gray-600 dark:text-gray-400">
            Total Videos
          </div>
        </div>

        {/* Detection Rate */}
        <div className="text-center">
          <div className="text-2xl font-bold text-red-600 dark:text-red-400">
            {fakePercentage}%
          </div>
          <div className="text-sm text-gray-600 dark:text-gray-400">
            Fake Detected
          </div>
        </div>

        {/* Average Confidence */}
        <div className="text-center">
          <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
            {(data.averageConfidence * 100).toFixed(1)}%
          </div>
          <div className="text-sm text-gray-600 dark:text-gray-400">
            Avg Confidence
          </div>
        </div>

        {/* Processing Time */}
        <div className="text-center">
          <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
            {data.averageProcessingTime.toFixed(1)}s
          </div>
          <div className="text-sm text-gray-600 dark:text-gray-400">
            Avg Time
          </div>
        </div>
      </div>

      <div className="mt-6 pt-6 border-t border-gray-200 dark:border-gray-800">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm text-gray-600 dark:text-gray-400">
              Top Performing Model
            </div>
            <div className="text-lg font-semibold text-gray-900 dark:text-white">
              {data.topPerformingModel}
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            {getTrendIcon()}
            <span className={`text-sm font-medium ${getTrendColor()}`}>
              {data.trendPercentage > 0 ? '+' : ''}{data.trendPercentage.toFixed(1)}%
            </span>
          </div>
        </div>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
        <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-3">
          <div className="text-green-800 dark:text-green-200 font-medium">
            Real Videos
          </div>
          <div className="text-green-600 dark:text-green-400 text-lg font-bold">
            {data.realDetected} ({realPercentage}%)
          </div>
        </div>
        
        <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-3">
          <div className="text-red-800 dark:text-red-200 font-medium">
            Fake Videos
          </div>
          <div className="text-red-600 dark:text-red-400 text-lg font-bold">
            {data.fakeDetected} ({fakePercentage}%)
          </div>
        </div>
      </div>
    </div>
  );
};

export default MetadataSummary;