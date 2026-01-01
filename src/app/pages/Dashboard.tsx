import { FileVideo, Clock, AlertCircle } from 'lucide-react';
import { RechartsDonutChart } from '../components/charts/RechartsDonutChart';
import MetadataSummary from '../components/charts/MetadataSummary';
import { getAnalyticsStats, getRecentAnalyses, formatRelativeTime } from '../../lib/supabase';
import { useState, useEffect } from 'react';

const Dashboard = () => {
  const [stats, setStats] = useState({
    totalAnalyses: 0,
    fakeDetected: 0,
    realDetected: 0,
    recentAnalyses: 0,
    averageConfidence: 0
  });
  const [recentAnalyses, setRecentAnalyses] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const [analyticsData, recentData] = await Promise.all([
          getAnalyticsStats(),
          getRecentAnalyses(10)
        ]);
        
        setStats(analyticsData);
        setRecentAnalyses(recentData);
      } catch (error) {
        console.error('Error fetching dashboard data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    
    // Refresh data every 30 seconds
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, []);

  const displayStats = [
    {
      label: 'Total Videos',
      value: loading ? '...' : stats.totalAnalyses.toLocaleString(),
      icon: <FileVideo className="w-5 h-5" />,
      change: '+' + stats.recentAnalyses,
    },
    {
      label: 'Detected Fakes',
      value: loading ? '...' : stats.fakeDetected.toLocaleString(),
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

  return (
    <div className="min-h-screen pt-32 pb-20">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="mb-12">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
            Dashboard
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            Track your analysis history and monitor detection performance
          </p>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          {displayStats.map((stat, index) => (
            <div
              key={index}
              className="bg-white/50 dark:bg-gray-900/50 backdrop-blur-md border border-gray-200 dark:border-gray-800 rounded-2xl p-6"
            >
              <div className="flex items-center justify-between mb-4">
                <div className="w-10 h-10 bg-blue-100 dark:bg-blue-900/30 rounded-xl flex items-center justify-center text-blue-600 dark:text-blue-400">
                  {stat.icon}
                </div>
                <span className="text-sm font-semibold text-green-600 dark:text-green-400">
                  {stat.change}
                </span>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">{stat.label}</p>
              <p className="text-3xl font-bold text-gray-900 dark:text-white">{stat.value}</p>
            </div>
          ))}
        </div>

        {/* Metadata Analysis Section */}
        <div className="mb-12 animate-fade-in-up">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            Metadata Analysis Dashboard
          </h2>
          
          {/* Summary Overview */}
          <div className="mb-8 animate-scale-in">
            <MetadataSummary
              data={{
                totalVideos: stats.totalAnalyses,
                fakeDetected: stats.fakeDetected,
                realDetected: stats.realDetected,
                averageConfidence: stats.averageConfidence,
                averageProcessingTime: 3.2,
                topPerformingModel: 'TM Model',
                trend: 'up',
                trendPercentage: 12.5
              }}
            />
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
            {/* Detection Results Distribution */}
            <div className="animate-slide-in-right">
              <RechartsDonutChart
                data={[
                  { label: 'Real Videos', value: stats.realDetected, color: '#10B981' },
                  { label: 'Fake Videos', value: stats.fakeDetected, color: '#EF4444' }
                ]}
                title="Detection Results"
                description="Real vs Fake video classification"
                totalLabel="Videos"
                trendText="Detection accuracy improved"
                trendPercentage="+8.2%"
              />
            </div>

            {/* Confidence Level Distribution */}
            <div className="animate-slide-in-right" style={{ animationDelay: '0.1s' }}>
              <RechartsDonutChart
                data={[
                  { label: 'High Confidence', value: Math.floor(stats.totalAnalyses * 0.45), color: '#059669' },
                  { label: 'Medium Confidence', value: Math.floor(stats.totalAnalyses * 0.35), color: '#F59E0B' },
                  { label: 'Low Confidence', value: Math.floor(stats.totalAnalyses * 0.20), color: '#EF4444' }
                ]}
                title="Confidence Levels"
                description="Analysis confidence distribution"
                totalLabel="Analyses"
                trendText="High confidence increased"
                trendPercentage="+12.5%"
              />
            </div>

            {/* Processing Speed Distribution */}
            <div className="animate-slide-in-right" style={{ animationDelay: '0.2s' }}>
              <RechartsDonutChart
                data={[
                  { label: 'Fast Processing', value: Math.floor(stats.totalAnalyses * 0.40), color: '#3B82F6' },
                  { label: 'Medium Speed', value: Math.floor(stats.totalAnalyses * 0.45), color: '#8B5CF6' },
                  { label: 'Slower Processing', value: Math.floor(stats.totalAnalyses * 0.15), color: '#F97316' }
                ]}
                title="Processing Speed"
                description="Analysis processing time breakdown"
                totalLabel="Videos"
                trendText="Processing speed improved"
                trendPercentage="+15.3%"
              />
            </div>
          </div>


        </div>

        {/* Analysis History */}
        <div className="bg-white/50 dark:bg-gray-900/50 backdrop-blur-md border border-gray-200 dark:border-gray-800 rounded-2xl p-8">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
              Recent Analyses
            </h2>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-800">
                  <th className="text-left py-3 px-4 text-sm font-semibold text-gray-600 dark:text-gray-400">
                    Filename
                  </th>
                  <th className="text-left py-3 px-4 text-sm font-semibold text-gray-600 dark:text-gray-400">
                    Time
                  </th>
                  <th className="text-left py-3 px-4 text-sm font-semibold text-gray-600 dark:text-gray-400">
                    Result
                  </th>
                  <th className="text-left py-3 px-4 text-sm font-semibold text-gray-600 dark:text-gray-400">
                    Confidence
                  </th>
                  <th className="text-left py-3 px-4 text-sm font-semibold text-gray-600 dark:text-gray-400">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody>
                {recentAnalyses.length > 0 ? recentAnalyses.map((analysis) => (
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
                      {formatRelativeTime(analysis.created_at)}
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
                )) : (
                  <tr>
                    <td colSpan={5} className="py-8 px-4 text-center text-gray-500 dark:text-gray-400">
                      {loading ? 'Loading recent analyses...' : 'No analyses found. Upload a video to get started!'}
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;