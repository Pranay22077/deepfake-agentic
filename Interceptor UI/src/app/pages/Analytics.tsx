import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell, AreaChart, Area } from 'recharts';
import { TrendingUp, Activity, Zap, CheckCircle, BarChart3 } from 'lucide-react';

const Analytics = () => {
  const stats = [
    {
      label: 'Total Videos',
      value: '1,247',
      icon: <BarChart3 className="w-5 h-5" />,
      change: '+156',
    },
    {
      label: 'Accuracy',
      value: '68.5%',
      icon: <TrendingUp className="w-5 h-5" />,
      change: '+2.3%',
    },
    {
      label: 'Avg Latency',
      value: '3.2s',
      icon: <Activity className="w-5 h-5" />,
      change: '-0.5s',
    },
  ];

  const trendData = [
    { month: 'Jan', processed: 850, accuracy: 63 },
    { month: 'Feb', processed: 920, accuracy: 64 },
    { month: 'Mar', processed: 1050, accuracy: 65 },
    { month: 'Apr', processed: 1180, accuracy: 66.5 },
    { month: 'May', processed: 1247, accuracy: 68.5 },
  ];

  const dailyData = [
    { day: 'Mon', scans: 45 },
    { day: 'Tue', scans: 52 },
    { day: 'Wed', scans: 38 },
    { day: 'Thu', scans: 61 },
    { day: 'Fri', scans: 55 },
    { day: 'Sat', scans: 42 },
    { day: 'Sun', scans: 35 },
  ];

  return (
    <div className="min-h-screen pt-32 pb-20">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="mb-12">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
            Statistics
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            Monitor usage, accuracy, and system performance in real-time
          </p>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          {stats.map((stat, index) => (
            <div
              key={index}
              className="bg-white/50 dark:bg-gray-900/50 backdrop-blur-md border border-gray-200 dark:border-gray-800 rounded-2xl p-6"
            >
              <div className="flex items-center justify-between mb-4">
                <div className="w-10 h-10 bg-blue-100 dark:bg-blue-900/30 rounded-xl flex items-center justify-center text-blue-600 dark:text-blue-400">
                  {stat.icon}
                </div>
                <div className="text-right">
                  <p className="text-sm text-green-600 dark:text-green-400 font-semibold">
                    {stat.change}
                  </p>
                </div>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">{stat.label}</p>
              <p className="text-3xl font-bold text-gray-900 dark:text-white">{stat.value}</p>
            </div>
          ))}
        </div>

        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
          {/* Trend Chart */}
          <div className="bg-white/50 dark:bg-gray-900/50 backdrop-blur-md border border-gray-200 dark:border-gray-800 rounded-2xl p-6">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-6">
              Monthly Trend
            </h2>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={trendData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" className="dark:stroke-gray-800" />
                <XAxis dataKey="month" stroke="#6B7280" />
                <YAxis stroke="#6B7280" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#FFFFFF',
                    border: '1px solid #E5E7EB',
                    borderRadius: '8px',
                  }}
                  className="dark:bg-gray-900 dark:border-gray-800"
                />
                <Line
                  type="monotone"
                  dataKey="processed"
                  stroke="#3B82F6"
                  strokeWidth={2}
                  dot={{ fill: '#3B82F6' }}
                />
                <Line
                  type="monotone"
                  dataKey="accuracy"
                  stroke="#10B981"
                  strokeWidth={2}
                  dot={{ fill: '#10B981' }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Daily Scans */}
          <div className="bg-white/50 dark:bg-gray-900/50 backdrop-blur-md border border-gray-200 dark:border-gray-800 rounded-2xl p-6">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-6">
              Weekly Activity
            </h2>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={dailyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" className="dark:stroke-gray-800" />
                <XAxis dataKey="day" stroke="#6B7280" />
                <YAxis stroke="#6B7280" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#FFFFFF',
                    border: '1px solid #E5E7EB',
                    borderRadius: '8px',
                  }}
                  className="dark:bg-gray-900 dark:border-gray-800"
                />
                <Area
                  type="monotone"
                  dataKey="scans"
                  stroke="#3B82F6"
                  fill="#3B82F6"
                  fillOpacity={0.2}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Real-time Stats */}
        <div className="bg-white/50 dark:bg-gray-900/50 backdrop-blur-md border border-gray-200 dark:border-gray-800 rounded-2xl p-8">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            Real-time Metrics
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <div className="bg-gray-50/50 dark:bg-gray-800/50 backdrop-blur-md rounded-xl p-4">
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Total Scans</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">1,247</p>
            </div>
            <div className="bg-gray-50/50 dark:bg-gray-800/50 backdrop-blur-md rounded-xl p-4">
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Avg Latency</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">3.2s</p>
            </div>
            <div className="bg-gray-50/50 dark:bg-gray-800/50 backdrop-blur-md rounded-xl p-4">
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Accuracy</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">68.5%</p>
            </div>
            <div className="bg-gray-50/50 dark:bg-gray-800/50 backdrop-blur-md rounded-xl p-4">
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Avg File Size</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">25MB</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Analytics;