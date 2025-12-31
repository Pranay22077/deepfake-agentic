import React from 'react';
import { MetadataFeaturesChart } from '../components/MetadataFeaturesChart';

const ChartDemo = () => {
  // Sample data for testing
  const sampleData = [
    { feature: "compression", confidence: 285, fill: "hsl(var(--chart-1))" },
    { feature: "lighting", confidence: 220, fill: "hsl(var(--chart-2))" },
    { feature: "temporal", confidence: 195, fill: "hsl(var(--chart-3))" },
    { feature: "artifacts", confidence: 160, fill: "hsl(var(--chart-4))" },
    { feature: "quality", confidence: 110, fill: "hsl(var(--chart-5))" },
  ];

  return (
    <div className="min-h-screen pt-32 pb-20">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="mb-12">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
            Chart Demo
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            Testing the MetadataFeaturesChart component with sample data.
          </p>
        </div>

        <div className="space-y-8">
          {/* Default Chart */}
          <div>
            <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
              Default Chart
            </h2>
            <MetadataFeaturesChart />
          </div>

          {/* Custom Chart */}
          <div>
            <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
              Custom Chart with Sample Data
            </h2>
            <MetadataFeaturesChart 
              title="Custom Feature Analysis"
              description="Sample confidence scores for testing"
              data={sampleData}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChartDemo;