import React, { useState } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import ApexHeatmap from './ApexHeatmap';
import EChartsPerformance from './EChartsPerformance';
import PlotlyStatistics from './PlotlyStatistics';
import D3CustomViz from './D3CustomViz';

interface AdvancedChartsShowcaseProps {
  analysisData?: {
    frameConfidences?: number[][];
    modelPerformance?: number[][];
    correlationMatrix?: number[][];
    timeSeriesData?: Array<{ timestamp: number; confidence: number }>;
  };
}

const AdvancedChartsShowcase: React.FC<AdvancedChartsShowcaseProps> = ({ analysisData }) => {
  const [activeTab, setActiveTab] = useState('apex');

  // Generate sample heatmap data for ApexCharts
  const generateApexHeatmapData = () => {
    const models = ['BG Model', 'AV Model', 'CM Model', 'RR Model', 'LL Model', 'TM Model'];
    return models.map(model => ({
      name: model,
      data: Array.from({ length: 20 }, (_, i) => ({
        x: `Frame ${i + 1}`,
        y: Math.random() * 0.6 + 0.2 // Random confidence between 0.2 and 0.8
      }))
    }));
  };

  // Generate sample data for ECharts
  const generateEChartsData = () => ({
    boxplotData: [
      [0.82, 0.85, 0.87, 0.89, 0.92], // BG Model performance distribution
      [0.78, 0.81, 0.84, 0.87, 0.90], // AV Model
      [0.85, 0.88, 0.90, 0.92, 0.95], // CM Model
      [0.80, 0.83, 0.86, 0.88, 0.91], // RR Model
      [0.75, 0.78, 0.81, 0.84, 0.87], // LL Model
      [0.88, 0.90, 0.92, 0.94, 0.97], // TM Model
    ],
    heatmapData: Array.from({ length: 200 }, (_, i) => {
      const frame = Math.floor(i / 20);
      const timeSlice = i % 20;
      return [frame, timeSlice, Math.random() * 100];
    }),
    scatterData: Array.from({ length: 500 }, () => [
      Math.random() * 100, // Quality score
      Math.random() * 100, // Confidence
      Math.random() * 200 + 50 // Processing time (bubble size)
    ])
  });

  // Generate sample data for Plotly
  const generatePlotlyData = () => ({
    heatmapZ: Array.from({ length: 6 }, () =>
      Array.from({ length: 6 }, () => Math.random() * 2 - 1) // Correlation values between -1 and 1
    ),
    heatmapX: ['Compression', 'Lighting', 'Temporal', 'Artifacts', 'Quality', 'Audio'],
    heatmapY: ['BG', 'AV', 'CM', 'RR', 'LL', 'TM'],
    regressionX: Array.from({ length: 100 }, (_, i) => i / 10),
    regressionY: Array.from({ length: 100 }, (_, i) => {
      const x = i / 10;
      return 0.6 * x + 2 + (Math.random() - 0.5) * 2; // Linear relationship with noise
    })
  });

  // Generate sample data for D3
  const generateD3Data = () => ({
    nodes: [
      { id: "Video Input", group: 1, confidence: 0.95 },
      { id: "Frame Sampler", group: 1, confidence: 0.90 },
      { id: "Face Detector", group: 2, confidence: 0.88 },
      { id: "Audio Extractor", group: 2, confidence: 0.85 },
      { id: "BG Model", group: 3, confidence: 0.82 },
      { id: "AV Model", group: 3, confidence: 0.87 },
      { id: "CM Model", group: 3, confidence: 0.91 },
      { id: "RR Model", group: 3, confidence: 0.89 },
      { id: "LL Model", group: 3, confidence: 0.78 },
      { id: "TM Model", group: 3, confidence: 0.93 },
      { id: "LangGraph Router", group: 4, confidence: 0.86 },
      { id: "Aggregator", group: 4, confidence: 0.84 },
      { id: "Explainer", group: 5, confidence: 0.88 },
      { id: "Final Result", group: 5, confidence: 0.85 }
    ],
    links: [
      { source: "Video Input", target: "Frame Sampler", value: 1.0 },
      { source: "Video Input", target: "Audio Extractor", value: 0.8 },
      { source: "Frame Sampler", target: "Face Detector", value: 0.9 },
      { source: "Face Detector", target: "BG Model", value: 0.7 },
      { source: "Face Detector", target: "AV Model", value: 0.8 },
      { source: "Face Detector", target: "CM Model", value: 0.9 },
      { source: "Face Detector", target: "RR Model", value: 0.85 },
      { source: "Face Detector", target: "LL Model", value: 0.6 },
      { source: "Face Detector", target: "TM Model", value: 0.95 },
      { source: "Audio Extractor", target: "AV Model", value: 0.9 },
      { source: "BG Model", target: "LangGraph Router", value: 0.82 },
      { source: "AV Model", target: "LangGraph Router", value: 0.87 },
      { source: "CM Model", target: "LangGraph Router", value: 0.91 },
      { source: "RR Model", target: "LangGraph Router", value: 0.89 },
      { source: "LL Model", target: "LangGraph Router", value: 0.78 },
      { source: "TM Model", target: "LangGraph Router", value: 0.93 },
      { source: "LangGraph Router", target: "Aggregator", value: 0.86 },
      { source: "Aggregator", target: "Explainer", value: 0.84 },
      { source: "Explainer", target: "Final Result", value: 0.88 }
    ],
    circularData: [
      { feature: "Compression", value: 0.85, angle: 0 },
      { feature: "Lighting", value: 0.72, angle: 60 },
      { feature: "Temporal", value: 0.91, angle: 120 },
      { feature: "Artifacts", value: 0.68, angle: 180 },
      { feature: "Quality", value: 0.79, angle: 240 },
      { feature: "Audio Sync", value: 0.83, angle: 300 }
    ]
  });

  return (
    <div className="space-y-8">
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
          Advanced Analysis Visualizations
        </h2>
        <p className="text-gray-600 dark:text-gray-400 max-w-3xl mx-auto">
          Comprehensive deepfake detection analysis using ApexCharts, ECharts, Plotly.js, and D3.js 
          for interactive heatmaps, statistical analysis, and custom visualizations.
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-4 mb-8">
          <TabsTrigger value="apex" className="text-sm">
            ApexCharts
            <span className="ml-2 text-xs bg-blue-100 dark:bg-blue-900 px-2 py-1 rounded">
              Heatmaps
            </span>
          </TabsTrigger>
          <TabsTrigger value="echarts" className="text-sm">
            ECharts
            <span className="ml-2 text-xs bg-green-100 dark:bg-green-900 px-2 py-1 rounded">
              Performance
            </span>
          </TabsTrigger>
          <TabsTrigger value="plotly" className="text-sm">
            Plotly.js
            <span className="ml-2 text-xs bg-purple-100 dark:bg-purple-900 px-2 py-1 rounded">
              Statistics
            </span>
          </TabsTrigger>
          <TabsTrigger value="d3" className="text-sm">
            D3.js
            <span className="ml-2 text-xs bg-orange-100 dark:bg-orange-900 px-2 py-1 rounded">
              Custom
            </span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="apex" className="space-y-8">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <ApexHeatmap
              data={generateApexHeatmapData()}
              title="Frame-by-Frame Confidence Heatmap"
              colorPalette="monokai"
              height={400}
            />
            <ApexHeatmap
              data={generateApexHeatmapData()}
              title="Model Performance Heatmap"
              colorPalette="viridis"
              height={400}
            />
          </div>
          <ApexHeatmap
            data={generateApexHeatmapData()}
            title="Temporal Analysis Heatmap (Full Width)"
            colorPalette="plasma"
            height={300}
          />
          <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-xl p-6">
            <h3 className="text-lg font-semibold text-blue-900 dark:text-blue-100 mb-2">
              ApexCharts Features
            </h3>
            <ul className="text-sm text-blue-800 dark:text-blue-200 space-y-1">
              <li>• Interactive brush charts for zooming and selection</li>
              <li>• Real-time data updates with smooth animations</li>
              <li>• Multiple color palettes (Monokai, Viridis, Plasma, Inferno)</li>
              <li>• Built-in toolbar with download, zoom, and pan controls</li>
              <li>• Responsive design that adapts to container size</li>
            </ul>
          </div>
        </TabsContent>

        <TabsContent value="echarts" className="space-y-8">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <EChartsPerformance
              data={generateEChartsData()}
              type="boxplot"
              title="Model Performance Distribution"
              height={400}
            />
            <EChartsPerformance
              data={generateEChartsData()}
              type="scatter3d"
              title="Quality vs Confidence Correlation"
              height={400}
            />
          </div>
          <EChartsPerformance
            data={generateEChartsData()}
            type="heatmap"
            title="Temporal Confidence Analysis"
            height={350}
          />
          <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-xl p-6">
            <h3 className="text-lg font-semibold text-green-900 dark:text-green-100 mb-2">
              ECharts Features
            </h3>
            <ul className="text-sm text-green-800 dark:text-green-200 space-y-1">
              <li>• Hardware-accelerated Canvas rendering for 50,000+ data points</li>
              <li>• Statistical visualizations: boxplots, scatter plots, heatmaps</li>
              <li>• Built-in brush selection and data zoom functionality</li>
              <li>• Optimized for real-time data streaming</li>
              <li>• Professional-grade performance monitoring</li>
            </ul>
          </div>
        </TabsContent>

        <TabsContent value="plotly" className="space-y-8">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <PlotlyStatistics
              data={generatePlotlyData()}
              type="correlation"
              title="Feature Correlation Matrix"
              height={400}
            />
            <PlotlyStatistics
              data={generatePlotlyData()}
              type="regression"
              title="Quality vs Confidence Regression"
              height={400}
            />
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <PlotlyStatistics
              data={generatePlotlyData()}
              type="heatmap"
              title="Model-Feature Interaction Heatmap"
              height={350}
            />
            <PlotlyStatistics
              data={generatePlotlyData()}
              type="surface3d"
              title="3D Confidence Surface"
              height={350}
            />
          </div>
          <div className="bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 rounded-xl p-6">
            <h3 className="text-lg font-semibold text-purple-900 dark:text-purple-100 mb-2">
              Plotly.js Features
            </h3>
            <ul className="text-sm text-purple-800 dark:text-purple-200 space-y-1">
              <li>• Scientific-grade statistical visualizations</li>
              <li>• Interactive 3D surface plots and regression analysis</li>
              <li>• Built-in modebar with PNG export and zoom controls</li>
              <li>• Z-axis scaling and marginal histograms</li>
              <li>• Industry standard for data science applications</li>
            </ul>
          </div>
        </TabsContent>

        <TabsContent value="d3" className="space-y-8">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <D3CustomViz
              data={generateD3Data()}
              type="force-directed"
              title="Agentic Pipeline Flow"
              width={600}
              height={400}
            />
            <D3CustomViz
              data={generateD3Data()}
              type="circular-heatmap"
              title="Feature Analysis Radar"
              width={600}
              height={400}
            />
          </div>
          <div className="flex justify-center">
            <D3CustomViz
              data={generateD3Data()}
              type="confidence-gauge"
              title="Overall Confidence Gauge"
              width={400}
              height={300}
            />
          </div>
          <div className="bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-800 rounded-xl p-6">
            <h3 className="text-lg font-semibold text-orange-900 dark:text-orange-100 mb-2">
              D3.js Features
            </h3>
            <ul className="text-sm text-orange-800 dark:text-orange-200 space-y-1">
              <li>• Complete DOM control for unique, bespoke visualizations</li>
              <li>• Force-directed graphs showing model relationships</li>
              <li>• Custom circular heatmaps and confidence gauges</li>
              <li>• Smooth animations and interactive drag-and-drop</li>
              <li>• Unlimited customization possibilities</li>
            </ul>
          </div>
        </TabsContent>
      </Tabs>

      <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 border border-blue-200 dark:border-blue-800 rounded-2xl p-8">
        <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
          Integration with Deepfake Analysis Pipeline
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm">
          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">Real-time Analysis</h4>
            <p className="text-gray-600 dark:text-gray-400">
              These visualizations update in real-time as your video is processed through the 
              6 specialist models (BG, AV, CM, RR, LL, TM), showing confidence scores, 
              feature correlations, and model performance metrics.
            </p>
          </div>
          <div>
            <h4 className="font-semibold text-gray-900 dark:text-white mb-2">Interactive Exploration</h4>
            <p className="text-gray-600 dark:text-gray-400">
              Users can interact with heatmaps to explore frame-by-frame analysis, 
              zoom into specific time ranges, and understand which features contributed 
              most to the final fake/real classification.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AdvancedChartsShowcase;