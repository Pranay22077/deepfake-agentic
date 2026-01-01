import React from 'react';
import Plot from 'react-plotly.js';

interface PlotlyStatisticsProps {
  data?: {
    heatmapZ?: number[][];
    heatmapX?: string[];
    heatmapY?: string[];
    regressionX?: number[];
    regressionY?: number[];
    histogramData?: number[];
  };
  type?: 'heatmap' | 'regression' | 'surface3d' | 'correlation';
  title?: string;
  height?: number;
}

const PlotlyStatistics: React.FC<PlotlyStatisticsProps> = ({
  data,
  type = 'heatmap',
  title = "Statistical Analysis",
  height = 400
}) => {
  // Default data for demonstration
  const defaultHeatmapZ = Array.from({ length: 10 }, () =>
    Array.from({ length: 10 }, () => Math.random())
  );
  
  const defaultHeatmapX = Array.from({ length: 10 }, (_, i) => `Feature ${i + 1}`);
  const defaultHeatmapY = Array.from({ length: 10 }, (_, i) => `Model ${i + 1}`);
  
  const defaultRegressionX = Array.from({ length: 100 }, (_, i) => i / 10);
  const defaultRegressionY = defaultRegressionX.map(x => 
    0.5 * x + 0.3 + (Math.random() - 0.5) * 0.2
  );

  const getHeatmapData = () => {
    const z = data?.heatmapZ || defaultHeatmapZ;
    const x = data?.heatmapX || defaultHeatmapX;
    const y = data?.heatmapY || defaultHeatmapY;

    return [
      {
        z: z,
        x: x,
        y: y,
        type: 'heatmap' as const,
        colorscale: [
          [0, '#440154'],
          [0.25, '#31688E'],
          [0.5, '#35B779'],
          [0.75, '#FDE725'],
          [1, '#FDE725']
        ],
        showscale: true,
        hoverongaps: false,
        hovertemplate: 
          '<b>%{y}</b><br>' +
          '%{x}<br>' +
          'Correlation: %{z:.3f}<br>' +
          '<extra></extra>',
        colorbar: {
          title: 'Correlation Coefficient',
          titleside: 'right'
        }
      }
    ];
  };

  const getRegressionData = () => {
    const x = data?.regressionX || defaultRegressionX;
    const y = data?.regressionY || defaultRegressionY;
    
    // Calculate regression line
    const n = x.length;
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
    const sumXX = x.reduce((sum, xi) => sum + xi * xi, 0);
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;
    
    const regressionLine = x.map(xi => slope * xi + intercept);

    return [
      {
        x: x,
        y: y,
        mode: 'markers' as const,
        type: 'scatter' as const,
        name: 'Data Points',
        marker: {
          color: 'rgba(59, 130, 246, 0.7)',
          size: 8,
          line: {
            color: 'rgba(59, 130, 246, 1)',
            width: 1
          }
        },
        hovertemplate: 
          'Quality: %{x:.2f}<br>' +
          'Confidence: %{y:.2f}<br>' +
          '<extra></extra>'
      },
      {
        x: x,
        y: regressionLine,
        mode: 'lines' as const,
        type: 'scatter' as const,
        name: `Regression Line (R² = ${(slope * slope).toFixed(3)})`,
        line: {
          color: 'rgba(239, 68, 68, 1)',
          width: 3
        },
        hovertemplate: 
          'Predicted: %{y:.2f}<br>' +
          '<extra></extra>'
      }
    ];
  };

  const getSurface3DData = () => {
    const size = 20;
    const z = Array.from({ length: size }, (_, i) =>
      Array.from({ length: size }, (_, j) => {
        const x = (i - size/2) / 5;
        const y = (j - size/2) / 5;
        return Math.sin(Math.sqrt(x*x + y*y)) * Math.exp(-0.1 * (x*x + y*y));
      })
    );

    return [
      {
        z: z,
        type: 'surface' as const,
        colorscale: 'Viridis',
        showscale: true,
        hovertemplate: 
          'X: %{x}<br>' +
          'Y: %{y}<br>' +
          'Confidence: %{z:.3f}<br>' +
          '<extra></extra>',
        colorbar: {
          title: 'Confidence Level',
          titleside: 'right'
        }
      }
    ];
  };

  const getCorrelationData = () => {
    // Generate correlation matrix for deepfake features
    const features = ['Compression', 'Lighting', 'Temporal', 'Artifacts', 'Quality', 'Audio'];
    const correlationMatrix = [
      [1.00, 0.23, 0.45, 0.67, 0.34, 0.12],
      [0.23, 1.00, 0.56, 0.78, 0.45, 0.23],
      [0.45, 0.56, 1.00, 0.34, 0.67, 0.45],
      [0.67, 0.78, 0.34, 1.00, 0.56, 0.34],
      [0.34, 0.45, 0.67, 0.56, 1.00, 0.78],
      [0.12, 0.23, 0.45, 0.34, 0.78, 1.00]
    ];

    return [
      {
        z: correlationMatrix,
        x: features,
        y: features,
        type: 'heatmap' as const,
        colorscale: [
          [0, '#313695'],
          [0.25, '#74add1'],
          [0.5, '#ffffcc'],
          [0.75, '#fdae61'],
          [1, '#d73027']
        ],
        showscale: true,
        hoverongaps: false,
        hovertemplate: 
          '<b>%{y} vs %{x}</b><br>' +
          'Correlation: %{z:.3f}<br>' +
          '<extra></extra>',
        colorbar: {
          title: 'Correlation',
          titleside: 'right',
          tick0: -1,
          dtick: 0.5
        }
      }
    ];
  };

  const getData = () => {
    switch (type) {
      case 'heatmap':
        return getHeatmapData();
      case 'regression':
        return getRegressionData();
      case 'surface3d':
        return getSurface3DData();
      case 'correlation':
        return getCorrelationData();
      default:
        return getHeatmapData();
    }
  };

  const getLayout = () => {
    const baseLayout = {
      title: {
        text: title,
        font: { size: 16, color: '#374151' }
      },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      height: height,
      margin: { t: 50, r: 50, b: 50, l: 50 },
      font: { color: '#374151' }
    };

    switch (type) {
      case 'regression':
        return {
          ...baseLayout,
          xaxis: { title: 'Quality Score', gridcolor: '#E5E7EB' },
          yaxis: { title: 'Confidence Level', gridcolor: '#E5E7EB' },
          showlegend: true
        };
      case 'surface3d':
        return {
          ...baseLayout,
          scene: {
            xaxis: { title: 'X Coordinate' },
            yaxis: { title: 'Y Coordinate' },
            zaxis: { title: 'Confidence' },
            camera: {
              eye: { x: 1.5, y: 1.5, z: 1.5 }
            }
          }
        };
      case 'correlation':
        return {
          ...baseLayout,
          xaxis: { title: 'Features', side: 'bottom' },
          yaxis: { title: 'Features' }
        };
      default:
        return {
          ...baseLayout,
          xaxis: { title: 'Features' },
          yaxis: { title: 'Models' }
        };
    }
  };

  const config = {
    displayModeBar: true,
    modeBarButtonsToAdd: ['downloadImage'],
    displaylogo: false,
    toImageButtonOptions: {
      format: 'png' as const,
      filename: `deepfake_analysis_${type}`,
      height: height,
      width: 800,
      scale: 2
    }
  };

  return (
    <div className="bg-white/50 dark:bg-gray-900/50 backdrop-blur-md border border-gray-200 dark:border-gray-800 rounded-2xl p-6">
      <Plot
        data={getData()}
        layout={getLayout()}
        config={config}
        style={{ width: '100%', height: `${height}px` }}
      />
    </div>
  );
};

export default PlotlyStatistics;