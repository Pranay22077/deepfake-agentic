import React from 'react';
import ReactECharts from 'echarts-for-react';
import * as echarts from 'echarts';

interface EChartsPerformanceProps {
  data?: {
    boxplotData?: number[][];
    heatmapData?: number[][];
    scatterData?: Array<[number, number, number]>;
  };
  type?: 'boxplot' | 'heatmap' | 'scatter3d';
  title?: string;
  height?: number;
}

const EChartsPerformance: React.FC<EChartsPerformanceProps> = ({
  data,
  type = 'boxplot',
  title = "Performance Analysis",
  height = 400
}) => {
  // Default data for demonstration
  const defaultBoxplotData = [
    [0.85, 0.87, 0.89, 0.91, 0.93], // Compression Model
    [0.82, 0.84, 0.86, 0.88, 0.90], // Lighting Model
    [0.88, 0.90, 0.92, 0.94, 0.96], // Temporal Model
    [0.80, 0.83, 0.85, 0.87, 0.89], // Artifact Model
    [0.86, 0.88, 0.90, 0.92, 0.94], // Quality Model
  ];

  const defaultHeatmapData = Array.from({ length: 10 }, (_, i) =>
    Array.from({ length: 20 }, (_, j) => [i, j, Math.random() * 100])
  ).flat();

  const defaultScatterData = Array.from({ length: 1000 }, () => [
    Math.random() * 100,
    Math.random() * 100,
    Math.random() * 50 + 10
  ]);

  const getBoxplotOption = () => ({
    title: {
      text: title,
      left: 'center',
      textStyle: {
        color: '#374151',
        fontSize: 16,
        fontWeight: 'bold'
      }
    },
    tooltip: {
      trigger: 'item',
      axisPointer: {
        type: 'shadow'
      },
      formatter: function(param: any) {
        const data = param.data;
        return `
          <div class="p-2">
            <div class="font-semibold">${param.name}</div>
            <div class="text-sm">Min: ${(data[1] * 100).toFixed(1)}%</div>
            <div class="text-sm">Q1: ${(data[2] * 100).toFixed(1)}%</div>
            <div class="text-sm">Median: ${(data[3] * 100).toFixed(1)}%</div>
            <div class="text-sm">Q3: ${(data[4] * 100).toFixed(1)}%</div>
            <div class="text-sm">Max: ${(data[5] * 100).toFixed(1)}%</div>
          </div>
        `;
      }
    },
    grid: {
      left: '10%',
      right: '10%',
      bottom: '15%'
    },
    xAxis: {
      type: 'category',
      data: ['Compression', 'Lighting', 'Temporal', 'Artifacts', 'Quality'],
      boundaryGap: true,
      nameGap: 30,
      splitArea: {
        show: false
      },
      axisLabel: {
        formatter: '{value}',
        rotate: 45
      },
      splitLine: {
        show: false
      }
    },
    yAxis: {
      type: 'value',
      name: 'Confidence Score',
      min: 0.7,
      max: 1.0,
      axisLabel: {
        formatter: function(value: number) {
          return (value * 100).toFixed(0) + '%';
        }
      },
      splitArea: {
        show: true
      }
    },
    series: [
      {
        name: 'Model Performance',
        type: 'boxplot',
        data: data?.boxplotData || defaultBoxplotData,
        itemStyle: {
          color: '#3B82F6',
          borderColor: '#1E40AF'
        },
        emphasis: {
          itemStyle: {
            color: '#60A5FA',
            borderColor: '#2563EB'
          }
        }
      }
    ]
  });

  const getHeatmapOption = () => ({
    title: {
      text: title,
      left: 'center'
    },
    tooltip: {
      position: 'top',
      formatter: function(params: any) {
        return `Frame ${params.data[0]}<br/>
                Time ${params.data[1]}s<br/>
                Confidence: ${params.data[2].toFixed(1)}%`;
      }
    },
    grid: {
      height: '50%',
      top: '10%'
    },
    xAxis: {
      type: 'category',
      data: Array.from({ length: 20 }, (_, i) => `${i}s`),
      splitArea: {
        show: true
      }
    },
    yAxis: {
      type: 'category',
      data: Array.from({ length: 10 }, (_, i) => `Frame ${i}`),
      splitArea: {
        show: true
      }
    },
    visualMap: {
      min: 0,
      max: 100,
      calculable: true,
      orient: 'horizontal',
      left: 'center',
      bottom: '15%',
      inRange: {
        color: ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffcc', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
      }
    },
    series: [{
      name: 'Confidence',
      type: 'heatmap',
      data: data?.heatmapData || defaultHeatmapData,
      label: {
        show: false
      },
      emphasis: {
        itemStyle: {
          shadowBlur: 10,
          shadowColor: 'rgba(0, 0, 0, 0.5)'
        }
      }
    }]
  });

  const getScatter3DOption = () => ({
    title: {
      text: title,
      left: 'center'
    },
    tooltip: {
      formatter: function(params: any) {
        const data = params.data;
        return `Quality: ${data[0].toFixed(1)}<br/>
                Confidence: ${data[1].toFixed(1)}%<br/>
                Processing Time: ${data[2].toFixed(1)}ms`;
      }
    },
    xAxis: {
      name: 'Quality Score',
      nameLocation: 'middle',
      nameGap: 30
    },
    yAxis: {
      name: 'Confidence %',
      nameLocation: 'middle',
      nameGap: 40
    },
    series: [{
      type: 'scatter',
      data: data?.scatterData || defaultScatterData,
      symbolSize: function(data: number[]) {
        return Math.sqrt(data[2]) / 2;
      },
      itemStyle: {
        color: function(params: any) {
          const confidence = params.data[1];
          if (confidence > 80) return '#EF4444';
          if (confidence > 60) return '#F59E0B';
          if (confidence > 40) return '#10B981';
          return '#3B82F6';
        },
        opacity: 0.7
      },
      emphasis: {
        itemStyle: {
          opacity: 1
        }
      }
    }]
  });

  const getOption = () => {
    switch (type) {
      case 'boxplot':
        return getBoxplotOption();
      case 'heatmap':
        return getHeatmapOption();
      case 'scatter3d':
        return getScatter3DOption();
      default:
        return getBoxplotOption();
    }
  };

  return (
    <div className="bg-white/50 dark:bg-gray-900/50 backdrop-blur-md border border-gray-200 dark:border-gray-800 rounded-2xl p-6">
      <ReactECharts
        option={getOption()}
        style={{ height: `${height}px` }}
        opts={{ renderer: 'canvas' }}
      />
    </div>
  );
};

export default EChartsPerformance;