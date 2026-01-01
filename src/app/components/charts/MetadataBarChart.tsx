import React from 'react';
import Chart from 'react-apexcharts';
import { ApexOptions } from 'apexcharts';

interface MetadataBarChartProps {
  data: Array<{
    category: string;
    value: number;
    color?: string;
  }>;
  title?: string;
  height?: number;
  type?: 'vertical' | 'horizontal';
}

const MetadataBarChart: React.FC<MetadataBarChartProps> = ({ 
  data, 
  title = "Metadata Analysis",
  height = 350,
  type = 'vertical'
}) => {
  const series = [{
    name: 'Confidence Score',
    data: data.map(item => item.value)
  }];

  const categories = data.map(item => item.category);
  const colors = data.map(item => item.color || '#3B82F6');

  const options: ApexOptions = {
    chart: {
      type: 'bar',
      height: height,
      background: 'transparent',
      toolbar: {
        show: true,
        tools: {
          download: true,
          selection: false,
          zoom: false,
          zoomin: false,
          zoomout: false,
          pan: false,
          reset: false
        }
      },
      animations: {
        enabled: true,
        easing: 'easeinout',
        speed: 800,
        animateGradually: {
          enabled: true,
          delay: 150
        },
        dynamicAnimation: {
          enabled: true,
          speed: 350
        }
      }
    },
    title: {
      text: title,
      style: {
        fontSize: '16px',
        fontWeight: 'bold',
        color: '#374151'
      }
    },
    plotOptions: {
      bar: {
        horizontal: type === 'horizontal',
        columnWidth: '70%',
        borderRadius: 12,
        dataLabels: {
          position: 'top'
        },
        distributed: true
      }
    },
    colors: colors,
    dataLabels: {
      enabled: true,
      formatter: function (val: number) {
        return (val * 100).toFixed(1) + '%';
      },
      offsetY: type === 'horizontal' ? 0 : -20,
      offsetX: type === 'horizontal' ? 10 : 0,
      style: {
        fontSize: '12px',
        fontWeight: 'bold',
        colors: ['#374151']
      }
    },
    xaxis: {
      categories: categories,
      labels: {
        style: {
          colors: '#6B7280',
          fontSize: '12px'
        },
        rotate: type === 'vertical' ? -45 : 0
      },
      axisBorder: {
        show: false
      },
      axisTicks: {
        show: false
      }
    },
    yaxis: {
      labels: {
        formatter: function (val: number) {
          return (val * 100).toFixed(0) + '%';
        },
        style: {
          colors: '#6B7280',
          fontSize: '12px'
        }
      },
      min: 0,
      max: 1
    },
    grid: {
      borderColor: '#E5E7EB',
      strokeDashArray: 3,
      xaxis: {
        lines: {
          show: false
        }
      },
      yaxis: {
        lines: {
          show: true
        }
      }
    },
    tooltip: {
      theme: 'dark',
      y: {
        formatter: function (val: number) {
          return `Confidence: ${(val * 100).toFixed(1)}%`;
        }
      }
    },
    legend: {
      show: false
    },
    responsive: [{
      breakpoint: 768,
      options: {
        plotOptions: {
          bar: {
            horizontal: true
          }
        },
        xaxis: {
          labels: {
            rotate: 0
          }
        }
      }
    }]
  };

  return (
    <div className="bg-white/50 dark:bg-gray-900/50 backdrop-blur-md border border-gray-200 dark:border-gray-800 rounded-2xl p-6">
      <Chart
        options={options}
        series={series}
        type="bar"
        height={height}
      />
    </div>
  );
};

export default MetadataBarChart;