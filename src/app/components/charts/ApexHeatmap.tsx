import React from 'react';
import Chart from 'react-apexcharts';
import { ApexOptions } from 'apexcharts';

interface ApexHeatmapProps {
  data: Array<{
    name: string;
    data: Array<{
      x: string;
      y: number;
    }>;
  }>;
  title?: string;
  height?: number;
  colorPalette?: 'monokai' | 'viridis' | 'plasma' | 'inferno';
}

const ApexHeatmap: React.FC<ApexHeatmapProps> = ({ 
  data, 
  title = "Analysis Heatmap",
  height = 350,
  colorPalette = 'viridis'
}) => {
  const colorPalettes = {
    monokai: ['#272822', '#49483E', '#75715E', '#A6E22E', '#F92672'],
    viridis: ['#440154', '#31688E', '#35B779', '#FDE725'],
    plasma: ['#0D0887', '#7E03A8', '#CC4678', '#F89441', '#F0F921'],
    inferno: ['#000004', '#420A68', '#932667', '#DD513A', '#FCA50A', '#FCFFA4']
  };

  // Transform data to ensure proper format
  const transformedData = data.map(series => ({
    name: series.name,
    data: series.data.map(point => ({
      x: point.x,
      y: Math.round(point.y * 100) // Convert to percentage and round
    }))
  }));

  const options: ApexOptions = {
    chart: {
      type: 'heatmap',
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
          reset: true
        }
      },
      animations: {
        enabled: true,
        speed: 800
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
      heatmap: {
        shadeIntensity: 0.5,
        radius: 6,
        useFillColorAsStroke: false,
        colorScale: {
          ranges: [
            { from: 0, to: 20, name: 'Critical', color: '#DC2626' },
            { from: 21, to: 35, name: 'Low', color: '#EA580C' },
            { from: 36, to: 50, name: 'Fair', color: '#F59E0B' },
            { from: 51, to: 65, name: 'Good', color: '#84CC16' },
            { from: 66, to: 80, name: 'Very Good', color: '#22C55E' },
            { from: 81, to: 90, name: 'Excellent', color: '#10B981' },
            { from: 91, to: 100, name: 'Outstanding', color: '#059669' }
          ]
        }
      }
    },
    dataLabels: {
      enabled: true,
      style: {
        colors: ['#ffffff'],
        fontSize: '10px',
        fontWeight: 'bold'
      },
      formatter: function(val: any) {
        return val + '%';
      }
    },
    stroke: {
      width: 2,
      colors: ['#ffffff']
    },
    tooltip: {
      theme: 'dark',
      style: {
        fontSize: '12px'
      },
      y: {
        formatter: function(val: any) {
          let status = '';
          if (val >= 91) status = '[OUTSTANDING] Outstanding';
          else if (val >= 81) status = '[EXCELLENT] Excellent';
          else if (val >= 66) status = '[OK] Very Good';
          else if (val >= 51) status = '[GOOD] Good';
          else if (val >= 36) status = '[WARNING] Fair';
          else if (val >= 21) status = '[LOW] Low';
          else status = '[CRITICAL] Critical';
          
          return val + '% - ' + status;
        }
      }
    },
    legend: {
      show: true,
      position: 'bottom',
      horizontalAlign: 'center',
      fontSize: '12px',
      fontWeight: 'bold',
      markers: {
        size: 16,
        shape: 'square'
      },
      itemMargin: {
        horizontal: 8,
        vertical: 4
      }
    },
    grid: {
      padding: {
        right: 20,
        left: 20,
        top: 20,
        bottom: 40
      }
    },
    xaxis: {
      type: 'category',
      labels: {
        style: {
          colors: '#6B7280',
          fontSize: '11px',
          fontWeight: 'bold'
        }
      }
    },
    yaxis: {
      labels: {
        style: {
          colors: '#6B7280',
          fontSize: '11px',
          fontWeight: 'bold'
        }
      }
    }
  };

  return (
    <div className="bg-white/50 dark:bg-gray-900/50 backdrop-blur-md border border-gray-200 dark:border-gray-800 rounded-2xl p-6">
      <Chart
        options={options}
        series={transformedData}
        type="heatmap"
        height={height}
      />
    </div>
  );
};

export default ApexHeatmap;