import React from 'react';
import Chart from 'react-apexcharts';
import { ApexOptions } from 'apexcharts';

interface MetadataPieChartProps {
  data: Array<{
    label: string;
    value: number;
    color?: string;
  }>;
  title?: string;
  height?: number;
}

const MetadataPieChart: React.FC<MetadataPieChartProps> = ({ 
  data, 
  title = "Metadata Distribution",
  height = 350
}) => {
  const series = data.map(item => item.value);
  const labels = data.map(item => item.label);
  const colors = data.map(item => item.color || '#3B82F6');

  const options: ApexOptions = {
    chart: {
      type: 'pie',
      height: height,
      background: 'transparent',
      animations: {
        enabled: true,
        easing: 'easeinout',
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
    labels: labels,
    colors: colors,
    legend: {
      position: 'bottom',
      horizontalAlign: 'center',
      fontSize: '12px',
      markers: {
        width: 12,
        height: 12,
        radius: 6
      }
    },
    plotOptions: {
      pie: {
        donut: {
          size: '65%',
          labels: {
            show: true,
            total: {
              show: true,
              label: 'Total',
              fontSize: '16px',
              fontWeight: 'bold',
              color: '#374151',
              formatter: function (w) {
                return w.globals.seriesTotals.reduce((a: number, b: number) => a + b, 0).toString();
              }
            },
            value: {
              show: true,
              fontSize: '24px',
              fontWeight: 'bold',
              color: '#374151'
            }
          }
        },
        expandOnClick: true,
        customScale: 1.1
      }
    },
    dataLabels: {
      enabled: true,
      formatter: function (val: number) {
        return val.toFixed(1) + '%';
      },
      style: {
        fontSize: '12px',
        fontWeight: 'bold',
        colors: ['#fff']
      },
      dropShadow: {
        enabled: true,
        top: 1,
        left: 1,
        blur: 1,
        opacity: 0.8
      }
    },
    tooltip: {
      theme: 'dark',
      y: {
        formatter: function (val: number, opts: any) {
          const label = opts.w.globals.labels[opts.seriesIndex];
          return `${label}: ${val} (${opts.w.globals.seriesPercent[opts.seriesIndex][0].toFixed(1)}%)`;
        }
      }
    },
    responsive: [{
      breakpoint: 480,
      options: {
        chart: {
          width: 300
        },
        legend: {
          position: 'bottom'
        }
      }
    }]
  };

  return (
    <div className="bg-white/50 dark:bg-gray-900/50 backdrop-blur-md border border-gray-200 dark:border-gray-800 rounded-2xl p-6">
      <Chart
        options={options}
        series={series}
        type="pie"
        height={height}
      />
    </div>
  );
};

export default MetadataPieChart;