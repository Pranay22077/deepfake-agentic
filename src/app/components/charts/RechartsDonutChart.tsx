"use client"

import * as React from "react"
import { TrendingUp } from "lucide-react"
import { Label, Pie, PieChart, Cell } from "recharts"
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "../ui/card"
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "../ui/chart"

interface RechartsDonutChartProps {
  data: Array<{
    label: string;
    value: number;
    color: string;
  }>;
  title: string;
  description?: string;
  totalLabel?: string;
  trendText?: string;
  trendPercentage?: string;
}

export function RechartsDonutChart({ 
  data, 
  title, 
  description = "",
  totalLabel = "Total",
  trendText = "Trending up this month",
  trendPercentage = "+5.2%"
}: RechartsDonutChartProps) {
  
  // Enhanced color palette for light/dark mode compatibility
  const enhancedData = data.map((item, index) => {
    const colors = [
      '#3B82F6', // Blue
      '#10B981', // Green
      '#8B5CF6', // Purple
      '#F59E0B', // Orange
      '#EF4444', // Red
      '#06B6D4', // Cyan
      '#F97316', // Orange-red
      '#84CC16', // Lime
    ];
    return {
      category: item.label.toLowerCase().replace(/\s+/g, '-'),
      value: item.value,
      label: item.label,
      fill: colors[index % colors.length]
    };
  });

  // Create chart config dynamically
  const chartConfig = React.useMemo(() => {
    const config: ChartConfig = {
      value: {
        label: "Value",
      },
    };
    
    enhancedData.forEach((item) => {
      config[item.category] = {
        label: item.label,
        color: item.fill,
      };
    });
    
    return config;
  }, [enhancedData]);

  const totalValue = React.useMemo(() => {
    return enhancedData.reduce((acc, curr) => acc + curr.value, 0)
  }, [enhancedData])

  return (
    <Card className="flex flex-col bg-white/80 dark:bg-gray-900/80 backdrop-blur-md border border-gray-200 dark:border-gray-700 shadow-lg">
      <CardHeader className="items-center pb-0">
        <CardTitle className="text-lg font-semibold text-gray-900 dark:text-white">
          {title}
        </CardTitle>
        {description && (
          <CardDescription className="text-sm text-gray-600 dark:text-gray-400">
            {description}
          </CardDescription>
        )}
      </CardHeader>
      <CardContent className="flex-1 pb-0">
        <ChartContainer
          config={chartConfig}
          className="mx-auto aspect-square max-h-[250px] w-full"
        >
          <PieChart>
            <ChartTooltip
              cursor={false}
              content={<ChartTooltipContent hideLabel />}
            />
            <Pie
              data={enhancedData}
              dataKey="value"
              nameKey="category"
              innerRadius={60}
              outerRadius={100}
              strokeWidth={3}
              stroke="#ffffff"
            >
              {enhancedData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.fill} />
              ))}
              <Label
                content={({ viewBox }) => {
                  if (viewBox && "cx" in viewBox && "cy" in viewBox) {
                    return (
                      <text
                        x={viewBox.cx}
                        y={viewBox.cy}
                        textAnchor="middle"
                        dominantBaseline="middle"
                        className="font-inter"
                      >
                        <tspan
                          x={viewBox.cx}
                          y={viewBox.cy}
                          className="fill-gray-900 dark:fill-white text-2xl font-bold"
                        >
                          {totalValue.toLocaleString()}
                        </tspan>
                        <tspan
                          x={viewBox.cx}
                          y={(viewBox.cy || 0) + 20}
                          className="fill-gray-600 dark:fill-gray-400 text-xs font-medium"
                        >
                          {totalLabel}
                        </tspan>
                      </text>
                    )
                  }
                }}
              />
            </Pie>
          </PieChart>
        </ChartContainer>
      </CardContent>
      <CardFooter className="flex-col gap-2 text-sm pt-4">
        <div className="flex items-center gap-2 leading-none font-medium text-gray-900 dark:text-white">
          {trendText} {trendPercentage} <TrendingUp className="h-4 w-4 text-green-500" />
        </div>
        <div className="text-gray-600 dark:text-gray-400 leading-none text-center">
          Showing analysis data for the current period
        </div>
      </CardFooter>
    </Card>
  )
}