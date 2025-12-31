"use client"

import * as React from "react"
import { TrendingUp } from "lucide-react"
import { Bar, BarChart, CartesianGrid, LabelList, XAxis, YAxis, Cell } from "recharts"
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

interface RechartsBarChartProps {
  data: Array<{
    category: string;
    value: number;
    color: string;
  }>;
  title: string;
  description?: string;
  trendText?: string;
  trendPercentage?: string;
  type?: 'vertical' | 'horizontal';
}

export function RechartsBarChart({ 
  data, 
  title, 
  description = "",
  trendText = "Trending up this month",
  trendPercentage = "+5.2%",
  type = 'vertical'
}: RechartsBarChartProps) {
  // Palette fallback in case color not provided
  const palette = [
    "#60a5fa",
    "#34d399",
    "#a78bfa",
    "#f59e0b",
    "#f97316",
    "#06b6d4",
  ];

  // Transform data for the chart with color preserved/fallback
  const chartData = data.map((item, idx) => ({
    category: item.category,
    value: item.value,
    color: item.color || palette[idx % palette.length],
    displayValue: `${(item.value * 100).toFixed(1)}%`
  }));

  // Create chart config dynamically
  const chartConfig = React.useMemo(() => {
    const config: ChartConfig = {
      value: {
        label: "Accuracy",
        color: "hsl(var(--chart-1))",
      },
    };
    
    return config;
  }, []);

  if (type === 'horizontal') {
    return (
      <Card className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-md border border-gray-200 dark:border-gray-700 shadow-lg">
        <CardHeader>
          <CardTitle className="text-lg font-semibold text-gray-900 dark:text-white">
            {title}
          </CardTitle>
          {description && (
            <CardDescription className="text-sm text-gray-600 dark:text-gray-400">
              {description}
            </CardDescription>
          )}
        </CardHeader>
        <CardContent>
          <ChartContainer config={chartConfig} className="h-[350px] w-full">
            <BarChart
              accessibilityLayer
              data={chartData}
              layout="horizontal"
              margin={{
                left: 100,
                right: 50,
                top: 20,
                bottom: 20,
              }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--muted-foreground))" opacity={0.3} />
              <XAxis 
                type="number" 
                domain={[0, 1]}
                tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                stroke="hsl(var(--muted-foreground))"
                fontSize={12}
                tickLine={false}
                axisLine={false}
              />
              <YAxis 
                dataKey="category" 
                type="category" 
                tickLine={false}
                tickMargin={10}
                axisLine={false}
                stroke="hsl(var(--muted-foreground))"
                fontSize={12}
                width={90}
              />
              <ChartTooltip
                cursor={{ fill: 'hsl(var(--muted))', opacity: 0.3 }}
                content={<ChartTooltipContent 
                  formatter={(value: any) => [`${(value * 100).toFixed(1)}%`, "Accuracy"]}
                  hideLabel 
                />}
              />
              <Bar 
                dataKey="value" 
                radius={[0, 4, 4, 0]}
              >
                {chartData.map((entry, index) => (
                  <Cell key={index} fill={entry.color ?? "hsl(var(--chart-1))"} />
                ))}
                <LabelList
                  dataKey="displayValue"
                  position="right"
                  offset={8}
                  className="fill-foreground"
                  fontSize={12}
                />
              </Bar>
            </BarChart>
          </ChartContainer>
        </CardContent>
        <CardFooter className="flex-col items-start gap-2 text-sm">
          <div className="flex gap-2 font-medium leading-none text-gray-900 dark:text-white">
            {trendText} {trendPercentage} <TrendingUp className="h-4 w-4 text-green-500" />
          </div>
          <div className="leading-none text-gray-600 dark:text-gray-400">
            Showing performance metrics for all models
          </div>
        </CardFooter>
      </Card>
    )
  }

  return (
    <Card className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-md border border-gray-200 dark:border-gray-700 shadow-lg">
      <CardHeader>
        <CardTitle className="text-lg font-semibold text-gray-900 dark:text-white">
          {title}
        </CardTitle>
        {description && (
          <CardDescription className="text-sm text-gray-600 dark:text-gray-400">
            {description}
          </CardDescription>
        )}
      </CardHeader>
      <CardContent>
        <ChartContainer config={chartConfig} className="h-[350px] w-full">
          <BarChart
            accessibilityLayer
            data={chartData}
            margin={{
              top: 30,
              left: 30,
              right: 30,
              bottom: 100,
            }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--muted-foreground))" opacity={0.3} />
            <XAxis
              dataKey="category"
              tickLine={false}
              tickMargin={10}
              axisLine={false}
              angle={-45}
              textAnchor="end"
              height={90}
              stroke="hsl(var(--muted-foreground))"
              fontSize={12}
            />
            <YAxis 
              domain={[0, 1]}
              tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
              stroke="hsl(var(--muted-foreground))"
              fontSize={12}
              tickLine={false}
              axisLine={false}
            />
            <ChartTooltip
              cursor={{ fill: 'hsl(var(--muted))', opacity: 0.3 }}
              content={<ChartTooltipContent 
                formatter={(value: any) => [`${(value * 100).toFixed(1)}%`, "Accuracy"]}
                hideLabel 
              />}
            />
            <Bar 
              dataKey="value" 
              radius={[4, 4, 0, 0]}
            >
              {chartData.map((entry, index) => (
                <Cell key={index} fill={entry.color ?? "hsl(var(--chart-1))"} />
              ))}
              <LabelList
                dataKey="displayValue"
                position="top"
                offset={8}
                className="fill-foreground"
                fontSize={12}
              />
            </Bar>
          </BarChart>
        </ChartContainer>
      </CardContent>
      <CardFooter className="flex-col items-start gap-2 text-sm">
        <div className="flex gap-2 font-medium leading-none text-gray-900 dark:text-white">
          {trendText} {trendPercentage} <TrendingUp className="h-4 w-4 text-green-500" />
        </div>
        <div className="leading-none text-gray-600 dark:text-gray-400">
          Showing performance metrics for all models
        </div>
      </CardFooter>
    </Card>
  )
}