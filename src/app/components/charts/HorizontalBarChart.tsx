"use client"

import { TrendingUp } from "lucide-react"
import { Bar, BarChart, XAxis, YAxis } from "recharts"
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

interface HorizontalBarChartProps {
  data?: Array<{
    category: string;
    value: number;
  }>;
  title?: string;
  description?: string;
  trendText?: string;
  trendPercentage?: string;
}

export function HorizontalBarChart({ 
  data,
  title = "Bar Chart - Horizontal",
  description = "January - June 2024",
  trendText = "Trending up by 5.2% this month",
  trendPercentage
}: HorizontalBarChartProps) {
  
  // Default data if none provided
  const defaultData = [
    { category: "January", value: 186 },
    { category: "February", value: 305 },
    { category: "March", value: 237 },
    { category: "April", value: 73 },
    { category: "May", value: 209 },
    { category: "June", value: 214 },
  ];

  const chartData = data || defaultData;

  const chartConfig = {
    value: {
      label: "Desktop",
      color: "var(--chart-1)",
    },
  } satisfies ChartConfig;

  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent>
        <ChartContainer config={chartConfig}>
          <BarChart
            accessibilityLayer
            data={chartData}
            layout="vertical"
            margin={{
              left: -20,
            }}
          >
            <XAxis type="number" dataKey="value" hide />
            <YAxis
              dataKey="category"
              type="category"
              tickLine={false}
              tickMargin={10}
              axisLine={false}
              tickFormatter={(value) => value.slice(0, 3)}
            />
            <ChartTooltip
              cursor={false}
              content={<ChartTooltipContent hideLabel />}
            />
            <Bar dataKey="value" fill="var(--color-value)" radius={5} />
          </BarChart>
        </ChartContainer>
      </CardContent>
      <CardFooter className="flex-col items-start gap-2 text-sm">
        <div className="flex gap-2 leading-none font-medium">
          {trendText} <TrendingUp className="h-4 w-4" />
        </div>
        <div className="text-muted-foreground leading-none">
          Showing total visitors for the last 6 months
        </div>
      </CardFooter>
    </Card>
  )
}