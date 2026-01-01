"use client"

import { TrendingUp } from "lucide-react"
import { Bar, BarChart, XAxis, YAxis, Cell } from "recharts"
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

interface AIModelPerformanceChartProps {
  title?: string
  description?: string
  data?: Array<{
    model: string
    accuracy: number
    fill?: string
  }>
}

const defaultData = [
  { model: "TM Model", accuracy: 94.5 },
  { model: "AV Model", accuracy: 93.4 },
  { model: "LL Model", accuracy: 92.3 },
  { model: "CM Model", accuracy: 89.1 },
  { model: "RR Model", accuracy: 87.6 },
  { model: "BG Model", accuracy: 86.2 },
]

const chartConfig = {
  accuracy: {
    label: "Accuracy",
    color: "var(--chart-1)",
  },
} satisfies ChartConfig

export function AIModelPerformanceChart({
  title = "AI Model Performance",
  description = "Accuracy comparison across specialist models",
  data = defaultData,
}: AIModelPerformanceChartProps) {
  const palette = [
    "#60a5fa",
    "#34d399",
    "#a78bfa",
    "#f59e0b",
    "#f97316",
    "#06b6d4",
  ]

  const dataWithFill = data.map((d, idx) => ({
    ...d,
    fill: palette[idx % palette.length],
  }))

  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent>
        <ChartContainer config={chartConfig} className="h-[320px] w-full">
          <BarChart
            accessibilityLayer
            data={dataWithFill}
            layout="vertical"
            margin={{
              left: -20,
            }}
          >
            <XAxis type="number" dataKey="accuracy" hide />
            <YAxis
              dataKey="model"
              type="category"
              tickLine={false}
              tickMargin={10}
              axisLine={false}
            />
            <ChartTooltip
              cursor={false}
              content={
                <ChartTooltipContent
                  hideLabel
                  formatter={(value: any) => [`${Number(value).toFixed(1)}%`, "Accuracy"]}
                />
              }
            />
            <Bar
              dataKey="accuracy"
              radius={8}
            >
              {dataWithFill.map((entry) => (
                <Cell key={entry.model} fill={entry.fill} />
              ))}
            </Bar>
          </BarChart>
        </ChartContainer>
      </CardContent>
      <CardFooter className="flex-col items-start gap-2 text-sm">
        <div className="flex gap-2 leading-none font-medium">
          Trending up by 5.2% this month <TrendingUp className="h-4 w-4" />
        </div>
        <div className="text-muted-foreground leading-none">
          Showing performance metrics for the specialist models
        </div>
      </CardFooter>
    </Card>
  )
}