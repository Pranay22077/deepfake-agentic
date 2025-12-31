"use client";

import { TrendingUp } from "lucide-react";
import { Bar, BarChart, XAxis, YAxis, Cell } from "recharts";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "./ui/card";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "./ui/chart";

export const description = "A mixed bar chart showing metadata features analysis";

const chartData = [
  { feature: "compression", confidence: 275, fill: "hsl(var(--chart-1))" },
  { feature: "lighting", confidence: 200, fill: "hsl(var(--chart-2))" },
  { feature: "temporal", confidence: 187, fill: "hsl(var(--chart-3))" },
  { feature: "artifacts", confidence: 173, fill: "hsl(var(--chart-4))" },
  { feature: "quality", confidence: 90, fill: "hsl(var(--chart-5))" },
];

const chartConfig = {
  confidence: {
    label: "Confidence Score",
  },
  compression: {
    label: "Compression Analysis",
    color: "hsl(var(--chart-1))",
  },
  lighting: {
    label: "Lighting Consistency",
    color: "hsl(var(--chart-2))",
  },
  temporal: {
    label: "Temporal Coherence",
    color: "hsl(var(--chart-3))",
  },
  artifacts: {
    label: "Artifact Detection",
    color: "hsl(var(--chart-4))",
  },
  quality: {
    label: "Quality Assessment",
    color: "hsl(var(--chart-5))",
  },
  // Weekly trends
  monday: {
    label: "Monday",
    color: "hsl(var(--chart-1))",
  },
  tuesday: {
    label: "Tuesday", 
    color: "hsl(var(--chart-2))",
  },
  wednesday: {
    label: "Wednesday",
    color: "hsl(var(--chart-3))",
  },
  thursday: {
    label: "Thursday",
    color: "hsl(var(--chart-4))",
  },
  friday: {
    label: "Friday",
    color: "hsl(var(--chart-5))",
  },
} satisfies ChartConfig;

interface MetadataFeaturesChartProps {
  data?: Array<{
    feature: string;
    confidence: number;
    fill: string;
  }>;
  title?: string;
  description?: string;
}

export function MetadataFeaturesChart({ 
  data = chartData, 
  title = "Metadata Features Analysis",
  description = "Feature confidence scores from deepfake detection"
}: MetadataFeaturesChartProps) {
  const palette = [
    "#60a5fa",
    "#34d399",
    "#a78bfa",
    "#f59e0b",
    "#f97316",
    "#06b6d4",
  ];

  const dataWithFill = data.map((d, idx) => ({
    ...d,
    fill: palette[idx % palette.length],
  }));

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
            data={dataWithFill}
            layout="vertical"
            margin={{
              left: 0,
            }}
          >
            <YAxis
              dataKey="feature"
              type="category"
              tickLine={false}
              tickMargin={10}
              axisLine={false}
              tickFormatter={(value) =>
                chartConfig[value as keyof typeof chartConfig]?.label || value
              }
            />
            <XAxis dataKey="confidence" type="number" hide />
            <ChartTooltip
              cursor={false}
              content={<ChartTooltipContent hideLabel />}
            />
            <Bar dataKey="confidence" layout="vertical" radius={5}>
              {dataWithFill.map((entry, idx) => (
                <Cell key={`${entry.feature}-${idx}`} fill={entry.fill} />
              ))}
            </Bar>
          </BarChart>
        </ChartContainer>
      </CardContent>
      <CardFooter className="flex-col items-start gap-2 text-sm">
        <div className="flex gap-2 leading-none font-medium">
          Analysis confidence trending up <TrendingUp className="h-4 w-4" />
        </div>
        <div className="text-muted-foreground leading-none">
          Showing feature analysis confidence scores
        </div>
      </CardFooter>
    </Card>
  );
}