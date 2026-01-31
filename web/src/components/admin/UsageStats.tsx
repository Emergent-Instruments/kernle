'use client';

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { HealthStats } from '@/lib/api';

interface UsageStatsProps {
  healthStats: HealthStats;
}

export function UsageStats({ healthStats }: UsageStatsProps) {
  const confidenceBuckets = healthStats.confidence_distribution;
  const values = Object.values(confidenceBuckets);
  const total = values.reduce((a, b) => a + b, 0);
  const max = Math.max(...values, 1);

  // Calculate weighted average confidence
  const weights = [0.1, 0.3, 0.5, 0.7, 0.9];
  const meanConfidence = total > 0
    ? values.reduce((sum, count, i) => sum + count * weights[i], 0) / total
    : 0.5;

  // Calculate points for the area chart
  const chartHeight = 80;
  const chartWidth = 100;
  const points = values.map((count, i) => ({
    x: (i / (values.length - 1)) * chartWidth,
    y: chartHeight - (count / max) * (chartHeight - 10),
  }));

  // Create smooth bezier path
  const linePath = createSmoothPath(points);
  const areaPath = `${linePath} L ${chartWidth} ${chartHeight} L 0 ${chartHeight} Z`;

  // Bucket labels for display
  const bucketLabels = ['Very Low', 'Low', 'Medium', 'High', 'Very High'];

  return (
    <Card>
      <CardHeader>
        <div className="flex justify-between items-start">
          <div>
            <CardTitle>Confidence Spectrum</CardTitle>
            <CardDescription>
              Distribution of confidence scores across memories
            </CardDescription>
          </div>
          <div className="text-right">
            <div className="text-2xl font-bold">{(meanConfidence * 100).toFixed(0)}%</div>
            <div className="text-xs text-muted-foreground">avg confidence</div>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {total === 0 ? (
          <p className="text-sm text-muted-foreground">No confidence data available</p>
        ) : (
          <div className="space-y-4">
            {/* Area Chart */}
            <div className="relative h-32">
              <svg
                viewBox={`0 0 ${chartWidth} ${chartHeight}`}
                preserveAspectRatio="none"
                className="w-full h-full"
              >
                {/* Gradient definition */}
                <defs>
                  <linearGradient id="confidenceGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#ef4444" stopOpacity="0.8" />
                    <stop offset="25%" stopColor="#f97316" stopOpacity="0.8" />
                    <stop offset="50%" stopColor="#eab308" stopOpacity="0.8" />
                    <stop offset="75%" stopColor="#84cc16" stopOpacity="0.8" />
                    <stop offset="100%" stopColor="#22c55e" stopOpacity="0.8" />
                  </linearGradient>
                  <linearGradient id="confidenceGradientFill" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#ef4444" stopOpacity="0.3" />
                    <stop offset="25%" stopColor="#f97316" stopOpacity="0.3" />
                    <stop offset="50%" stopColor="#eab308" stopOpacity="0.3" />
                    <stop offset="75%" stopColor="#84cc16" stopOpacity="0.3" />
                    <stop offset="100%" stopColor="#22c55e" stopOpacity="0.3" />
                  </linearGradient>
                </defs>

                {/* Grid lines */}
                {[0.25, 0.5, 0.75].map((ratio) => (
                  <line
                    key={ratio}
                    x1={0}
                    y1={chartHeight * ratio}
                    x2={chartWidth}
                    y2={chartHeight * ratio}
                    stroke="currentColor"
                    strokeOpacity={0.1}
                    strokeDasharray="2,2"
                  />
                ))}

                {/* Filled area */}
                <path
                  d={areaPath}
                  fill="url(#confidenceGradientFill)"
                  className="transition-all duration-500"
                />

                {/* Line on top */}
                <path
                  d={linePath}
                  fill="none"
                  stroke="url(#confidenceGradient)"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="transition-all duration-500"
                />

                {/* Data points */}
                {points.map((point, i) => (
                  <circle
                    key={i}
                    cx={point.x}
                    cy={point.y}
                    r="3"
                    className="fill-background stroke-2"
                    style={{
                      stroke: getPointColor(i / (points.length - 1)),
                    }}
                  />
                ))}
              </svg>

              {/* X-axis labels */}
              <div className="absolute bottom-0 left-0 right-0 flex justify-between text-xs text-muted-foreground translate-y-5">
                <span>Low</span>
                <span>Medium</span>
                <span>High</span>
              </div>
            </div>

            {/* Detailed breakdown */}
            <div className="grid grid-cols-5 gap-1 pt-4 border-t">
              {Object.entries(confidenceBuckets).map(([bucket, count], i) => {
                const percent = total > 0 ? (count / total) * 100 : 0;
                return (
                  <div key={bucket} className="text-center">
                    <div
                      className="text-lg font-semibold"
                      style={{ color: getPointColor(i / 4) }}
                    >
                      {count.toLocaleString()}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {bucketLabels[i]}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      ({percent.toFixed(1)}%)
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function createSmoothPath(points: { x: number; y: number }[]): string {
  if (points.length < 2) return '';

  // Start at first point
  let path = `M ${points[0].x} ${points[0].y}`;

  // Use bezier curves for smooth transitions
  for (let i = 1; i < points.length; i++) {
    const prev = points[i - 1];
    const curr = points[i];

    // Control points for smooth curve
    const cpX1 = prev.x + (curr.x - prev.x) / 2;
    const cpX2 = prev.x + (curr.x - prev.x) / 2;

    path += ` C ${cpX1} ${prev.y}, ${cpX2} ${curr.y}, ${curr.x} ${curr.y}`;
  }

  return path;
}

function getPointColor(ratio: number): string {
  // Gradient from red to green
  if (ratio < 0.25) return '#ef4444';
  if (ratio < 0.5) return '#f97316';
  if (ratio < 0.75) return '#eab308';
  if (ratio < 0.875) return '#84cc16';
  return '#22c55e';
}
