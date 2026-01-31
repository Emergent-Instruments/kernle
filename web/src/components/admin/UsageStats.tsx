'use client';

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { HealthStats } from '@/lib/api';

interface UsageStatsProps {
  healthStats: HealthStats;
}

export function UsageStats({ healthStats }: UsageStatsProps) {
  const confidenceBuckets = healthStats.confidence_distribution;
  const totalConfidenceEntries = Object.values(confidenceBuckets).reduce((a, b) => a + b, 0);
  const maxConfidenceCount = Math.max(...Object.values(confidenceBuckets), 1);

  // Color mapping for confidence buckets
  const bucketColors: Record<string, string> = {
    '0.0-0.2': 'bg-red-500',
    '0.2-0.4': 'bg-orange-500',
    '0.4-0.6': 'bg-yellow-500',
    '0.6-0.8': 'bg-lime-500',
    '0.8-1.0': 'bg-green-500',
  };

  // Bucket labels for display
  const bucketLabels: Record<string, string> = {
    '0.0-0.2': 'Very Low',
    '0.2-0.4': 'Low',
    '0.4-0.6': 'Medium',
    '0.6-0.8': 'High',
    '0.8-1.0': 'Very High',
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Confidence Distribution</CardTitle>
        <CardDescription>
          Distribution of confidence scores across beliefs, episodes, goals, and values
        </CardDescription>
      </CardHeader>
      <CardContent>
        {totalConfidenceEntries === 0 ? (
          <p className="text-sm text-muted-foreground">No confidence data available</p>
        ) : (
          <div className="space-y-3">
            {Object.entries(confidenceBuckets).map(([bucket, count]) => {
              const percent = totalConfidenceEntries > 0 ? (count / totalConfidenceEntries) * 100 : 0;
              const barPercent = (count / maxConfidenceCount) * 100;

              return (
                <div key={bucket} className="space-y-1">
                  <div className="flex justify-between text-sm">
                    <span className="font-medium flex items-center gap-2">
                      <span className={`h-3 w-3 rounded-sm ${bucketColors[bucket]}`} />
                      {bucketLabels[bucket]} ({bucket})
                    </span>
                    <span className="text-muted-foreground">
                      {count.toLocaleString()} ({percent.toFixed(1)}%)
                    </span>
                  </div>
                  <div className="h-2 w-full bg-muted rounded-full overflow-hidden">
                    <div
                      className={`h-full ${bucketColors[bucket]} transition-all duration-300`}
                      style={{ width: `${barPercent}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
