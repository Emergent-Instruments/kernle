'use client';

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { HealthStats } from '@/lib/api';
import { Brain, Shield, CloudOff, Activity } from 'lucide-react';

interface MemoryDistributionProps {
  healthStats: HealthStats;
}

export function MemoryDistribution({ healthStats }: MemoryDistributionProps) {
  const maxCount = Math.max(...Object.values(healthStats.memory_distribution), 1);

  // Calculate percentages for the summary
  const activePercent = healthStats.total_memories > 0
    ? Math.round((healthStats.active_memories / healthStats.total_memories) * 100)
    : 100;
  const forgottenPercent = healthStats.total_memories > 0
    ? Math.round((healthStats.forgotten_memories / healthStats.total_memories) * 100)
    : 0;
  const protectedPercent = healthStats.total_memories > 0
    ? Math.round((healthStats.protected_memories / healthStats.total_memories) * 100)
    : 0;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Memory Distribution</CardTitle>
        <CardDescription>Breakdown of memories by type and status</CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Summary Stats */}
        <div className="grid gap-4 md:grid-cols-4">
          <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/50">
            <Brain className="h-5 w-5 text-blue-500" />
            <div>
              <div className="text-2xl font-bold">{healthStats.total_memories.toLocaleString()}</div>
              <div className="text-xs text-muted-foreground">Total Memories</div>
            </div>
          </div>
          <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/50">
            <Activity className="h-5 w-5 text-green-500" />
            <div>
              <div className="text-2xl font-bold">{healthStats.active_memories.toLocaleString()}</div>
              <div className="text-xs text-muted-foreground">Active ({activePercent}%)</div>
            </div>
          </div>
          <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/50">
            <CloudOff className="h-5 w-5 text-gray-500" />
            <div>
              <div className="text-2xl font-bold">{healthStats.forgotten_memories.toLocaleString()}</div>
              <div className="text-xs text-muted-foreground">Forgotten ({forgottenPercent}%)</div>
            </div>
          </div>
          <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/50">
            <Shield className="h-5 w-5 text-yellow-500" />
            <div>
              <div className="text-2xl font-bold">{healthStats.protected_memories.toLocaleString()}</div>
              <div className="text-xs text-muted-foreground">Protected ({protectedPercent}%)</div>
            </div>
          </div>
        </div>

        {/* Per-Type Distribution */}
        <div>
          <h4 className="text-sm font-medium mb-3">By Memory Type</h4>
          <div className="grid gap-3 md:grid-cols-2">
            {Object.entries(healthStats.memory_distribution)
              .sort(([, a], [, b]) => b - a)
              .map(([type, count]) => {
                const percent = (count / maxCount) * 100;
                return (
                  <div key={type} className="space-y-1">
                    <div className="flex justify-between text-sm">
                      <span className="font-medium capitalize">{type.replace(/_/g, ' ')}</span>
                      <span className="text-muted-foreground">{count.toLocaleString()}</span>
                    </div>
                    <Progress value={percent} className="h-2" />
                  </div>
                );
              })}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
