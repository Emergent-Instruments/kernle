'use client';

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { HealthStats } from '@/lib/api';
import { Brain, Shield, CloudOff, Activity } from 'lucide-react';

interface MemoryDistributionProps {
  healthStats: HealthStats;
}

// Color palette for memory types - semantically meaningful
const MEMORY_TYPE_COLORS: Record<string, string> = {
  episodes: '#f97316',      // Orange - narrative events
  beliefs: '#14b8a6',       // Teal - convictions
  values: '#eab308',        // Yellow - core principles
  goals: '#3b82f6',         // Blue - aspirations
  notes: '#6b7280',         // Gray - observations
  drives: '#8b5cf6',        // Purple - motivations
  relationships: '#ec4899', // Pink - connections
  checkpoints: '#06b6d4',   // Cyan - state saves
  raw_captures: '#64748b',  // Slate - unprocessed
  playbooks: '#22c55e',     // Green - procedures
  emotional_memories: '#f43f5e', // Rose - feelings
};

export function MemoryDistribution({ healthStats }: MemoryDistributionProps) {
  const total = Object.values(healthStats.memory_distribution).reduce((a, b) => a + b, 0);
  const sortedTypes = Object.entries(healthStats.memory_distribution)
    .sort(([, a], [, b]) => b - a);

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

  // Calculate donut segments
  const segments = calculateDonutSegments(sortedTypes, total);

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

        {/* Donut Chart + Legend */}
        <div className="flex flex-col lg:flex-row gap-6 items-center">
          {/* Donut Chart */}
          <div className="relative w-48 h-48 flex-shrink-0">
            <svg viewBox="0 0 100 100" className="w-full h-full -rotate-90">
              {segments.map((segment) => (
                <DonutSegment
                  key={segment.type}
                  {...segment}
                />
              ))}
            </svg>
            {/* Center stats */}
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <span className="text-3xl font-bold">{total.toLocaleString()}</span>
              <span className="text-xs text-muted-foreground">total</span>
            </div>
          </div>

          {/* Legend */}
          <div className="flex-1 grid grid-cols-2 sm:grid-cols-3 gap-2">
            {sortedTypes.map(([type, count]) => {
              const percent = total > 0 ? ((count / total) * 100).toFixed(1) : '0';
              const color = MEMORY_TYPE_COLORS[type] || '#6b7280';
              return (
                <div key={type} className="flex items-center gap-2 text-sm">
                  <div
                    className="h-3 w-3 rounded-sm flex-shrink-0"
                    style={{ backgroundColor: color }}
                  />
                  <span className="truncate capitalize">{type.replace(/_/g, ' ')}</span>
                  <span className="text-muted-foreground ml-auto">{percent}%</span>
                </div>
              );
            })}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

interface DonutSegmentData {
  type: string;
  count: number;
  startAngle: number;
  angle: number;
  color: string;
}

function calculateDonutSegments(
  sortedTypes: [string, number][],
  total: number
): DonutSegmentData[] {
  if (total === 0) return [];

  let cumulativeAngle = 0;
  return sortedTypes.map(([type, count]) => {
    const angle = (count / total) * 360;
    const startAngle = cumulativeAngle;
    cumulativeAngle += angle;
    return {
      type,
      count,
      startAngle,
      angle,
      color: MEMORY_TYPE_COLORS[type] || '#6b7280',
    };
  });
}

function DonutSegment({ type, startAngle, angle, color }: DonutSegmentData) {
  const innerRadius = 30;
  const outerRadius = 45;

  // Handle full circle case
  if (angle >= 359.99) {
    return (
      <g>
        <circle
          cx={50}
          cy={50}
          r={(innerRadius + outerRadius) / 2}
          fill="none"
          stroke={color}
          strokeWidth={outerRadius - innerRadius}
          className="transition-all duration-300 hover:opacity-80"
        />
      </g>
    );
  }

  // Convert angles to radians
  const startRad = (startAngle * Math.PI) / 180;
  const endRad = ((startAngle + angle) * Math.PI) / 180;

  // Calculate arc points
  const x1 = 50 + innerRadius * Math.cos(startRad);
  const y1 = 50 + innerRadius * Math.sin(startRad);
  const x2 = 50 + outerRadius * Math.cos(startRad);
  const y2 = 50 + outerRadius * Math.sin(startRad);
  const x3 = 50 + outerRadius * Math.cos(endRad);
  const y3 = 50 + outerRadius * Math.sin(endRad);
  const x4 = 50 + innerRadius * Math.cos(endRad);
  const y4 = 50 + innerRadius * Math.sin(endRad);

  const largeArc = angle > 180 ? 1 : 0;

  const pathD = `
    M ${x1} ${y1}
    L ${x2} ${y2}
    A ${outerRadius} ${outerRadius} 0 ${largeArc} 1 ${x3} ${y3}
    L ${x4} ${y4}
    A ${innerRadius} ${innerRadius} 0 ${largeArc} 0 ${x1} ${y1}
  `;

  return (
    <path
      d={pathD}
      fill={color}
      className="transition-all duration-300 hover:opacity-80"
    >
      <title>{type}: {((angle / 360) * 100).toFixed(1)}%</title>
    </path>
  );
}
