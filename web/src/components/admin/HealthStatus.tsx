'use client';

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { HealthStats } from '@/lib/api';
import { Database, Server, RefreshCw } from 'lucide-react';
import { cn } from '@/lib/utils';

interface HealthStatusProps {
  healthStats: HealthStats;
  isConnected: boolean;
}

export function HealthStatus({ healthStats, isConnected }: HealthStatusProps) {
  const isHealthy = healthStats.database_status === 'connected' &&
                    healthStats.api_status === 'healthy';

  return (
    <Card className="relative overflow-hidden">
      {/* Animated background pulse when healthy */}
      {isHealthy && isConnected && (
        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-primary/5 to-transparent animate-pulse" />
      )}

      <CardHeader className="pb-2">
        <div className="flex justify-between items-center">
          <div className="flex items-center gap-3">
            {/* Central pulse indicator */}
            <div className="relative">
              <div className={cn(
                "h-3 w-3 rounded-full transition-colors",
                isConnected ? "bg-green-500" : "bg-red-500"
              )} />
              {isConnected && (
                <div className="absolute inset-0 h-3 w-3 rounded-full bg-green-500 animate-ping opacity-75" />
              )}
            </div>
            <div>
              <CardTitle className="text-lg">System Vitals</CardTitle>
              <CardDescription className="font-mono text-xs">
                {isConnected ? 'ONLINE' : 'RECONNECTING...'}
              </CardDescription>
            </div>
          </div>
          <Badge variant={isConnected ? 'default' : 'destructive'}>
            {isConnected ? 'Live' : 'Offline'}
          </Badge>
        </div>
      </CardHeader>

      <CardContent>
        <div className="grid grid-cols-3 gap-4">
          {/* Database Status */}
          <StatusIndicator
            label="Database"
            status={healthStats.database_status}
            icon={<Database className="h-4 w-4" />}
          />

          {/* API Status */}
          <StatusIndicator
            label="API"
            status={healthStats.api_status}
            icon={<Server className="h-4 w-4" />}
          />

          {/* Sync Queue with flow animation */}
          <div className="text-center">
            <div className="flex items-center justify-center gap-2 mb-1">
              <RefreshCw className={cn(
                "h-4 w-4 text-muted-foreground",
                healthStats.pending_syncs > 0 && "animate-spin"
              )} />
              <span className="text-sm text-muted-foreground">Sync Queue</span>
            </div>
            <div className="flex flex-col items-center">
              <SyncFlowIndicator
                pendingCount={healthStats.pending_syncs}
                lagSeconds={healthStats.avg_sync_lag_seconds}
              />
              <span className={cn(
                "text-lg font-semibold",
                healthStats.pending_syncs > 10 ? "text-yellow-500" :
                healthStats.pending_syncs > 0 ? "text-muted-foreground" : "text-green-500"
              )}>
                {healthStats.pending_syncs}
              </span>
              {healthStats.pending_syncs > 0 && healthStats.avg_sync_lag_seconds > 0 && (
                <span className="text-xs text-muted-foreground">
                  {healthStats.avg_sync_lag_seconds.toFixed(1)}s avg lag
                </span>
              )}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

interface StatusIndicatorProps {
  label: string;
  status: string;
  icon: React.ReactNode;
}

function StatusIndicator({ label, status, icon }: StatusIndicatorProps) {
  const isGood = ['connected', 'healthy'].includes(status);
  const isDegraded = status === 'degraded';

  const statusColor = isGood ? 'text-green-500' : isDegraded ? 'text-yellow-500' : 'text-red-500';
  const dotColor = isGood ? 'bg-green-500' : isDegraded ? 'bg-yellow-500' : 'bg-red-500';

  return (
    <div className="text-center">
      <div className="flex items-center justify-center gap-2 mb-1">
        <span className="text-muted-foreground">{icon}</span>
        <span className="text-sm text-muted-foreground">{label}</span>
      </div>
      <div className="flex items-center justify-center gap-2">
        <div className="relative">
          <div className={cn("h-2 w-2 rounded-full", dotColor)} />
          {isGood && (
            <div className={cn(
              "absolute inset-0 h-2 w-2 rounded-full animate-ping opacity-50",
              dotColor
            )} />
          )}
        </div>
        <span className={cn("text-sm font-medium uppercase", statusColor)}>
          {status}
        </span>
      </div>
    </div>
  );
}

interface SyncFlowIndicatorProps {
  pendingCount: number;
  lagSeconds: number;
}

function SyncFlowIndicator({ pendingCount, lagSeconds }: SyncFlowIndicatorProps) {
  if (pendingCount === 0) {
    return (
      <div className="h-2 w-16 bg-muted rounded-full mb-1 flex items-center justify-center">
        <div className="h-1 w-1 rounded-full bg-green-500" />
      </div>
    );
  }

  // Show flowing particles when there are pending syncs
  const particleCount = Math.min(pendingCount, 3);
  // Slower animation when there's more lag
  const duration = Math.max(1, Math.min(3, lagSeconds / 2 + 0.5));

  return (
    <div className="h-2 w-16 bg-muted rounded-full mb-1 overflow-hidden relative">
      {Array.from({ length: particleCount }).map((_, i) => (
        <div
          key={i}
          className="absolute top-0.5 h-1 w-1 rounded-full bg-yellow-500"
          style={{
            animation: `flowRight ${duration}s linear infinite`,
            animationDelay: `${i * (duration / particleCount)}s`,
          }}
        />
      ))}
    </div>
  );
}
