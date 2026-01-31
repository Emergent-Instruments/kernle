'use client';

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { HealthStats } from '@/lib/api';
import { Database, Server, RefreshCw, Wifi, WifiOff } from 'lucide-react';

interface HealthStatusProps {
  healthStats: HealthStats;
  isConnected: boolean;
}

export function HealthStatus({ healthStats, isConnected }: HealthStatusProps) {
  const getDatabaseStatusColor = (status: string) => {
    switch (status) {
      case 'connected':
        return 'bg-green-500';
      case 'degraded':
        return 'bg-yellow-500';
      default:
        return 'bg-red-500';
    }
  };

  const getApiStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'bg-green-500';
      case 'degraded':
        return 'bg-yellow-500';
      default:
        return 'bg-red-500';
    }
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex justify-between items-start">
          <div>
            <CardTitle>System Health</CardTitle>
            <CardDescription>Real-time system status</CardDescription>
          </div>
          <Badge variant={isConnected ? 'default' : 'destructive'} className="flex items-center gap-1">
            {isConnected ? (
              <>
                <Wifi className="h-3 w-3" />
                Live
              </>
            ) : (
              <>
                <WifiOff className="h-3 w-3" />
                Offline
              </>
            )}
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="flex flex-wrap gap-4">
          {/* Database Status */}
          <div className="flex items-center gap-2">
            <Database className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm text-muted-foreground">Database:</span>
            <Badge variant="outline" className="flex items-center gap-1">
              <span className={`h-2 w-2 rounded-full ${getDatabaseStatusColor(healthStats.database_status)}`} />
              {healthStats.database_status}
            </Badge>
          </div>

          {/* API Status */}
          <div className="flex items-center gap-2">
            <Server className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm text-muted-foreground">API:</span>
            <Badge variant="outline" className="flex items-center gap-1">
              <span className={`h-2 w-2 rounded-full ${getApiStatusColor(healthStats.api_status)}`} />
              {healthStats.api_status}
            </Badge>
          </div>

          {/* Sync Queue */}
          <div className="flex items-center gap-2">
            <RefreshCw className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm text-muted-foreground">Pending Syncs:</span>
            <Badge variant={healthStats.pending_syncs > 0 ? 'secondary' : 'outline'}>
              {healthStats.pending_syncs}
              {healthStats.pending_syncs > 0 && healthStats.avg_sync_lag_seconds > 0 && (
                <span className="ml-1 text-xs">
                  (avg {healthStats.avg_sync_lag_seconds.toFixed(1)}s lag)
                </span>
              )}
            </Badge>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
