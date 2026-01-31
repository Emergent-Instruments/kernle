'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Loader2, Zap, ShieldX } from 'lucide-react';
import { backfillEmbeddings, BackfillResponse } from '@/lib/api';
import { useRealtimeStats } from '@/hooks/useRealtimeStats';
import { AdminDashboardSkeleton } from '@/components/admin/AdminSkeleton';
import { HealthStatus } from '@/components/admin/HealthStatus';
import { MemoryDistribution } from '@/components/admin/MemoryDistribution';
import { UsageStats } from '@/components/admin/UsageStats';

function CircularProgress({ value, size }: { value: number; size: number }) {
  const strokeWidth = 4;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (value / 100) * circumference;

  return (
    <svg width={size} height={size} className="-rotate-90">
      <circle
        cx={size / 2}
        cy={size / 2}
        r={radius}
        fill="none"
        stroke="currentColor"
        strokeWidth={strokeWidth}
        className="text-muted"
      />
      <circle
        cx={size / 2}
        cy={size / 2}
        r={radius}
        fill="none"
        stroke="currentColor"
        strokeWidth={strokeWidth}
        strokeDasharray={circumference}
        strokeDashoffset={offset}
        strokeLinecap="round"
        className="text-primary transition-all duration-500"
      />
    </svg>
  );
}

export default function AdminPage() {
  const router = useRouter();
  const { stats, isConnected, isLoading, error, accessDenied } = useRealtimeStats({
    pollingInterval: 5000,
  });
  const [backfilling, setBackfilling] = useState<string | null>(null);
  const [lastBackfill, setLastBackfill] = useState<BackfillResponse | null>(null);
  const [backfillError, setBackfillError] = useState<string | null>(null);

  const handleBackfill = async (agentId: string) => {
    setBackfilling(agentId);
    setLastBackfill(null);
    setBackfillError(null);
    try {
      const result = await backfillEmbeddings(agentId, 100);
      setLastBackfill(result);
    } catch (e) {
      setBackfillError(e instanceof Error ? e.message : 'Backfill failed');
    } finally {
      setBackfilling(null);
    }
  };

  // Show skeleton during initial load
  if (isLoading && !stats.systemStats) {
    return <AdminDashboardSkeleton />;
  }

  if (accessDenied) {
    return (
      <div className="flex flex-col items-center justify-center h-64 space-y-4">
        <ShieldX className="h-16 w-16 text-red-500" />
        <h2 className="text-2xl font-bold">Access Denied</h2>
        <p className="text-muted-foreground">You do not have permission to access the admin dashboard.</p>
        <Button onClick={() => router.push('/dashboard')}>
          Return to Dashboard
        </Button>
      </div>
    );
  }

  if (error && !stats.systemStats) {
    return (
      <div className="space-y-4">
        <div className="text-red-500">Error: {error}</div>
        <Button onClick={() => window.location.reload()}>Retry</Button>
      </div>
    );
  }

  const { systemStats, healthStats, agents } = stats;

  return (
    <div className="space-y-8">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-3xl font-bold">Admin Dashboard</h2>
          <p className="text-muted-foreground">System overview and management</p>
        </div>
        <Badge variant={isConnected ? 'default' : 'secondary'} className="text-xs">
          {isConnected ? 'Live' : 'Reconnecting...'}
        </Badge>
      </div>

      {/* System Stats */}
      {systemStats && (
        <div className="grid gap-4 md:grid-cols-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Total Agents</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{systemStats.total_agents}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Total Memories</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{systemStats.total_memories.toLocaleString()}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">With Embeddings</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{systemStats.memories_with_embeddings.toLocaleString()}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Embedding Coverage</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-3">
                <CircularProgress value={systemStats.embedding_coverage_percent} size={48} />
                <div className="text-2xl font-bold">{systemStats.embedding_coverage_percent}%</div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* System Health */}
      {healthStats && (
        <HealthStatus healthStats={healthStats} isConnected={isConnected} />
      )}

      {/* Memory Distribution */}
      {healthStats && (
        <MemoryDistribution healthStats={healthStats} />
      )}

      {/* Confidence Distribution */}
      {healthStats && (
        <UsageStats healthStats={healthStats} />
      )}

      {/* Embedding Coverage by Table */}
      {systemStats && (
        <Card>
          <CardHeader>
            <CardTitle>Embedding Coverage by Table</CardTitle>
            <CardDescription>System-wide embedding statistics</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-3">
              {Object.entries(systemStats.by_table).map(([table, data]) => (
                <div key={table} className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="font-medium">{table}</span>
                    <span className="text-muted-foreground">
                      {data.with_embedding}/{data.total}
                    </span>
                  </div>
                  <Progress value={data.percent} />
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Last Backfill Result */}
      {lastBackfill && (
        <Card className="border-green-500">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Backfill Complete</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-sm">
              Agent: <strong>{lastBackfill.agent_id}</strong> |
              Processed: <strong>{lastBackfill.processed}</strong> |
              Failed: <strong>{lastBackfill.failed}</strong>
              {Object.keys(lastBackfill.tables_updated).length > 0 && (
                <span> | Tables: {Object.entries(lastBackfill.tables_updated).map(([t, n]) => `${t}(${n})`).join(', ')}</span>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Backfill Error */}
      {backfillError && (
        <Card className="border-red-500">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-red-500">Backfill Error</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-sm text-red-500">{backfillError}</div>
          </CardContent>
        </Card>
      )}

      {/* Agents Table */}
      <Card>
        <CardHeader>
          <CardTitle>Agents</CardTitle>
          <CardDescription>All registered agents and their memory stats</CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Agent ID</TableHead>
                <TableHead>Tier</TableHead>
                <TableHead>Memories</TableHead>
                <TableHead>Embedding %</TableHead>
                <TableHead>Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {agents.map((agent) => {
                const totalMemories = Object.values(agent.memory_counts).reduce((a, b) => a + b, 0);
                const totalEmbedded = Object.values(agent.embedding_coverage).reduce(
                  (a, b) => a + b.with_embedding, 0
                );
                const coveragePercent = totalMemories > 0
                  ? Math.round(totalEmbedded / totalMemories * 100)
                  : 100;
                const needsBackfill = coveragePercent < 100 && totalMemories > 0;

                return (
                  <TableRow key={agent.agent_id}>
                    <TableCell className="font-mono">{agent.agent_id}</TableCell>
                    <TableCell>
                      <Badge variant={agent.tier === 'unlimited' ? 'default' : 'secondary'}>
                        {agent.tier}
                      </Badge>
                    </TableCell>
                    <TableCell>{totalMemories.toLocaleString()}</TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <Progress value={coveragePercent} className="w-20" />
                        <span className="text-sm text-muted-foreground">{coveragePercent}%</span>
                      </div>
                    </TableCell>
                    <TableCell>
                      {needsBackfill && (
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => handleBackfill(agent.agent_id)}
                          disabled={backfilling === agent.agent_id}
                        >
                          {backfilling === agent.agent_id ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : (
                            <>
                              <Zap className="h-4 w-4 mr-1" />
                              Backfill
                            </>
                          )}
                        </Button>
                      )}
                      {!needsBackfill && (
                        <span className="text-sm text-green-600">Complete</span>
                      )}
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
