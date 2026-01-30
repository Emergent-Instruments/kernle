'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Loader2, RefreshCw, Zap, ShieldX } from 'lucide-react';
import { getSystemStats, listAgents, backfillEmbeddings, SystemStats, AgentSummary, BackfillResponse, ApiError } from '@/lib/api';

export default function AdminPage() {
  const router = useRouter();
  const [stats, setStats] = useState<SystemStats | null>(null);
  const [agents, setAgents] = useState<AgentSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [accessDenied, setAccessDenied] = useState(false);
  const [backfilling, setBackfilling] = useState<string | null>(null);
  const [lastBackfill, setLastBackfill] = useState<BackfillResponse | null>(null);

  const loadData = async () => {
    setLoading(true);
    setError(null);
    setAccessDenied(false);
    try {
      const [statsData, agentsData] = await Promise.all([
        getSystemStats(),
        listAgents(50, 0),
      ]);
      setStats(statsData);
      setAgents(agentsData.agents);
    } catch (e) {
      // Check for 403 Forbidden - user is not an admin
      if (e instanceof ApiError && (e.status === 403 || e.status === 401)) {
        setAccessDenied(true);
        return;
      }
      setError(e instanceof Error ? e.message : 'Failed to load data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleBackfill = async (agentId: string) => {
    setBackfilling(agentId);
    setLastBackfill(null);
    try {
      const result = await backfillEmbeddings(agentId, 100);
      setLastBackfill(result);
      // Reload data to show updated stats
      await loadData();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Backfill failed');
    } finally {
      setBackfilling(null);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin" />
      </div>
    );
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

  if (error) {
    return (
      <div className="space-y-4">
        <div className="text-red-500">Error: {error}</div>
        <Button onClick={loadData}>Retry</Button>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-3xl font-bold">Admin Dashboard</h2>
          <p className="text-muted-foreground">System overview and management</p>
        </div>
        <Button variant="outline" onClick={loadData}>
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      {/* System Stats */}
      {stats && (
        <div className="grid gap-4 md:grid-cols-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Total Agents</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.total_agents}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Total Memories</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.total_memories.toLocaleString()}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">With Embeddings</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.memories_with_embeddings.toLocaleString()}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Embedding Coverage</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.embedding_coverage_percent}%</div>
              <Progress value={stats.embedding_coverage_percent} className="mt-2" />
            </CardContent>
          </Card>
        </div>
      )}

      {/* Embedding Coverage by Table */}
      {stats && (
        <Card>
          <CardHeader>
            <CardTitle>Embedding Coverage by Table</CardTitle>
            <CardDescription>System-wide embedding statistics</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-3">
              {Object.entries(stats.by_table).map(([table, data]) => (
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
                        <span className="text-sm text-green-600">✓ Complete</span>
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
