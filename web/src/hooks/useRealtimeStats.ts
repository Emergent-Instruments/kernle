'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { getHealthStats, getSystemStats, listAgents, HealthStats, SystemStats, AgentSummary, ApiError } from '@/lib/api';

export interface RealtimeStatsData {
  healthStats: HealthStats | null;
  systemStats: SystemStats | null;
  agents: AgentSummary[];
  totalAgents: number;
}

export interface UseRealtimeStatsOptions {
  pollingInterval?: number;
  enabled?: boolean;
}

export interface UseRealtimeStatsResult {
  stats: RealtimeStatsData;
  isConnected: boolean;
  isLoading: boolean;
  error: string | null;
  accessDenied: boolean;
  refresh: () => Promise<void>;
}

// Minimum polling interval to prevent accidental self-DoS
const MIN_POLLING_INTERVAL = 3000;

export function useRealtimeStats({
  pollingInterval = 5000,
  enabled = true,
}: UseRealtimeStatsOptions = {}): UseRealtimeStatsResult {
  // Enforce minimum polling interval
  const safePollingInterval = Math.max(pollingInterval, MIN_POLLING_INTERVAL);

  const [stats, setStats] = useState<RealtimeStatsData>({
    healthStats: null,
    systemStats: null,
    agents: [],
    totalAgents: 0,
  });
  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [accessDenied, setAccessDenied] = useState(false);

  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const mountedRef = useRef(true);

  const fetchData = useCallback(async () => {
    if (!mountedRef.current) return;

    try {
      const [healthStatsData, systemStatsData, agentsData] = await Promise.all([
        getHealthStats(),
        getSystemStats(),
        listAgents(50, 0),
      ]);

      if (!mountedRef.current) return;

      setStats({
        healthStats: healthStatsData,
        systemStats: systemStatsData,
        agents: agentsData.agents,
        totalAgents: agentsData.total,
      });
      setIsConnected(true);
      setError(null);
      setAccessDenied(false);
    } catch (e) {
      if (!mountedRef.current) return;

      // Check for 403/401 - user is not an admin
      if (e instanceof ApiError && (e.status === 403 || e.status === 401)) {
        setAccessDenied(true);
        setIsConnected(false);
        return;
      }

      setError(e instanceof Error ? e.message : 'Failed to load data');
      setIsConnected(false);
    } finally {
      if (mountedRef.current) {
        setIsLoading(false);
      }
    }
  }, []);

  const refresh = useCallback(async () => {
    setIsLoading(true);
    await fetchData();
  }, [fetchData]);

  useEffect(() => {
    mountedRef.current = true;

    if (enabled) {
      // Initial fetch
      fetchData();

      // Set up polling with enforced minimum interval
      intervalRef.current = setInterval(fetchData, safePollingInterval);
    }

    return () => {
      mountedRef.current = false;
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [enabled, safePollingInterval, fetchData]);

  return {
    stats,
    isConnected,
    isLoading,
    error,
    accessDenied,
    refresh,
  };
}
