import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import { useRealtimeStats } from '../../hooks/useRealtimeStats';
import { ApiError } from '../../lib/api';

// Mock the API module
vi.mock('../../lib/api', async () => {
  const actual = await vi.importActual('../../lib/api');
  return {
    ...actual,
    getHealthStats: vi.fn(),
    getSystemStats: vi.fn(),
    listAgents: vi.fn(),
  };
});

import { getHealthStats, getSystemStats, listAgents } from '../../lib/api';

// Type the mocked functions
const mockGetHealthStats = vi.mocked(getHealthStats);
const mockGetSystemStats = vi.mocked(getSystemStats);
const mockListAgents = vi.mocked(listAgents);

describe('useRealtimeStats', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  // Sample test data
  const mockHealthStats = {
    database_status: 'connected',
    api_status: 'healthy',
    memory_distribution: { episodes: 100, beliefs: 50 },
    pending_syncs: 5,
    avg_sync_lag_seconds: 2.5,
    confidence_distribution: {
      '0.0-0.2': 10,
      '0.2-0.4': 20,
      '0.4-0.6': 30,
      '0.6-0.8': 25,
      '0.8-1.0': 15,
    },
    total_memories: 150,
    active_memories: 140,
    forgotten_memories: 10,
    protected_memories: 20,
  };

  const mockSystemStats = {
    total_agents: 10,
    total_memories: 150,
    memories_with_embeddings: 140,
    embedding_coverage_percent: 93.3,
    by_table: {
      episodes: { total: 100, with_embedding: 95, percent: 95 },
      beliefs: { total: 50, with_embedding: 45, percent: 90 },
    },
  };

  const mockAgentsResponse = {
    agents: [
      {
        agent_id: 'agent-1',
        user_id: 'user-1',
        tier: 'free',
        created_at: '2024-01-01T00:00:00Z',
        last_sync_at: '2024-01-15T00:00:00Z',
        memory_counts: { episodes: 50, beliefs: 25 },
        embedding_coverage: {
          episodes: { total: 50, with_embedding: 48, percent: 96 },
        },
      },
    ],
    total: 1,
  };

  describe('Initial fetch behavior', () => {
    it('fetches data immediately on mount when enabled', async () => {
      mockGetHealthStats.mockResolvedValue(mockHealthStats);
      mockGetSystemStats.mockResolvedValue(mockSystemStats);
      mockListAgents.mockResolvedValue(mockAgentsResponse);

      const { result } = renderHook(() => useRealtimeStats({ enabled: true }));

      // Initially loading
      expect(result.current.isLoading).toBe(true);

      // Wait for initial fetch to complete
      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // Verify all APIs were called (at least once due to React StrictMode potential double render)
      expect(mockGetHealthStats).toHaveBeenCalled();
      expect(mockGetSystemStats).toHaveBeenCalled();
      expect(mockListAgents).toHaveBeenCalledWith(50, 0);
    });

    it('does not fetch when disabled', async () => {
      const { result } = renderHook(() => useRealtimeStats({ enabled: false }));

      // Wait a bit to ensure no fetch happens
      await new Promise(resolve => setTimeout(resolve, 100));

      expect(mockGetHealthStats).not.toHaveBeenCalled();
      expect(mockGetSystemStats).not.toHaveBeenCalled();
      expect(mockListAgents).not.toHaveBeenCalled();
      expect(result.current.isLoading).toBe(true); // Never completed loading
    });

    it('updates state with fetched data', async () => {
      mockGetHealthStats.mockResolvedValue(mockHealthStats);
      mockGetSystemStats.mockResolvedValue(mockSystemStats);
      mockListAgents.mockResolvedValue(mockAgentsResponse);

      const { result } = renderHook(() => useRealtimeStats({ enabled: true }));

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // Verify state was updated correctly
      expect(result.current.stats.healthStats).toEqual(mockHealthStats);
      expect(result.current.stats.systemStats).toEqual(mockSystemStats);
      expect(result.current.stats.agents).toEqual(mockAgentsResponse.agents);
      expect(result.current.stats.totalAgents).toBe(mockAgentsResponse.total);
      expect(result.current.isConnected).toBe(true);
      expect(result.current.error).toBeNull();
      expect(result.current.accessDenied).toBe(false);
    });
  });

  describe('Polling behavior', () => {
    beforeEach(() => {
      vi.useFakeTimers({ shouldAdvanceTime: true });
    });

    afterEach(() => {
      vi.useRealTimers();
    });

    it('polls at the specified interval', async () => {
      mockGetHealthStats.mockResolvedValue(mockHealthStats);
      mockGetSystemStats.mockResolvedValue(mockSystemStats);
      mockListAgents.mockResolvedValue(mockAgentsResponse);

      const pollingInterval = 3000;

      renderHook(() =>
        useRealtimeStats({ enabled: true, pollingInterval })
      );

      // Wait for initial fetch
      await vi.waitFor(() => {
        expect(mockGetHealthStats).toHaveBeenCalled();
      });

      const callsAfterInitial = mockGetHealthStats.mock.calls.length;

      // Advance time to trigger poll
      await act(async () => {
        vi.advanceTimersByTime(pollingInterval);
      });

      // Wait for the poll to complete
      await vi.waitFor(() => {
        expect(mockGetHealthStats.mock.calls.length).toBeGreaterThan(callsAfterInitial);
      });
    });

    it('stops polling when unmounted', async () => {
      mockGetHealthStats.mockResolvedValue(mockHealthStats);
      mockGetSystemStats.mockResolvedValue(mockSystemStats);
      mockListAgents.mockResolvedValue(mockAgentsResponse);

      const { unmount } = renderHook(() =>
        useRealtimeStats({ enabled: true, pollingInterval: 1000 })
      );

      // Wait for initial fetch
      await vi.waitFor(() => {
        expect(mockGetHealthStats).toHaveBeenCalled();
      });

      const callsBeforeUnmount = mockGetHealthStats.mock.calls.length;

      // Unmount
      unmount();

      // Advance time - should NOT trigger more fetches
      await act(async () => {
        vi.advanceTimersByTime(5000);
      });

      // Give some time for any pending calls to settle
      await new Promise(resolve => setTimeout(resolve, 10));

      // No additional calls after unmount
      expect(mockGetHealthStats.mock.calls.length).toBe(callsBeforeUnmount);
    });
  });

  describe('Error handling', () => {
    it('handles network errors gracefully', async () => {
      mockGetHealthStats.mockRejectedValue(new Error('Network error'));
      mockGetSystemStats.mockResolvedValue(mockSystemStats);
      mockListAgents.mockResolvedValue(mockAgentsResponse);

      const { result } = renderHook(() => useRealtimeStats({ enabled: true }));

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.error).toBe('Network error');
      expect(result.current.isConnected).toBe(false);
      expect(result.current.accessDenied).toBe(false);
    });

    it('handles 403 access denied errors', async () => {
      mockGetHealthStats.mockRejectedValue(new ApiError(403, 'Admin access required'));
      mockGetSystemStats.mockResolvedValue(mockSystemStats);
      mockListAgents.mockResolvedValue(mockAgentsResponse);

      const { result } = renderHook(() => useRealtimeStats({ enabled: true }));

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.accessDenied).toBe(true);
      expect(result.current.isConnected).toBe(false);
      // Note: error is NOT set for access denied - it's a separate state
      expect(result.current.error).toBeNull();
    });

    it('handles 401 unauthorized errors', async () => {
      mockGetHealthStats.mockRejectedValue(new ApiError(401, 'Not authenticated'));
      mockGetSystemStats.mockResolvedValue(mockSystemStats);
      mockListAgents.mockResolvedValue(mockAgentsResponse);

      const { result } = renderHook(() => useRealtimeStats({ enabled: true }));

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.accessDenied).toBe(true);
      expect(result.current.isConnected).toBe(false);
    });

    it('handles non-Error exceptions', async () => {
      mockGetHealthStats.mockRejectedValue('string error');
      mockGetSystemStats.mockResolvedValue(mockSystemStats);
      mockListAgents.mockResolvedValue(mockAgentsResponse);

      const { result } = renderHook(() => useRealtimeStats({ enabled: true }));

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.error).toBe('Failed to load data');
    });
  });

  describe('Manual refresh', () => {
    it('provides a refresh function that fetches data', async () => {
      mockGetHealthStats.mockResolvedValue(mockHealthStats);
      mockGetSystemStats.mockResolvedValue(mockSystemStats);
      mockListAgents.mockResolvedValue(mockAgentsResponse);

      // Use a long polling interval so it doesn't interfere
      const { result } = renderHook(() =>
        useRealtimeStats({ enabled: true, pollingInterval: 60000 })
      );

      // Wait for initial fetch
      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      const initialCallCount = mockGetHealthStats.mock.calls.length;

      // Manual refresh
      await act(async () => {
        await result.current.refresh();
      });

      expect(mockGetHealthStats.mock.calls.length).toBeGreaterThan(initialCallCount);
    });
  });

  describe('Initial state', () => {
    it('returns correct initial state structure', () => {
      // Render with enabled: false to prevent any fetching
      const { result } = renderHook(() =>
        useRealtimeStats({ enabled: false })
      );

      expect(result.current.stats).toEqual({
        healthStats: null,
        systemStats: null,
        agents: [],
        totalAgents: 0,
      });
      expect(result.current.isConnected).toBe(false);
      expect(result.current.isLoading).toBe(true);
      expect(result.current.error).toBeNull();
      expect(result.current.accessDenied).toBe(false);
      expect(typeof result.current.refresh).toBe('function');
    });
  });

  describe('Data transformation', () => {
    it('correctly maps agents response to stats', async () => {
      const multiAgentResponse = {
        agents: [
          { agent_id: 'a1', user_id: 'u1', tier: 'free', created_at: null, last_sync_at: null, memory_counts: {}, embedding_coverage: {} },
          { agent_id: 'a2', user_id: 'u2', tier: 'paid', created_at: null, last_sync_at: null, memory_counts: {}, embedding_coverage: {} },
        ],
        total: 25,
      };

      mockGetHealthStats.mockResolvedValue(mockHealthStats);
      mockGetSystemStats.mockResolvedValue(mockSystemStats);
      mockListAgents.mockResolvedValue(multiAgentResponse);

      const { result } = renderHook(() => useRealtimeStats({ enabled: true }));

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // Verify agents array is correctly set
      expect(result.current.stats.agents).toHaveLength(2);
      expect(result.current.stats.agents[0].agent_id).toBe('a1');
      expect(result.current.stats.agents[1].tier).toBe('paid');

      // Total should be the full count, not array length
      expect(result.current.stats.totalAgents).toBe(25);
    });
  });

  describe('Cleanup behavior', () => {
    it('does not update state after unmount when fetch completes', async () => {
      // Use a delayed promise to simulate slow network
      let resolvePromise: () => void;
      const slowPromise = new Promise<typeof mockHealthStats>((resolve) => {
        resolvePromise = () => resolve(mockHealthStats);
      });

      mockGetHealthStats.mockReturnValue(slowPromise);
      mockGetSystemStats.mockResolvedValue(mockSystemStats);
      mockListAgents.mockResolvedValue(mockAgentsResponse);

      const { unmount, result } = renderHook(() => useRealtimeStats({ enabled: true }));

      // The hook should be in loading state
      expect(result.current.isLoading).toBe(true);

      // Unmount before resolve
      unmount();

      // Resolve after unmount - should not throw "state update on unmounted component"
      resolvePromise!();

      // Give a moment for any potential state updates
      await new Promise(resolve => setTimeout(resolve, 50));

      // Test passes if no React warnings about state updates on unmounted components
    });
  });
});

describe('useRealtimeStats - Aggregation Logic Tests', () => {
  /**
   * These tests verify the business logic for how the hook aggregates data.
   * Rather than testing mock returns equal mock values (tautological),
   * we test that the hook correctly transforms and combines API responses.
   */

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('combines data from all three API endpoints into a single stats object', async () => {
    const healthStats = {
      database_status: 'connected',
      api_status: 'healthy',
      memory_distribution: { episodes: 50, beliefs: 30 },
      pending_syncs: 2,
      avg_sync_lag_seconds: 1.5,
      confidence_distribution: { '0.8-1.0': 20 },
      total_memories: 80,
      active_memories: 75,
      forgotten_memories: 5,
      protected_memories: 10,
    };

    const systemStats = {
      total_agents: 3,
      total_memories: 80,
      memories_with_embeddings: 78,
      embedding_coverage_percent: 97.5,
      by_table: {},
    };

    const agentsResponse = {
      agents: [
        { agent_id: 'test', user_id: 'u1', tier: 'free', created_at: null, last_sync_at: null, memory_counts: {}, embedding_coverage: {} }
      ],
      total: 3,
    };

    const mockGetHealthStats = vi.mocked(getHealthStats);
    const mockGetSystemStats = vi.mocked(getSystemStats);
    const mockListAgents = vi.mocked(listAgents);

    mockGetHealthStats.mockResolvedValue(healthStats);
    mockGetSystemStats.mockResolvedValue(systemStats);
    mockListAgents.mockResolvedValue(agentsResponse);

    const { result } = renderHook(() => useRealtimeStats({ enabled: true }));

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    // Verify the stats object combines all data correctly
    expect(result.current.stats.healthStats).toEqual(healthStats);
    expect(result.current.stats.systemStats).toEqual(systemStats);
    expect(result.current.stats.agents).toEqual(agentsResponse.agents);
    expect(result.current.stats.totalAgents).toBe(agentsResponse.total);

    // These are derived from the actual data, not just mock passthrough
    expect(result.current.stats.totalAgents).not.toBe(result.current.stats.agents.length);
    expect(result.current.stats.totalAgents).toBe(3); // From response.total
    expect(result.current.stats.agents.length).toBe(1); // Only one in array

    // Verify the connection state reflects successful fetch
    expect(result.current.isConnected).toBe(true);
    expect(result.current.accessDenied).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it('handles partial API failures gracefully', async () => {
    // If one API fails, the whole fetch should fail
    const mockGetHealthStats = vi.mocked(getHealthStats);
    const mockGetSystemStats = vi.mocked(getSystemStats);
    const mockListAgents = vi.mocked(listAgents);

    // Only healthStats fails
    mockGetHealthStats.mockRejectedValue(new Error('Health check failed'));
    mockGetSystemStats.mockResolvedValue({
      total_agents: 5,
      total_memories: 100,
      memories_with_embeddings: 100,
      embedding_coverage_percent: 100,
      by_table: {},
    });
    mockListAgents.mockResolvedValue({ agents: [], total: 5 });

    const { result } = renderHook(() => useRealtimeStats({ enabled: true }));

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    // Should have error state
    expect(result.current.error).toBe('Health check failed');
    expect(result.current.isConnected).toBe(false);

    // Stats should remain at initial values
    expect(result.current.stats.healthStats).toBeNull();
  });

  it('distinguishes between auth errors and other errors', async () => {
    const mockGetHealthStats = vi.mocked(getHealthStats);
    const mockGetSystemStats = vi.mocked(getSystemStats);
    const mockListAgents = vi.mocked(listAgents);

    // 403 should set accessDenied, not error
    mockGetHealthStats.mockRejectedValue(new ApiError(403, 'Forbidden'));
    mockGetSystemStats.mockResolvedValue({
      total_agents: 0,
      total_memories: 0,
      memories_with_embeddings: 0,
      embedding_coverage_percent: 0,
      by_table: {},
    });
    mockListAgents.mockResolvedValue({ agents: [], total: 0 });

    const { result } = renderHook(() => useRealtimeStats({ enabled: true }));

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    // Access denied should be set, but error should be null
    expect(result.current.accessDenied).toBe(true);
    expect(result.current.error).toBeNull();
    expect(result.current.isConnected).toBe(false);
  });
});

describe('useRealtimeStats - Minimum Polling Interval Security', () => {
  /**
   * These tests verify that the minimum polling interval is enforced
   * to prevent accidental self-DoS attacks from rapid polling.
   *
   * The security fix ensures that even if a developer passes a very low
   * polling interval, the hook enforces a minimum of 3000ms.
   */

  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers({ shouldAdvanceTime: true });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('enforces minimum polling interval of 3000ms when lower value is provided', async () => {
    const mockGetHealthStats = vi.mocked(getHealthStats);
    const mockGetSystemStats = vi.mocked(getSystemStats);
    const mockListAgents = vi.mocked(listAgents);

    mockGetHealthStats.mockResolvedValue({
      database_status: 'connected',
      api_status: 'healthy',
      memory_distribution: {},
      pending_syncs: 0,
      avg_sync_lag_seconds: 0,
      confidence_distribution: { '0.0-0.2': 0, '0.2-0.4': 0, '0.4-0.6': 0, '0.6-0.8': 0, '0.8-1.0': 0 },
      total_memories: 0,
      active_memories: 0,
      forgotten_memories: 0,
      protected_memories: 0,
    });
    mockGetSystemStats.mockResolvedValue({
      total_agents: 0,
      total_memories: 0,
      memories_with_embeddings: 0,
      embedding_coverage_percent: 0,
      by_table: {},
    });
    mockListAgents.mockResolvedValue({ agents: [], total: 0 });

    // Request a dangerously low polling interval (100ms)
    renderHook(() => useRealtimeStats({ enabled: true, pollingInterval: 100 }));

    // Wait for initial fetch
    await vi.waitFor(() => {
      expect(mockGetHealthStats).toHaveBeenCalled();
    });

    const callsAfterInitial = mockGetHealthStats.mock.calls.length;

    // Advance by 500ms - if minimum wasn't enforced, this would trigger multiple polls
    await act(async () => {
      vi.advanceTimersByTime(500);
    });

    // Should NOT have polled again yet (minimum is 3000ms)
    expect(mockGetHealthStats.mock.calls.length).toBe(callsAfterInitial);

    // Advance by another 2000ms (total 2500ms) - still under 3000ms minimum
    await act(async () => {
      vi.advanceTimersByTime(2000);
    });

    // Still should NOT have polled again
    expect(mockGetHealthStats.mock.calls.length).toBe(callsAfterInitial);

    // Now advance past the 3000ms minimum
    await act(async () => {
      vi.advanceTimersByTime(600);
    });

    // NOW it should have polled
    await vi.waitFor(() => {
      expect(mockGetHealthStats.mock.calls.length).toBeGreaterThan(callsAfterInitial);
    });
  });

  it('uses provided polling interval when it exceeds minimum', async () => {
    const mockGetHealthStats = vi.mocked(getHealthStats);
    const mockGetSystemStats = vi.mocked(getSystemStats);
    const mockListAgents = vi.mocked(listAgents);

    mockGetHealthStats.mockResolvedValue({
      database_status: 'connected',
      api_status: 'healthy',
      memory_distribution: {},
      pending_syncs: 0,
      avg_sync_lag_seconds: 0,
      confidence_distribution: { '0.0-0.2': 0, '0.2-0.4': 0, '0.4-0.6': 0, '0.6-0.8': 0, '0.8-1.0': 0 },
      total_memories: 0,
      active_memories: 0,
      forgotten_memories: 0,
      protected_memories: 0,
    });
    mockGetSystemStats.mockResolvedValue({
      total_agents: 0,
      total_memories: 0,
      memories_with_embeddings: 0,
      embedding_coverage_percent: 0,
      by_table: {},
    });
    mockListAgents.mockResolvedValue({ agents: [], total: 0 });

    // Request a valid polling interval of 10 seconds (above minimum)
    const customInterval = 10000;
    renderHook(() => useRealtimeStats({ enabled: true, pollingInterval: customInterval }));

    // Wait for initial fetch
    await vi.waitFor(() => {
      expect(mockGetHealthStats).toHaveBeenCalled();
    });

    const callsAfterInitial = mockGetHealthStats.mock.calls.length;

    // Advance by 5000ms - should NOT poll yet (custom interval is 10000ms)
    await act(async () => {
      vi.advanceTimersByTime(5000);
    });

    // Should NOT have polled again
    expect(mockGetHealthStats.mock.calls.length).toBe(callsAfterInitial);

    // Advance by another 5100ms to exceed the 10 second interval
    await act(async () => {
      vi.advanceTimersByTime(5100);
    });

    // Should have polled now
    await vi.waitFor(() => {
      expect(mockGetHealthStats.mock.calls.length).toBeGreaterThan(callsAfterInitial);
    });
  });

  it('enforces minimum for exactly 3000ms boundary value', async () => {
    const mockGetHealthStats = vi.mocked(getHealthStats);
    const mockGetSystemStats = vi.mocked(getSystemStats);
    const mockListAgents = vi.mocked(listAgents);

    mockGetHealthStats.mockResolvedValue({
      database_status: 'connected',
      api_status: 'healthy',
      memory_distribution: {},
      pending_syncs: 0,
      avg_sync_lag_seconds: 0,
      confidence_distribution: { '0.0-0.2': 0, '0.2-0.4': 0, '0.4-0.6': 0, '0.6-0.8': 0, '0.8-1.0': 0 },
      total_memories: 0,
      active_memories: 0,
      forgotten_memories: 0,
      protected_memories: 0,
    });
    mockGetSystemStats.mockResolvedValue({
      total_agents: 0,
      total_memories: 0,
      memories_with_embeddings: 0,
      embedding_coverage_percent: 0,
      by_table: {},
    });
    mockListAgents.mockResolvedValue({ agents: [], total: 0 });

    // Request exactly the minimum (3000ms) - should use as-is
    renderHook(() => useRealtimeStats({ enabled: true, pollingInterval: 3000 }));

    // Wait for initial fetch
    await vi.waitFor(() => {
      expect(mockGetHealthStats).toHaveBeenCalled();
    });

    const callsAfterInitial = mockGetHealthStats.mock.calls.length;

    // Advance by 2900ms - should NOT poll yet
    await act(async () => {
      vi.advanceTimersByTime(2900);
    });

    expect(mockGetHealthStats.mock.calls.length).toBe(callsAfterInitial);

    // Advance past 3000ms
    await act(async () => {
      vi.advanceTimersByTime(200);
    });

    // Should have polled now
    await vi.waitFor(() => {
      expect(mockGetHealthStats.mock.calls.length).toBeGreaterThan(callsAfterInitial);
    });
  });

  it('prevents rapid polling that could cause self-DoS', async () => {
    const mockGetHealthStats = vi.mocked(getHealthStats);
    const mockGetSystemStats = vi.mocked(getSystemStats);
    const mockListAgents = vi.mocked(listAgents);

    mockGetHealthStats.mockResolvedValue({
      database_status: 'connected',
      api_status: 'healthy',
      memory_distribution: {},
      pending_syncs: 0,
      avg_sync_lag_seconds: 0,
      confidence_distribution: { '0.0-0.2': 0, '0.2-0.4': 0, '0.4-0.6': 0, '0.6-0.8': 0, '0.8-1.0': 0 },
      total_memories: 0,
      active_memories: 0,
      forgotten_memories: 0,
      protected_memories: 0,
    });
    mockGetSystemStats.mockResolvedValue({
      total_agents: 0,
      total_memories: 0,
      memories_with_embeddings: 0,
      embedding_coverage_percent: 0,
      by_table: {},
    });
    mockListAgents.mockResolvedValue({ agents: [], total: 0 });

    // Request extremely aggressive polling (1ms) - a common mistake
    renderHook(() => useRealtimeStats({ enabled: true, pollingInterval: 1 }));

    // Wait for initial fetch
    await vi.waitFor(() => {
      expect(mockGetHealthStats).toHaveBeenCalled();
    });

    const callsAfterInitial = mockGetHealthStats.mock.calls.length;

    // Advance by 1000ms - without protection, this would be 1000 calls!
    await act(async () => {
      vi.advanceTimersByTime(1000);
    });

    // With protection, should still be at initial call count
    // This is the security assertion - without the fix, this would have hammered the API
    expect(mockGetHealthStats.mock.calls.length).toBe(callsAfterInitial);

    // Advance by another 1500ms (total 2500ms) - still under 3000ms minimum
    await act(async () => {
      vi.advanceTimersByTime(1500);
    });

    // Still at initial count (haven't reached 3000ms total yet from poll setup)
    expect(mockGetHealthStats.mock.calls.length).toBe(callsAfterInitial);

    // Now hit the 3100ms mark from initial - should get exactly one more call
    await act(async () => {
      vi.advanceTimersByTime(600);
    });

    await vi.waitFor(() => {
      expect(mockGetHealthStats.mock.calls.length).toBe(callsAfterInitial + 1);
    });

    // Critically: without the minimum interval protection, at 1ms polling,
    // we would have seen 3100+ calls by now. But we only see initial + 1.
    // This proves the security fix is working.
  });
});
