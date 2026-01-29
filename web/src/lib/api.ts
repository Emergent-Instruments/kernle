// In production, NEXT_PUBLIC_API_URL must be set to avoid localhost fallback
// which triggers Chrome's Local Network Access permission prompt
const API_URL = process.env.NEXT_PUBLIC_API_URL || (
  typeof window !== 'undefined' && window.location.hostname === 'localhost'
    ? 'http://localhost:8000'
    : undefined
);

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

async function fetchApi<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    ...options.headers,
  };

  const response = await fetch(`${API_URL}${endpoint}`, {
    ...options,
    headers,
    credentials: 'include',  // Send httpOnly cookies automatically
    cache: 'no-store',  // Prevent browser caching - always fetch fresh data
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Request failed' }));
    throw new ApiError(response.status, error.detail || 'Request failed');
  }

  return response.json();
}

// Auth
export interface RegisterRequest {
  email: string;
  password: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
}

export interface User {
  user_id: string;
  email: string;
  created_at: string;
}

export async function register(data: RegisterRequest): Promise<User> {
  return fetchApi<User>('/auth/register', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export async function login(email: string, password: string): Promise<TokenResponse> {
  const formData = new URLSearchParams();
  formData.append('username', email);
  formData.append('password', password);

  const response = await fetch(`${API_URL}/auth/token`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Login failed' }));
    throw new ApiError(response.status, error.detail || 'Login failed');
  }

  return response.json();
}

export async function getMe(): Promise<User> {
  return fetchApi<User>('/auth/me');
}

export async function exchangeOAuthToken(supabaseToken: string): Promise<TokenResponse> {
  return fetchApi<TokenResponse>('/auth/oauth/token', {
    method: 'POST',
    body: JSON.stringify({ access_token: supabaseToken }),
  });
}

// API Keys - matches backend APIKeyInfo model
export interface ApiKey {
  id: string;  // UUID
  key_prefix: string;  // e.g. "knl_sk_XXXXX..."
  name: string;
  created_at: string;
  last_used_at: string | null;
  is_active: boolean;
}

// Response from creating a new key - includes raw key ONCE
export interface CreateKeyResponse {
  id: string;
  key: string;  // Full API key, shown only once
  key_prefix: string;
  name: string;
  created_at: string;
}

export async function listApiKeys(): Promise<ApiKey[]> {
  return fetchApi<ApiKey[]>('/auth/keys');
}

export async function createApiKey(name?: string): Promise<CreateKeyResponse> {
  return fetchApi<CreateKeyResponse>('/auth/keys', {
    method: 'POST',
    body: JSON.stringify({ name: name || null }),
  });
}

export async function revokeApiKey(keyId: string): Promise<void> {
  await fetchApi<void>(`/auth/keys/${keyId}`, {
    method: 'DELETE',
  });
}

export async function cycleApiKey(keyId: string): Promise<CreateKeyResponse> {
  return fetchApi<CreateKeyResponse>(`/auth/keys/${keyId}/cycle`, {
    method: 'POST',
  });
}

// Admin
export interface AgentSummary {
  agent_id: string;
  user_id: string;
  tier: string;
  created_at: string | null;
  last_sync_at: string | null;
  memory_counts: Record<string, number>;
  embedding_coverage: Record<string, { total: number; with_embedding: number; percent: number }>;
}

export interface SystemStats {
  total_agents: number;
  total_memories: number;
  memories_with_embeddings: number;
  embedding_coverage_percent: number;
  by_table: Record<string, { total: number; with_embedding: number; percent: number }>;
}

export interface BackfillResponse {
  agent_id: string;
  processed: number;
  failed: number;
  tables_updated: Record<string, number>;
}

export async function getSystemStats(): Promise<SystemStats> {
  return fetchApi<SystemStats>('/admin/stats');
}

export async function listAgents(limit = 50, offset = 0): Promise<{ agents: AgentSummary[]; total: number }> {
  return fetchApi<{ agents: AgentSummary[]; total: number }>(`/admin/agents?limit=${limit}&offset=${offset}`);
}

export async function getAgent(agentId: string): Promise<AgentSummary> {
  return fetchApi<AgentSummary>(`/admin/agents/${agentId}`);
}

export async function backfillEmbeddings(agentId: string, limit = 100): Promise<BackfillResponse> {
  return fetchApi<BackfillResponse>('/admin/embeddings/backfill', {
    method: 'POST',
    body: JSON.stringify({ agent_id: agentId, limit }),
  });
}
