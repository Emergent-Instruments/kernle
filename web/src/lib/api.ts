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
  const token = typeof window !== 'undefined' ? localStorage.getItem('token') : null;
  
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    ...options.headers,
  };
  
  if (token) {
    (headers as Record<string, string>)['Authorization'] = `Bearer ${token}`;
  }
  
  const response = await fetch(`${API_URL}${endpoint}`, {
    ...options,
    headers,
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

// API Keys
export interface ApiKey {
  key_id: string;
  prefix: string;
  name: string | null;
  created_at: string;
  last_used_at: string | null;
  revoked_at: string | null;
}

export interface CreateKeyResponse {
  key_id: string;
  api_key: string;
  prefix: string;
  name: string | null;
}

export async function listApiKeys(): Promise<ApiKey[]> {
  return fetchApi<ApiKey[]>('/keys');
}

export async function createApiKey(name?: string): Promise<CreateKeyResponse> {
  return fetchApi<CreateKeyResponse>('/keys', {
    method: 'POST',
    body: JSON.stringify({ name: name || null }),
  });
}

export async function revokeApiKey(keyId: string): Promise<void> {
  await fetchApi<void>(`/keys/${keyId}`, {
    method: 'DELETE',
  });
}

export async function cycleApiKey(keyId: string): Promise<CreateKeyResponse> {
  return fetchApi<CreateKeyResponse>(`/keys/${keyId}/cycle`, {
    method: 'POST',
  });
}
