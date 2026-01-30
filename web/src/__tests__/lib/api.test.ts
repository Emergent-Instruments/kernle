import { describe, it, expect, vi, beforeEach } from 'vitest';
import { ApiError, register, login, getMe, listApiKeys, createApiKey, revokeApiKey } from '../../lib/api';

// Mock global fetch
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('ApiError', () => {
  it('creates error with status and message', () => {
    const error = new ApiError(401, 'Unauthorized');
    expect(error.status).toBe(401);
    expect(error.message).toBe('Unauthorized');
    expect(error.name).toBe('ApiError');
  });

  it('is instance of Error', () => {
    const error = new ApiError(500, 'Server error');
    expect(error).toBeInstanceOf(Error);
  });
});

describe('Auth API functions', () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  describe('register', () => {
    it('sends POST request with email and password', async () => {
      const mockUser = { user_id: 'usr_123', email: 'test@example.com', created_at: '2024-01-01' };
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockUser),
      });

      const result = await register({ email: 'test@example.com', password: 'password123' });

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/auth/register'),
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({ email: 'test@example.com', password: 'password123' }),
        })
      );
      expect(result).toEqual(mockUser);
    });

    it('throws ApiError on failure', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: () => Promise.resolve({ detail: 'Email already exists' }),
      });

      await expect(register({ email: 'test@example.com', password: 'pass' }))
        .rejects.toThrow(ApiError);
    });
  });

  describe('login', () => {
    it('sends form-urlencoded POST request', async () => {
      const mockToken = { access_token: 'token123', token_type: 'bearer' };
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockToken),
      });

      const result = await login('test@example.com', 'password');

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/auth/token'),
        expect.objectContaining({
          method: 'POST',
          headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        })
      );
      // Verify form data contains username and password
      const call = mockFetch.mock.calls[0];
      const body = call[1].body as URLSearchParams;
      expect(body.get('username')).toBe('test@example.com');
      expect(body.get('password')).toBe('password');
      expect(result).toEqual(mockToken);
    });

    it('throws ApiError on invalid credentials', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 401,
        json: () => Promise.resolve({ detail: 'Invalid credentials' }),
      });

      await expect(login('bad@example.com', 'wrong'))
        .rejects.toThrow('Invalid credentials');
    });
  });

  describe('getMe', () => {
    it('fetches current user with credentials', async () => {
      const mockUser = { user_id: 'usr_123', email: 'test@example.com', created_at: '2024-01-01' };
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockUser),
      });

      const result = await getMe();

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/auth/me'),
        expect.objectContaining({
          credentials: 'include',
        })
      );
      expect(result).toEqual(mockUser);
    });

    it('throws ApiError when not authenticated', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 401,
        json: () => Promise.resolve({ detail: 'Not authenticated' }),
      });

      await expect(getMe()).rejects.toThrow(ApiError);
    });
  });
});

describe('API Keys functions', () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  describe('listApiKeys', () => {
    it('fetches list of API keys', async () => {
      const mockKeys = [
        { id: 'key1', key_prefix: 'knl_sk_xxx', name: 'Test Key', created_at: '2024-01-01', last_used_at: null, is_active: true },
      ];
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockKeys),
      });

      const result = await listApiKeys();

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/auth/keys'),
        expect.any(Object)
      );
      expect(result).toEqual(mockKeys);
    });
  });

  describe('createApiKey', () => {
    it('creates new API key with optional name', async () => {
      const mockResponse = {
        id: 'key1',
        key: 'knl_sk_full_key_here',
        key_prefix: 'knl_sk_xxx',
        name: 'My Key',
        created_at: '2024-01-01',
      };
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await createApiKey('My Key');

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/auth/keys'),
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({ name: 'My Key' }),
        })
      );
      expect(result).toEqual(mockResponse);
    });

    it('creates key with null name when not provided', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ id: 'key1', key: 'xxx', key_prefix: 'knl', name: '', created_at: '' }),
      });

      await createApiKey();

      expect(mockFetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          body: JSON.stringify({ name: null }),
        })
      );
    });
  });

  describe('revokeApiKey', () => {
    it('sends DELETE request for key', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(undefined),
      });

      await revokeApiKey('key-id-123');

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/auth/keys/key-id-123'),
        expect.objectContaining({
          method: 'DELETE',
        })
      );
    });
  });
});
