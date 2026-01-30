import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, act } from '@testing-library/react';
import { AuthProvider, useAuth } from '../../lib/auth';
import * as api from '../../lib/api';

// Mock the API module
vi.mock('../../lib/api', () => ({
  getMe: vi.fn(),
}));

const mockGetMe = vi.mocked(api.getMe);

// Test component that uses useAuth
function TestConsumer() {
  const { user, isLoading, logout } = useAuth();
  return (
    <div>
      <span data-testid="loading">{isLoading ? 'loading' : 'ready'}</span>
      <span data-testid="user">{user ? user.email : 'no user'}</span>
      <button onClick={logout}>Logout</button>
    </div>
  );
}

describe('AuthProvider', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    global.fetch = vi.fn();
  });

  it('starts in loading state', () => {
    mockGetMe.mockImplementation(() => new Promise(() => {})); // Never resolves

    render(
      <AuthProvider>
        <TestConsumer />
      </AuthProvider>
    );

    expect(screen.getByTestId('loading')).toHaveTextContent('loading');
  });

  it('loads user on mount', async () => {
    const mockUser = { user_id: 'usr_123', email: 'test@example.com', created_at: '2024-01-01' };
    mockGetMe.mockResolvedValueOnce(mockUser);

    render(
      <AuthProvider>
        <TestConsumer />
      </AuthProvider>
    );

    await waitFor(() => {
      expect(screen.getByTestId('loading')).toHaveTextContent('ready');
    });

    expect(screen.getByTestId('user')).toHaveTextContent('test@example.com');
  });

  it('sets user to null on getMe error', async () => {
    mockGetMe.mockRejectedValueOnce(new Error('Not authenticated'));

    render(
      <AuthProvider>
        <TestConsumer />
      </AuthProvider>
    );

    await waitFor(() => {
      expect(screen.getByTestId('loading')).toHaveTextContent('ready');
    });

    expect(screen.getByTestId('user')).toHaveTextContent('no user');
  });

  it('logout clears user and calls API', async () => {
    const mockUser = { user_id: 'usr_123', email: 'test@example.com', created_at: '2024-01-01' };
    mockGetMe.mockResolvedValueOnce(mockUser);
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({ ok: true });

    render(
      <AuthProvider>
        <TestConsumer />
      </AuthProvider>
    );

    // Wait for initial load
    await waitFor(() => {
      expect(screen.getByTestId('user')).toHaveTextContent('test@example.com');
    });

    // Click logout
    await act(async () => {
      screen.getByText('Logout').click();
    });

    // User should be cleared
    expect(screen.getByTestId('user')).toHaveTextContent('no user');

    // Logout API should have been called
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/auth/logout'),
      expect.objectContaining({ method: 'POST', credentials: 'include' })
    );
  });
});

describe('useAuth', () => {
  it('throws when used outside AuthProvider', () => {
    // Suppress console.error for this test
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    expect(() => {
      render(<TestConsumer />);
    }).toThrow('useAuth must be used within an AuthProvider');

    consoleSpy.mockRestore();
  });
});
