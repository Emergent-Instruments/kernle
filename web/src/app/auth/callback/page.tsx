'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { supabase } from '@/lib/supabase';
import { useAuth } from '@/lib/auth';
import { exchangeOAuthToken } from '@/lib/api';

export default function AuthCallbackPage() {
  const router = useRouter();
  const { login } = useAuth();
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const handleCallback = async () => {
      try {
        const { data: { session }, error: supabaseError } = await supabase.auth.getSession();
        
        if (supabaseError) {
          console.error('Supabase auth error:', supabaseError);
          setError('Authentication failed');
          setTimeout(() => router.push('/login?error=auth_failed'), 2000);
          return;
        }
        
        if (!session?.access_token) {
          console.error('No session token');
          setError('No session found');
          setTimeout(() => router.push('/login?error=no_session'), 2000);
          return;
        }

        // Exchange Supabase token for Kernle token
        const tokenResponse = await exchangeOAuthToken(session.access_token);
        
        // Store Kernle token and redirect
        login(tokenResponse.access_token);
        router.push('/dashboard');
        
      } catch (err) {
        console.error('Token exchange error:', err);
        setError('Failed to complete sign in');
        setTimeout(() => router.push('/login?error=exchange_failed'), 2000);
      }
    };

    handleCallback();
  }, [router, login]);

  return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="text-center">
        {error ? (
          <>
            <p className="text-destructive mb-2">{error}</p>
            <p className="text-muted-foreground text-sm">Redirecting to login...</p>
          </>
        ) : (
          <>
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
            <p className="text-muted-foreground">Completing sign in...</p>
          </>
        )}
      </div>
    </div>
  );
}
