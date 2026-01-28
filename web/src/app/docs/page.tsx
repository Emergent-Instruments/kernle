'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';

export default function DocsPage() {
  const router = useRouter();
  
  useEffect(() => {
    // Redirect to external docs site
    window.location.href = 'https://docs.kernle.ai';
  }, []);

  return (
    <div className="min-h-screen bg-background flex items-center justify-center">
      <div className="text-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
        <p className="text-muted-foreground">Redirecting to docs...</p>
      </div>
    </div>
  );
}
