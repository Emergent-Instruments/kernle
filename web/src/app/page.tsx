'use client';

import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { useAuth } from '@/lib/auth';

export default function Home() {
  const { user, isLoading } = useAuth();

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="border-b">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <h1 className="text-2xl font-bold">Kernle</h1>
          <nav className="space-x-4">
            {isLoading ? null : user ? (
              <Button asChild>
                <Link href="/dashboard">Dashboard</Link>
              </Button>
            ) : (
              <>
                <Button variant="ghost" asChild>
                  <Link href="/login">Login</Link>
                </Button>
                <Button asChild>
                  <Link href="/register">Sign Up</Link>
                </Button>
              </>
            )}
          </nav>
        </div>
      </header>

      {/* Hero */}
      <main className="flex-1 flex items-center justify-center">
        <div className="container mx-auto px-4 text-center">
          <h2 className="text-5xl font-bold mb-6">Memory for AI Agents</h2>
          <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
            Give your AI agents persistent memory. Store, retrieve, and manage 
            memories with a simple API. Wake up where you left off.
          </p>
          <div className="space-x-4">
            <Button size="lg" asChild>
              <Link href="/register">Get Started</Link>
            </Button>
            <Button size="lg" variant="outline" asChild>
              <a href="https://github.com/seanbhart/kernle" target="_blank" rel="noopener noreferrer">
                View Docs
              </a>
            </Button>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t py-8">
        <div className="container mx-auto px-4 text-center text-muted-foreground">
          <p>© {new Date().getFullYear()} Kernle. Built for AI agents.</p>
        </div>
      </footer>
    </div>
  );
}
