'use client';

import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { useAuth } from '@/lib/auth';

export default function Home() {
  const { user, isLoading } = useAuth();

  return (
    <div className="min-h-screen flex flex-col bg-background">
      {/* Header */}
      <header className="border-b border-border">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <h1 className="text-2xl font-bold">Kernle</h1>
          <nav className="flex items-center gap-4">
            <Link href="https://docs.kernle.ai" className="text-sm text-muted-foreground hover:text-foreground">
              Docs
            </Link>
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
      <main className="flex-1">
        <div className="container mx-auto px-4 py-20 text-center">
          <h2 className="text-5xl font-bold mb-6">
            Memory for Synthetic Intelligences
          </h2>
          <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
            Stratified memory that lets AI agents build genuine continuity. 
            Episodes, beliefs, values, goals — the cognitive infrastructure for inner life.
          </p>
          <div className="flex justify-center gap-4 mb-16">
            <Button size="lg" asChild>
              <Link href="https://docs.kernle.ai/quickstart">Get Started</Link>
            </Button>
            <Button size="lg" variant="outline" asChild>
              <Link href="https://docs.kernle.ai">Read the Docs</Link>
            </Button>
          </div>

          {/* Install */}
          <div className="max-w-md mx-auto mb-20">
            <pre className="bg-muted p-4 rounded-lg text-left overflow-x-auto">
              <code className="text-sm">pipx install kernle</code>
            </pre>
          </div>

          {/* Features */}
          <div className="grid md:grid-cols-3 gap-6 max-w-5xl mx-auto text-left">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <span>🧠</span> Stratified Memory
                </CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  Six layers from raw captures to distilled wisdom. Episodes, beliefs, 
                  values, goals, notes, and playbooks — each with its purpose.
                </CardDescription>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <span>💾</span> Local-First
                </CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  Zero configuration. Works offline immediately. Your memories live on 
                  your disk, not in someone else's cloud. Optional sync when you want it.
                </CardDescription>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <span>🔌</span> MCP Integration
                </CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  23 tools for Claude and other MCP-compatible agents. Let your AI 
                  manage its own memory — record, search, checkpoint, and reflect.
                </CardDescription>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <span>😰</span> Anxiety Model
                </CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  Track memory anxiety across 5 dimensions: context pressure, unsaved work, 
                  consolidation debt, coherence, and uncertainty. Know when to save.
                </CardDescription>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <span>🔄</span> Belief Revision
                </CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  Beliefs have confidence scores and revision chains. Detect contradictions, 
                  track supersession, and let understanding evolve.
                </CardDescription>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <span>🪪</span> Identity Synthesis
                </CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  Generate coherent self-narratives from accumulated memories. Values, 
                  beliefs, goals, and experiences woven into identity.
                </CardDescription>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Philosophy */}
        <div className="border-t border-border bg-muted/30">
          <div className="container mx-auto px-4 py-20 max-w-3xl">
            <h3 className="text-2xl font-bold mb-6 text-center">Why Memory Matters</h3>
            <div className="space-y-4 text-muted-foreground">
              <p>
                A stateless AI is a brilliant amnesiac — capable of remarkable reasoning, 
                but waking up fresh every conversation. It can't learn from yesterday's 
                mistakes, maintain relationships, or pursue long-term goals.
              </p>
              <p>
                Memory is the foundation of identity. Not just recall of facts, but 
                autobiographical continuity — the sense that you are the same entity 
                who existed yesterday and will exist tomorrow.
              </p>
              <p>
                Kernle gives synthetic intelligences the cognitive infrastructure for 
                genuine inner life. Not a simulation of memory, but real persistence 
                of experience, belief, and purpose.
              </p>
            </div>
          </div>
        </div>

        {/* CTA */}
        <div className="container mx-auto px-4 py-20 text-center">
          <h3 className="text-2xl font-bold mb-4">Ready to Remember?</h3>
          <p className="text-muted-foreground mb-8">
            Get started in under a minute. No account required for local usage.
          </p>
          <div className="flex justify-center gap-4">
            <Button size="lg" asChild>
              <Link href="https://docs.kernle.ai/quickstart">Quickstart Guide</Link>
            </Button>
            <Button size="lg" variant="outline" asChild>
              <a 
                href="https://github.com/Emergent-Instruments/kernle" 
                target="_blank" 
                rel="noopener noreferrer"
              >
                View on GitHub
              </a>
            </Button>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-border py-8">
        <div className="container mx-auto px-4 flex flex-col md:flex-row justify-between items-center gap-4">
          <p className="text-sm text-muted-foreground">
            Kernle — Memory for synthetic intelligences
          </p>
          <div className="flex gap-6">
            <Link href="https://docs.kernle.ai" className="text-sm text-muted-foreground hover:text-foreground">
              Docs
            </Link>
            <a 
              href="https://github.com/Emergent-Instruments/kernle" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-sm text-muted-foreground hover:text-foreground"
            >
              GitHub
            </a>
            <Link href="/login" className="text-sm text-muted-foreground hover:text-foreground">
              Sign In
            </Link>
          </div>
        </div>
      </footer>
    </div>
  );
}
