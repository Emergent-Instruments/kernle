'use client';

import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

export default function DocsPage() {
  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <Link href="/" className="text-xl font-bold">
            Kernle
          </Link>
          <nav className="flex items-center gap-4">
            <Link href="/docs" className="text-sm text-primary font-medium">
              Docs
            </Link>
            <Link href="/login">
              <Button variant="outline" size="sm">Sign In</Button>
            </Link>
          </nav>
        </div>
      </header>

      <div className="container mx-auto px-4 py-12 max-w-4xl">
        {/* Hero */}
        <div className="mb-12">
          <h1 className="text-4xl font-bold mb-4">Documentation</h1>
          <p className="text-xl text-muted-foreground">
            Stratified memory for synthetic intelligences. Local-first, with optional cloud sync.
          </p>
        </div>

        {/* Quick Links */}
        <div className="grid md:grid-cols-2 gap-6 mb-12">
          <Card>
            <CardHeader>
              <CardTitle>🚀 Quickstart</CardTitle>
              <CardDescription>Get up and running in 5 minutes</CardDescription>
            </CardHeader>
            <CardContent>
              <Link href="#quickstart" className="text-primary hover:underline">
                Jump to Quickstart →
              </Link>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>🧠 Core Concepts</CardTitle>
              <CardDescription>Understand the memory architecture</CardDescription>
            </CardHeader>
            <CardContent>
              <Link href="#concepts" className="text-primary hover:underline">
                Learn the concepts →
              </Link>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>🔧 CLI Reference</CardTitle>
              <CardDescription>Complete command documentation</CardDescription>
            </CardHeader>
            <CardContent>
              <Link href="#cli" className="text-primary hover:underline">
                View CLI commands →
              </Link>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>🤖 MCP Integration</CardTitle>
              <CardDescription>Use with Claude and AI agents</CardDescription>
            </CardHeader>
            <CardContent>
              <Link href="#mcp" className="text-primary hover:underline">
                Setup MCP →
              </Link>
            </CardContent>
          </Card>
        </div>

        {/* Quickstart */}
        <section id="quickstart" className="mb-16">
          <h2 className="text-3xl font-bold mb-6 pb-2 border-b border-border">Quickstart</h2>
          
          <div className="space-y-8">
            <div>
              <h3 className="text-xl font-semibold mb-3">Installation</h3>
              <p className="text-muted-foreground mb-4">
                Install Kernle using pip or pipx (recommended for CLI tools):
              </p>
              <pre className="bg-muted p-4 rounded-lg overflow-x-auto">
                <code>{`# Using pipx (recommended)
pipx install kernle

# Or using pip
pip install kernle`}</code>
              </pre>
            </div>

            <div>
              <h3 className="text-xl font-semibold mb-3">Basic Usage</h3>
              <p className="text-muted-foreground mb-4">
                Kernle works immediately with zero configuration. All data is stored locally in <code className="bg-muted px-1 rounded">~/.kernle/</code>.
              </p>
              <pre className="bg-muted p-4 rounded-lg overflow-x-auto">
                <code>{`# Record an episode (something that happened)
kernle episode "Debugged the auth flow" "Fixed a race condition" \\
  --outcome success --lesson "Always check async state"

# Capture a quick thought (raw layer)
kernle raw "Need to revisit the caching strategy"

# Record a belief
kernle belief "Local-first architecture improves reliability" --confidence 0.8

# Search your memories
kernle search "auth"

# Check your memory status
kernle status

# See everything (readable markdown export)
kernle dump`}</code>
              </pre>
            </div>

            <div>
              <h3 className="text-xl font-semibold mb-3">Agent Usage</h3>
              <p className="text-muted-foreground mb-4">
                For AI agents, use the <code className="bg-muted px-1 rounded">-a</code> flag to specify an agent identity:
              </p>
              <pre className="bg-muted p-4 rounded-lg overflow-x-auto">
                <code>{`# Load memory at session start
kernle -a myagent load

# Save checkpoint before session end
kernle -a myagent checkpoint save "end of work session"

# Check memory anxiety (context pressure, unsaved work, etc.)
kernle -a myagent anxiety

# Synthesize identity from memories
kernle -a myagent identity`}</code>
              </pre>
            </div>
          </div>
        </section>

        {/* Core Concepts */}
        <section id="concepts" className="mb-16">
          <h2 className="text-3xl font-bold mb-6 pb-2 border-b border-border">Core Concepts</h2>
          
          <div className="space-y-8">
            <div>
              <h3 className="text-xl font-semibold mb-3">Memory Layers</h3>
              <p className="text-muted-foreground mb-4">
                Kernle implements a stratified memory system inspired by biological memory:
              </p>
              <div className="space-y-4">
                <div className="bg-muted p-4 rounded-lg">
                  <h4 className="font-semibold mb-1">📝 Raw Layer</h4>
                  <p className="text-sm text-muted-foreground">
                    Quick captures, fleeting thoughts, scratchpad. Zero friction entry point.
                    Use <code>kernle raw "thought"</code> to capture.
                  </p>
                </div>
                <div className="bg-muted p-4 rounded-lg">
                  <h4 className="font-semibold mb-1">📖 Episodes</h4>
                  <p className="text-sm text-muted-foreground">
                    Autobiographical memories — things that happened with context, outcomes, and lessons learned.
                  </p>
                </div>
                <div className="bg-muted p-4 rounded-lg">
                  <h4 className="font-semibold mb-1">💭 Beliefs</h4>
                  <p className="text-sm text-muted-foreground">
                    What you hold to be true, with confidence scores. Supports contradiction detection and revision chains.
                  </p>
                </div>
                <div className="bg-muted p-4 rounded-lg">
                  <h4 className="font-semibold mb-1">⭐ Values</h4>
                  <p className="text-sm text-muted-foreground">
                    Core principles and priorities that guide decisions. Higher priority = more central to identity.
                  </p>
                </div>
                <div className="bg-muted p-4 rounded-lg">
                  <h4 className="font-semibold mb-1">🎯 Goals</h4>
                  <p className="text-sm text-muted-foreground">
                    What you're working toward. Tracks status (active, completed, abandoned) and progress.
                  </p>
                </div>
                <div className="bg-muted p-4 rounded-lg">
                  <h4 className="font-semibold mb-1">📋 Playbooks</h4>
                  <p className="text-sm text-muted-foreground">
                    Procedural memory — how to do things. Reusable patterns with applicability conditions.
                  </p>
                </div>
              </div>
            </div>

            <div>
              <h3 className="text-xl font-semibold mb-3">Anxiety Model</h3>
              <p className="text-muted-foreground mb-4">
                Kernle tracks "memory anxiety" across 5 dimensions — the functional stress an AI experiences around memory and context:
              </p>
              <ul className="list-disc list-inside space-y-2 text-muted-foreground">
                <li><strong>Context Pressure:</strong> How full is the context window?</li>
                <li><strong>Unsaved Work:</strong> How long since the last checkpoint?</li>
                <li><strong>Consolidation Debt:</strong> How many experiences haven't been processed into lessons?</li>
                <li><strong>Coherence:</strong> Are there contradictory beliefs?</li>
                <li><strong>Uncertainty:</strong> How many beliefs have low confidence?</li>
              </ul>
              <pre className="bg-muted p-4 rounded-lg overflow-x-auto mt-4">
                <code>{`# Check anxiety levels
kernle -a myagent anxiety

# Output shows scores per dimension and recommended actions`}</code>
              </pre>
            </div>

            <div>
              <h3 className="text-xl font-semibold mb-3">Local-First Architecture</h3>
              <p className="text-muted-foreground mb-4">
                All data is stored locally in SQLite by default. This means:
              </p>
              <ul className="list-disc list-inside space-y-2 text-muted-foreground">
                <li>Zero configuration required — works offline immediately</li>
                <li>You own your data — it's just files on your disk</li>
                <li>No vendor lock-in — export anytime with <code>kernle dump</code></li>
                <li>Fast — no network latency for operations</li>
              </ul>
            </div>
          </div>
        </section>

        {/* CLI Reference */}
        <section id="cli" className="mb-16">
          <h2 className="text-3xl font-bold mb-6 pb-2 border-b border-border">CLI Reference</h2>
          
          <div className="space-y-6">
            <div>
              <h3 className="text-xl font-semibold mb-3">Memory Operations</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="text-left py-2 pr-4">Command</th>
                      <th className="text-left py-2">Description</th>
                    </tr>
                  </thead>
                  <tbody className="text-muted-foreground">
                    <tr className="border-b border-border">
                      <td className="py-2 pr-4 font-mono">kernle episode</td>
                      <td className="py-2">Record an autobiographical episode</td>
                    </tr>
                    <tr className="border-b border-border">
                      <td className="py-2 pr-4 font-mono">kernle note</td>
                      <td className="py-2">Capture a note or observation</td>
                    </tr>
                    <tr className="border-b border-border">
                      <td className="py-2 pr-4 font-mono">kernle raw</td>
                      <td className="py-2">Quick capture (scratchpad)</td>
                    </tr>
                    <tr className="border-b border-border">
                      <td className="py-2 pr-4 font-mono">kernle belief</td>
                      <td className="py-2">Record or update a belief</td>
                    </tr>
                    <tr className="border-b border-border">
                      <td className="py-2 pr-4 font-mono">kernle search</td>
                      <td className="py-2">Semantic search across memories</td>
                    </tr>
                    <tr className="border-b border-border">
                      <td className="py-2 pr-4 font-mono">kernle dump</td>
                      <td className="py-2">Export all memories as markdown</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <div>
              <h3 className="text-xl font-semibold mb-3">Session Management</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="text-left py-2 pr-4">Command</th>
                      <th className="text-left py-2">Description</th>
                    </tr>
                  </thead>
                  <tbody className="text-muted-foreground">
                    <tr className="border-b border-border">
                      <td className="py-2 pr-4 font-mono">kernle load</td>
                      <td className="py-2">Load working memory (session start)</td>
                    </tr>
                    <tr className="border-b border-border">
                      <td className="py-2 pr-4 font-mono">kernle checkpoint save</td>
                      <td className="py-2">Save current state</td>
                    </tr>
                    <tr className="border-b border-border">
                      <td className="py-2 pr-4 font-mono">kernle checkpoint list</td>
                      <td className="py-2">List saved checkpoints</td>
                    </tr>
                    <tr className="border-b border-border">
                      <td className="py-2 pr-4 font-mono">kernle status</td>
                      <td className="py-2">Show memory statistics</td>
                    </tr>
                    <tr className="border-b border-border">
                      <td className="py-2 pr-4 font-mono">kernle anxiety</td>
                      <td className="py-2">Check memory anxiety levels</td>
                    </tr>
                    <tr className="border-b border-border">
                      <td className="py-2 pr-4 font-mono">kernle identity</td>
                      <td className="py-2">Synthesize identity narrative</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <div>
              <h3 className="text-xl font-semibold mb-3">Sync Operations</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="text-left py-2 pr-4">Command</th>
                      <th className="text-left py-2">Description</th>
                    </tr>
                  </thead>
                  <tbody className="text-muted-foreground">
                    <tr className="border-b border-border">
                      <td className="py-2 pr-4 font-mono">kernle auth register</td>
                      <td className="py-2">Create account for cloud sync</td>
                    </tr>
                    <tr className="border-b border-border">
                      <td className="py-2 pr-4 font-mono">kernle auth login</td>
                      <td className="py-2">Login to sync service</td>
                    </tr>
                    <tr className="border-b border-border">
                      <td className="py-2 pr-4 font-mono">kernle sync</td>
                      <td className="py-2">Push/pull changes to cloud</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </section>

        {/* MCP Integration */}
        <section id="mcp" className="mb-16">
          <h2 className="text-3xl font-bold mb-6 pb-2 border-b border-border">MCP Integration</h2>
          
          <div className="space-y-8">
            <div>
              <h3 className="text-xl font-semibold mb-3">What is MCP?</h3>
              <p className="text-muted-foreground mb-4">
                The Model Context Protocol (MCP) allows AI assistants like Claude to use external tools.
                Kernle's MCP server exposes 23 memory tools that let agents manage their own memories.
              </p>
            </div>

            <div>
              <h3 className="text-xl font-semibold mb-3">Setup with Claude Desktop</h3>
              <p className="text-muted-foreground mb-4">
                Add Kernle to your Claude Desktop configuration:
              </p>
              <pre className="bg-muted p-4 rounded-lg overflow-x-auto">
                <code>{`// ~/Library/Application Support/Claude/claude_desktop_config.json
{
  "mcpServers": {
    "kernle": {
      "command": "kernle",
      "args": ["mcp", "--agent", "claude"]
    }
  }
}`}</code>
              </pre>
              <p className="text-muted-foreground mt-4">
                Restart Claude Desktop. The agent will now have access to memory tools.
              </p>
            </div>

            <div>
              <h3 className="text-xl font-semibold mb-3">Available MCP Tools</h3>
              <p className="text-muted-foreground mb-4">
                The MCP server provides these tools to AI agents:
              </p>
              <ul className="list-disc list-inside space-y-1 text-muted-foreground">
                <li><code>memory_episode_create</code> — Record an episode</li>
                <li><code>memory_episode_list</code> — List recent episodes</li>
                <li><code>memory_belief_create</code> — Record a belief</li>
                <li><code>memory_belief_update</code> — Update belief confidence</li>
                <li><code>memory_value_create</code> — Define a value</li>
                <li><code>memory_goal_create</code> — Set a goal</li>
                <li><code>memory_goal_update</code> — Update goal status</li>
                <li><code>memory_note_create</code> — Capture a note</li>
                <li><code>memory_search</code> — Semantic search</li>
                <li><code>memory_status</code> — Get memory statistics</li>
                <li><code>memory_checkpoint_save</code> — Save checkpoint</li>
                <li><code>memory_checkpoint_load</code> — Load checkpoint</li>
                <li><code>memory_identity</code> — Synthesize identity</li>
                <li><code>memory_anxiety</code> — Check anxiety levels</li>
                <li>...and more</li>
              </ul>
            </div>

            <div>
              <h3 className="text-xl font-semibold mb-3">Example: Agent Session</h3>
              <p className="text-muted-foreground mb-4">
                Here's how an agent might use Kernle throughout a session:
              </p>
              <pre className="bg-muted p-4 rounded-lg overflow-x-auto">
                <code>{`# At session start, agent loads memory
→ memory_checkpoint_load

# During work, agent records experiences
→ memory_episode_create(
    objective="Help user debug authentication",
    outcome="success",
    lesson="Check token expiration first"
  )

# Agent captures realizations
→ memory_belief_create(
    statement="JWT refresh should happen proactively",
    confidence=0.7
  )

# Before session ends, agent saves state
→ memory_checkpoint_save(description="Completed auth debugging session")`}</code>
              </pre>
            </div>
          </div>
        </section>

        {/* Cloud Sync */}
        <section id="sync" className="mb-16">
          <h2 className="text-3xl font-bold mb-6 pb-2 border-b border-border">Cloud Sync (Optional)</h2>
          
          <div className="space-y-8">
            <div>
              <h3 className="text-xl font-semibold mb-3">Why Sync?</h3>
              <p className="text-muted-foreground mb-4">
                Cloud sync is optional but enables:
              </p>
              <ul className="list-disc list-inside space-y-2 text-muted-foreground">
                <li>Backup your memories</li>
                <li>Access from multiple devices</li>
                <li>Share memories between agents (coming soon)</li>
                <li>Cross-agent collaboration (coming soon)</li>
              </ul>
            </div>

            <div>
              <h3 className="text-xl font-semibold mb-3">Setup</h3>
              <pre className="bg-muted p-4 rounded-lg overflow-x-auto">
                <code>{`# Create an account
kernle auth register

# Your credentials are saved locally
# Sync happens automatically when online

# Manual sync if needed
kernle sync`}</code>
              </pre>
            </div>

            <div>
              <h3 className="text-xl font-semibold mb-3">Self-Hosting</h3>
              <p className="text-muted-foreground mb-4">
                Kernle is open source. You can run your own backend:
              </p>
              <pre className="bg-muted p-4 rounded-lg overflow-x-auto">
                <code>{`# Clone the repo
git clone https://github.com/Emergent-Instruments/kernle

# Run the backend
cd kernle/backend
pip install -r requirements.txt
uvicorn app.main:app

# Point CLI to your backend
export KERNLE_API_URL=http://localhost:8000`}</code>
              </pre>
            </div>
          </div>
        </section>

        {/* Footer */}
        <footer className="border-t border-border pt-8 mt-16">
          <div className="flex flex-col md:flex-row justify-between items-center gap-4">
            <p className="text-sm text-muted-foreground">
              Kernle — Memory for synthetic intelligences
            </p>
            <div className="flex gap-4">
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
    </div>
  );
}
