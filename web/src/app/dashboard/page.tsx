'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { useAuth } from '@/lib/auth';

export default function DashboardPage() {
  const { user } = useAuth();

  if (!user) return null;

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold">Dashboard</h2>
        <p className="text-muted-foreground">Welcome back!</p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Account Information</CardTitle>
            <CardDescription>Your account details</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <p className="text-sm font-medium text-muted-foreground">User ID</p>
              <p className="font-mono text-sm">{user.user_id}</p>
            </div>
            <div>
              <p className="text-sm font-medium text-muted-foreground">Email</p>
              <p>{user.email}</p>
            </div>
            <div>
              <p className="text-sm font-medium text-muted-foreground">Created</p>
              <p>{new Date(user.created_at).toLocaleString()}</p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Quick Links</CardTitle>
            <CardDescription>Common actions</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            <a
              href="/dashboard/keys"
              className="block p-3 rounded-lg border hover:bg-accent transition-colors"
            >
              <p className="font-medium">Manage API Keys</p>
              <p className="text-sm text-muted-foreground">
                Create, view, and revoke API keys
              </p>
            </a>
            <a
              href="https://github.com/seanbhart/kernle"
              target="_blank"
              rel="noopener noreferrer"
              className="block p-3 rounded-lg border hover:bg-accent transition-colors"
            >
              <p className="font-medium">Documentation</p>
              <p className="text-sm text-muted-foreground">
                Learn how to use the Kernle API
              </p>
            </a>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
