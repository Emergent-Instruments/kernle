'use client';

import { useState, useEffect, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { ApiKey, CreateKeyResponse, listApiKeys, createApiKey, revokeApiKey, cycleApiKey, ApiError } from '@/lib/api';

export default function ApiKeysPage() {
  const [keys, setKeys] = useState<ApiKey[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');
  const [newKey, setNewKey] = useState<CreateKeyResponse | null>(null);
  const [keyName, setKeyName] = useState('');
  const [isCreateOpen, setIsCreateOpen] = useState(false);
  const [copied, setCopied] = useState(false);

  const fetchKeys = useCallback(async () => {
    try {
      const data = await listApiKeys();
      setKeys(data);
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError('Failed to load API keys');
      }
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchKeys();
  }, [fetchKeys]);

  const handleCreate = async () => {
    try {
      const response = await createApiKey(keyName || undefined);
      setNewKey(response);
      setKeyName('');
      setIsCreateOpen(false);
      fetchKeys();
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError('Failed to create API key');
      }
    }
  };

  const handleRevoke = async (keyId: string) => {
    if (!confirm('Are you sure you want to revoke this key? This cannot be undone.')) {
      return;
    }

    try {
      await revokeApiKey(keyId);
      fetchKeys();
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError('Failed to revoke API key');
      }
    }
  };

  const handleCycle = async (keyId: string) => {
    if (!confirm('This will revoke the current key and create a new one. Continue?')) {
      return;
    }

    try {
      const response = await cycleApiKey(keyId);
      setNewKey(response);
      fetchKeys();
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError('Failed to cycle API key');
      }
    }
  };

  const copyToClipboard = async (text: string) => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const activeKeys = keys.filter((k) => k.is_active);
  const revokedKeys = keys.filter((k) => !k.is_active);

  return (
    <div className="space-y-8">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-3xl font-bold">API Keys</h2>
          <p className="text-muted-foreground">Manage your Kernle API keys</p>
        </div>
        <Dialog open={isCreateOpen} onOpenChange={setIsCreateOpen}>
          <DialogTrigger asChild>
            <Button>Create New Key</Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Create API Key</DialogTitle>
              <DialogDescription>
                Create a new API key for accessing the Kernle API.
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label htmlFor="keyName">Key Name (optional)</Label>
                <Input
                  id="keyName"
                  placeholder="e.g., Production, Development"
                  value={keyName}
                  onChange={(e) => setKeyName(e.target.value)}
                />
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setIsCreateOpen(false)}>
                Cancel
              </Button>
              <Button onClick={handleCreate}>Create Key</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {newKey && (
        <Alert className="border-green-500 bg-green-50 dark:bg-green-950">
          <AlertTitle className="text-green-700 dark:text-green-300">
            🔑 New API Key Created
          </AlertTitle>
          <AlertDescription className="space-y-4">
            <p className="text-green-700 dark:text-green-300">
              <strong>Copy this key now!</strong> It will only be shown once.
            </p>
            <div className="flex items-center space-x-2">
              <code className="flex-1 p-3 bg-white dark:bg-black rounded border font-mono text-sm break-all">
                {newKey.key}
              </code>
              <Button
                variant="outline"
                onClick={() => copyToClipboard(newKey.key)}
              >
                {copied ? 'Copied!' : 'Copy'}
              </Button>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setNewKey(null)}
              className="text-green-700 dark:text-green-300"
            >
              I&apos;ve saved this key
            </Button>
          </AlertDescription>
        </Alert>
      )}

      <Card>
        <CardHeader>
          <CardTitle>Active Keys</CardTitle>
          <CardDescription>
            Keys that are currently active and can be used for API access
          </CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <p>Loading...</p>
          ) : activeKeys.length === 0 ? (
            <p className="text-muted-foreground">No active API keys</p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Name</TableHead>
                  <TableHead>Prefix</TableHead>
                  <TableHead>Created</TableHead>
                  <TableHead>Last Used</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {activeKeys.map((key) => (
                  <TableRow key={key.id}>
                    <TableCell>
                      {key.name || <span className="text-muted-foreground">Unnamed</span>}
                    </TableCell>
                    <TableCell>
                      <code className="text-sm">{key.key_prefix}...</code>
                    </TableCell>
                    <TableCell>
                      {new Date(key.created_at).toLocaleDateString()}
                    </TableCell>
                    <TableCell>
                      {key.last_used_at
                        ? new Date(key.last_used_at).toLocaleDateString()
                        : 'Never'}
                    </TableCell>
                    <TableCell className="text-right space-x-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleCycle(key.id)}
                      >
                        Cycle
                      </Button>
                      <Button
                        variant="destructive"
                        size="sm"
                        onClick={() => handleRevoke(key.id)}
                      >
                        Revoke
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      {revokedKeys.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Revoked Keys</CardTitle>
            <CardDescription>Keys that have been revoked</CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Name</TableHead>
                  <TableHead>Prefix</TableHead>
                  <TableHead>Created</TableHead>
                  <TableHead>Revoked</TableHead>
                  <TableHead>Status</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {revokedKeys.map((key) => (
                  <TableRow key={key.id}>
                    <TableCell>
                      {key.name || <span className="text-muted-foreground">Unnamed</span>}
                    </TableCell>
                    <TableCell>
                      <code className="text-sm">{key.key_prefix}...</code>
                    </TableCell>
                    <TableCell>
                      {new Date(key.created_at).toLocaleDateString()}
                    </TableCell>
                    <TableCell>
                      <span className="text-muted-foreground">—</span>
                    </TableCell>
                    <TableCell>
                      <Badge variant="destructive">Revoked</Badge>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
