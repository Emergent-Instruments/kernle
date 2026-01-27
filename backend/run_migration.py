#!/usr/bin/env python3
"""Run Supabase migration using Supabase client."""

from supabase import create_client

# Supabase credentials
SUPABASE_URL = "https://lbtjwflskpgmaijxreei.supabase.co"
SUPABASE_SERVICE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImxidGp3Zmxza3BnbWFpanhyZWVpIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2OTU1MTA4MywiZXhwIjoyMDg1MTI3MDgzfQ.GWINkCDaYtteCUs3ecQTiKVO4dYjWb58I3hQsDC8NL0"

print("Connecting to Supabase...")
client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

print("Connected! Reading migration...")
with open('supabase/migrations/001_initial_schema.sql', 'r') as f:
    migration_sql = f.read()

# Split into individual statements
statements = []
current = []
for line in migration_sql.split('\n'):
    line = line.strip()
    if line.startswith('--') or not line:
        continue
    current.append(line)
    if line.endswith(';'):
        statements.append(' '.join(current))
        current = []

print(f"Found {len(statements)} statements")

# Try executing via RPC if a function exists, or we need to use the dashboard
print("\n⚠️  Supabase Python client doesn't support raw SQL execution.")
print("Please run the migration manually:")
print("1. Go to: https://supabase.com/dashboard/project/lbtjwflskpgmaijxreei/sql")
print("2. Paste the contents of: supabase/migrations/001_initial_schema.sql")
print("3. Click 'Run'")
print("\nAlternatively, let's try creating tables via the API...")

# Try creating the agents table via the API first to verify connectivity
try:
    # Check if agents table exists
    result = client.table('agents').select('*').limit(1).execute()
    print(f"\n✓ 'agents' table already exists! Data: {result.data}")
except Exception as e:
    if "does not exist" in str(e) or "relation" in str(e):
        print(f"\n✗ 'agents' table doesn't exist yet. Error: {e}")
        print("\nYou need to run the migration in the Supabase Dashboard.")
    else:
        print(f"\nConnection test result: {e}")
