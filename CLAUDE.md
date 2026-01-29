# Development Guidelines

## Workflow Requirements

### Parallel Work & Delegation
- Use specialist subagents (Task tool) to delegate work for parallel execution
- Don't do everything sequentially when tasks can be parallelized
- Use the Explore agent for codebase discovery to preserve main context
- Delegate research and investigation to subagents when possible

### Task Management
- Use TaskCreate/TaskUpdate for any multi-step work (3+ steps)
- Break complex work into discrete, trackable tasks
- Update task status as work progresses (pending → in_progress → completed)
- Use task dependencies (blockedBy) when order matters

### Before Marking Work Complete
Always run audits using specialist agents before considering work done:

**Security (security-auditor agent):**
- Input validation and sanitization
- Authentication/authorization correctness
- SQL injection, XSS, command injection risks
- Secrets exposure (no hardcoded credentials, API keys)
- OWASP Top 10 vulnerabilities

**Test Quality:**
- Adequate test coverage for changes
- No tautological tests (tests that can't fail, assert True, mock returns what you assert)
- Tests actually exercise the code path, not just mocks
- Edge cases and error conditions covered
- Integration tests where appropriate

**Code Quality:**
- Follows existing codebase patterns and conventions
- No code duplication that should be abstracted
- Proper error handling with meaningful messages
- Type hints on public interfaces (Python) / TypeScript types
- No commented-out code or debug statements left behind

**Architecture (senior-developer agent for significant changes):**
- Changes follow established patterns (repository pattern, etc.)
- No circular dependencies introduced
- Database migrations are safe (reversible, no data loss)
- API changes are backwards compatible or properly versioned
- Performance implications considered (N+1 queries, unnecessary loops)

**Documentation:**
- Public APIs have clear docstrings/comments
- Breaking changes documented
- README updated if setup/usage changes

### Planning
- Use EnterPlanMode for non-trivial implementations
- Get user sign-off on approach before significant code changes
- Use AskUserQuestion when requirements are ambiguous

### Git Hygiene
- Atomic commits with clear messages
- Don't commit secrets, large binaries, or generated files
- Verify .gitignore covers sensitive/generated content

## Project Context

### Tech Stack
- **Backend**: Python, FastAPI, SQLAlchemy, Pydantic
- **Frontend**: TypeScript, Next.js, React
- **Testing**: pytest (Python), Jest/Vitest (TypeScript)
- **Database**: PostgreSQL with Alembic migrations

### Patterns in Use
- Repository pattern for data access
- Pydantic models for API request/response validation
- Service layer for business logic

## Context Management

- Proactively summarize findings before context grows large
- Delegate exploration to subagents to preserve main context
- When approaching complex tasks, consider what can be parallelized
- Use background tasks (run_in_background) for long-running operations

## Known Issues - DO NOT "FIX"

These are intentional design decisions that may look like bugs or security issues. Do NOT change them without explicit user approval.

### Supabase OAuth: Use JWKS Public Key Verification (backend/app/routes/auth.py)

**DO NOT** change the OAuth token exchange to use Supabase API calls with API keys.

**Background:** The OAuth verification uses JWKS (JSON Web Key Set) public key verification:
1. Fetches public keys from `{issuer}/.well-known/jwks.json` (no API key needed)
2. Verifies the JWT using the matching public key
3. Uses whatever algorithm the token specifies (RS256, ES256, etc.)

**This approach is correct because:**
- No API keys required (avoids "Legacy API keys disabled" errors)
- Works with any algorithm Supabase uses
- Public JWKS endpoint is always available

**DO NOT change to use Supabase API calls:**
```python
# WRONG - requires API keys that can be disabled/rotated
auth_url = f"{settings.supabase_url}/auth/v1/user"
headers = {"Authorization": f"Bearer {token}", "apikey": api_key}  # API key can break!
```

**DO NOT add strict algorithm validation:**
```python
# WRONG - breaks when Supabase changes algorithms
if alg not in ("HS256", "RS256"):  # ES256 will fail!
    raise "Unsupported algorithm"
```

**The correct approach (current implementation):**
```python
# Fetch JWKS (no API key needed)
jwks_url = f"{issuer}/.well-known/jwks.json"
jwks_response = await client.get(jwks_url)
# Verify with whatever algorithm the token uses
payload = jwt.decode(token, public_key, algorithms=[alg])
```

This was broken on 2026-01-28 when Supabase disabled legacy API keys, proving that the API-based approach is fragile.
