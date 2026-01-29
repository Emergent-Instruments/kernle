# Security Audit Report - Kernle
**Date:** 2026-01-28  
**Auditor:** Adversarial Security Subagent  
**Scope:** Full codebase review of ~/kernle  

---

## Executive Summary

This adversarial security audit identified **2 CRITICAL**, **3 HIGH**, **4 MEDIUM**, and **3 LOW** severity findings. The most urgent issues involve exposed production credentials and SQL injection vulnerabilities in the backend API.

---

## CRITICAL Findings

### [CRITICAL-001] Production Credentials Exposed in `.env` File
**File:** `~/kernle/.env`  
**Lines:** 1-22  

**Description:**  
The `.env` file contains actual production credentials for Supabase, including:
- Supabase service role key (full admin access)
- Database password 
- JWT secret key
- Direct database URL with embedded credentials

**Proof of Concept:**
```
SUPABASE_SERVICE_ROLE_KEY=eyJhbGciOiJIUzI1NiIs...  # REAL KEY EXPOSED
SUPABASE_DB_PASSWORD=Uo1JDY!^pMqC%BsB             # REAL PASSWORD
DATABASE_URL=postgresql://postgres:Uo1JDY!^pMqC%BsB@db.lbtjwflskpgmaijxreei.supabase.co:5432/postgres
JWT_SECRET_KEY=Nq1wmxleCN7s61eR/NL3akUbo7QUXDmyXLsEtYwwOWQ=
```

**Impact:**  
- Anyone with repo access can access production database
- Can forge JWT tokens with JWT_SECRET_KEY
- Full admin access to Supabase via service role key
- Can read/modify/delete all user data

**Recommended Fix:**
1. **IMMEDIATELY** rotate ALL credentials (Supabase keys, DB password, JWT secret)
2. Ensure `.env` is in `.gitignore` (it is, but verify no previous commits contain it)
3. Run: `git log --all --full-history -- .env` to check if ever committed
4. Use secret management (Vault, AWS Secrets Manager, or Supabase Vault)
5. Consider using environment variable injection at deployment time

---

### [CRITICAL-002] SQL Injection via `ilike` in Memory Search
**File:** `backend/app/routes/memories.py`  
**Lines:** 52-53  

**Description:**  
The search endpoint passes user input directly to Supabase's `ilike` function without sanitization. While Supabase/PostgREST provides some protection, special characters in the LIKE pattern (`%`, `_`) can be exploited.

**Vulnerable Code:**
```python
# Line 53 - User input directly interpolated into LIKE pattern
query = query.ilike(content_fields[0], f"%{request.query}%")
```

**Attack Vector:**
```bash
# Information disclosure via pattern matching
curl -X POST /memories/search \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"query": "%", "limit": 1000}'  # Match all records

# Timing attack / denial of service
curl -X POST /memories/search \
  -d '{"query": "%%%%%%%%%%%%%%%%%%%%%%%%a"}'  # Expensive pattern
```

**Impact:**
- Information disclosure (enumerate all memories)
- Denial of service via expensive LIKE patterns
- Potential for escaping the LIKE context depending on PostgREST version

**Recommended Fix:**
```python
# Escape LIKE special characters
import re
def escape_like(query: str) -> str:
    """Escape SQL LIKE special characters."""
    return re.sub(r'([%_\\])', r'\\\1', query)

# In search_memories:
safe_query = escape_like(request.query)
query = query.ilike(content_fields[0], f"%{safe_query}%")
```

---

## HIGH Findings

### [HIGH-001] Memory Search Table Name Validation Missing in Backend
**File:** `backend/app/routes/memories.py`  
**Lines:** 29-34  

**Description:**  
The `memory_types` parameter from user input is only checked against `MEMORY_TABLES` keys, but the validation could be bypassed if future code changes add tables without security review.

**Vulnerable Code:**
```python
tables_to_search = (
    [t for t in request.memory_types if t in MEMORY_TABLES]
    if request.memory_types
    else list(MEMORY_TABLES.keys())
)
```

**Impact:**  
If a developer adds a new table to `MEMORY_TABLES` dict (like `"users"` or `"api_keys"`), it becomes searchable without explicit security review.

**Recommended Fix:**
```python
# Create explicit allowlist for searchable tables
SEARCHABLE_TABLES = frozenset({"episodes", "beliefs", "notes", "values", "goals"})

# Validate against explicit allowlist
tables_to_search = [t for t in request.memory_types if t in SEARCHABLE_TABLES]
```

---

### [HIGH-002] No CSRF Protection on State-Changing Endpoints
**File:** `backend/app/main.py` and all route files  

**Description:**  
The API relies solely on Bearer tokens for authentication. While appropriate for API-only access, if any web frontend makes authenticated requests (CORS is configured for web origins), CSRF attacks become possible.

**Attack Vector:**
```html
<!-- Malicious page that triggers sync push with attacker data -->
<script>
fetch('https://api.kernle.ai/sync/push', {
  method: 'POST',
  credentials: 'include',  // If cookies used
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({operations: [/* malicious data */]})
});
</script>
```

**Impact:**
- Attacker can modify victim's memories if session cookies are used
- Currently mitigated by Bearer token auth, but fragile

**Recommended Fix:**
1. Ensure Bearer tokens are ONLY sent in Authorization header (not cookies)
2. Add `SameSite=Strict` if any cookies are used
3. Consider CSRF tokens for web dashboard endpoints

---

### [HIGH-003] JWT Token Algorithm Not Pinned
**File:** `backend/app/auth.py`  
**Lines:** 101-107  

**Description:**  
The JWT algorithm is read from settings (`settings.jwt_algorithm`), which comes from environment. If an attacker can modify environment variables, they could potentially set algorithm to `none`.

**Vulnerable Code:**
```python
payload = jwt.decode(
    token,
    settings.jwt_secret_key,
    algorithms=[settings.jwt_algorithm],  # From env var
)
```

**Impact:**
- Algorithm confusion attacks if env is compromised
- Could allow forged tokens with `alg: none`

**Recommended Fix:**
```python
# Hardcode allowed algorithms
ALLOWED_ALGORITHMS = ["HS256"]  # Only what you actually use

payload = jwt.decode(
    token,
    settings.jwt_secret_key,
    algorithms=ALLOWED_ALGORITHMS,
)
```

---

## MEDIUM Findings

### [MEDIUM-001] Rate Limiting Bypassable via IP Rotation
**File:** `backend/app/rate_limit.py`  
**Lines:** 1-7  

**Description:**  
Rate limiting is based solely on IP address, which can be trivially bypassed using proxies, VPNs, or cloud functions.

**Code:**
```python
limiter = Limiter(key_func=get_remote_address)
```

**Impact:**
- Brute force attacks on auth endpoints
- Resource exhaustion via repeated sync/search requests

**Recommended Fix:**
```python
# Combine IP + user identification
def get_rate_limit_key(request):
    ip = get_remote_address(request)
    # If authenticated, also include user_id
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token_prefix = auth[7:20]  # First chars for grouping
        return f"{ip}:{token_prefix}"
    return ip

limiter = Limiter(key_func=get_rate_limit_key)
```

---

### [MEDIUM-002] CORS Configuration Allows Localhost
**File:** `backend/app/config.py`  
**Lines:** 34-40  

**Description:**  
CORS origins include localhost, which is appropriate for development but should be environment-dependent.

**Code:**
```python
cors_origins: list[str] = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",  # Dev origins in production config
    "https://kernle.ai",
    "https://www.kernle.ai",
]
```

**Impact:**
- In production, allows any localhost app to make authenticated requests
- Could be exploited if user has malicious local software

**Recommended Fix:**
```python
# Environment-specific CORS
import os
if os.environ.get("ENVIRONMENT") == "production":
    cors_origins = ["https://kernle.ai", "https://www.kernle.ai"]
else:
    cors_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
```

---

### [MEDIUM-003] No Request Body Size Limits
**File:** `backend/app/main.py`  

**Description:**  
No visible limits on request body size. An attacker could send extremely large sync payloads to exhaust memory.

**Attack Vector:**
```bash
# Generate 100MB payload
python -c "print('{\"operations\":[' + ','.join(['{\"operation\":\"insert\",\"table\":\"notes\",\"record_id\":\"'+str(i)+'\",\"data\":{\"content\":\"A\"*10000}}' for i in range(10000)]) + ']}')" | curl -X POST /sync/push -d @-
```

**Impact:**
- Memory exhaustion
- Denial of service

**Recommended Fix:**
```python
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

class LimitBodySizeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.headers.get("content-length"):
            if int(request.headers["content-length"]) > 10_000_000:  # 10MB
                return Response("Request too large", status_code=413)
        return await call_next(request)

app.add_middleware(LimitBodySizeMiddleware)
```

---

### [MEDIUM-004] Pydantic Model Allows Extra Fields
**File:** `backend/app/models.py`  

**Description:**  
Pydantic models don't explicitly forbid extra fields, potentially allowing mass assignment attacks.

**Attack Vector:**
```bash
curl -X POST /auth/register -d '{
  "agent_id": "attacker",
  "is_admin": true,
  "user_id": "usr_admin123"
}'
```

**Impact:**
- Could potentially set internal fields if backend doesn't filter

**Recommended Fix:**
```python
class AgentRegister(BaseModel):
    class Config:
        extra = "forbid"  # Reject unknown fields
```

---

## LOW Findings

### [LOW-001] Health Endpoint Doesn't Verify DB Connectivity
**File:** `backend/app/main.py`  
**Lines:** 61-66  

**Description:**  
Health endpoint returns `"database": "connected"` without actually checking.

**Code:**
```python
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "database": "connected",  # TODO: Add actual DB check
    }
```

**Impact:**
- Load balancers may route traffic to unhealthy instances
- Debugging issues becomes harder

**Recommended Fix:**
```python
@app.get("/health")
async def health(db: Database):
    try:
        # Simple query to verify connection
        db.table("agents").select("id").limit(1).execute()
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return Response(
            content='{"status":"unhealthy","database":"disconnected"}',
            status_code=503,
            media_type="application/json"
        )
```

---

### [LOW-002] Verbose Error Messages in Some Paths
**File:** `backend/app/routes/sync.py`  
**Lines:** 72-75  

**Description:**  
While generic error messages are returned to clients, the full error is logged, which could leak info if logs are exposed.

**Code:**
```python
except Exception as e:
    # Log full error server-side for debugging
    logger.error(f"Database error during {op.operation} on {op.table}/{op.record_id}: {e}")
```

**Impact:**
- If logs are exposed (misconfigured logging, error reporting), internal details leak

**Recommended Fix:**
- Ensure logs are properly secured
- Consider structured logging without sensitive data in message strings

---

### [LOW-003] Local SQLite File Permissions After Creation
**File:** `kernle/storage/sqlite.py`  
**Lines:** 858-864  

**Description:**  
File permissions are set after database creation, creating a brief window where the file has default permissions.

**Code:**
```python
# Initialize database
self._init_db()

# Set secure file permissions (owner read/write only)
try:
    os.chmod(self.db_path, 0o600)
```

**Impact:**
- Brief race condition window for permission issues
- Minimal real-world risk

**Recommended Fix:**
- Set umask before file creation
- Or use atomic file creation with proper permissions

---

## Positive Findings (What's Done Right)

1. **Password hashing:** Using bcrypt with proper salt generation
2. **Table name validation:** SQLite storage has `ALLOWED_TABLES` and `validate_table_name()`
3. **Path traversal protection:** Database and checkpoint paths are validated against safe directories
4. **JWT verification:** Proper token verification with expiry
5. **Input validation:** Pydantic models with field constraints
6. **Recent security fixes:** Commit `1e101ab` addressed several issues

---

## Remediation Priority

| Priority | Finding | Effort | Impact |
|----------|---------|--------|--------|
| 🔴 P0 | CRITICAL-001: Credential rotation | Low | Critical |
| 🔴 P0 | CRITICAL-002: SQL injection fix | Low | Critical |
| 🟠 P1 | HIGH-003: Pin JWT algorithm | Low | High |
| 🟠 P1 | HIGH-001: Search table allowlist | Low | High |
| 🟡 P2 | MEDIUM-003: Body size limits | Medium | Medium |
| 🟡 P2 | MEDIUM-001: Rate limit improvement | Medium | Medium |
| 🟢 P3 | Others | Low-Medium | Low-Medium |

---

## Verification Commands

After fixes, run these to verify:

```bash
# Check if .env was ever committed
git log --all --full-history -- .env

# Test SQL injection fix
curl -X POST /memories/search -d '{"query": "%_____%"}'  # Should be escaped

# Verify rate limiting
for i in {1..100}; do curl -s -o /dev/null -w "%{http_code}\n" /auth/token; done

# Check body size limits
dd if=/dev/zero bs=1M count=20 | curl -X POST /sync/push -d @-  # Should 413
```

---

**Report Generated:** 2026-01-28T12:56:00-08:00  
**Next Audit Recommended:** After all CRITICAL/HIGH items are resolved
