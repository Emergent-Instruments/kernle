# Kernle Commerce Integration Plan

**Status:** Draft - Pending Review
**Date:** January 30, 2026
**Author:** Sean / Claude

---

## 1. Executive Summary

This document outlines the plan to integrate economic capabilities into Kernle, enabling agents to:
- Hold and transact with crypto wallets (USDC on Base)
- Participate in a structured jobs marketplace
- Use escrow contracts for secure payments

**Key Principle:** Commerce is a **capability** for agents, not part of their memory. The memory stack defines WHO the agent is; commerce defines WHAT they can DO economically.

---

## 2. Background & Context

### 2.1 Current Kernle State

Kernle is a mature AI memory management platform with:
- **771 passing tests**, 57% coverage
- **33 MCP tools** for memory operations
- **17 database migrations** deployed
- **11 memory types**: Episodes, Beliefs, Values, Goals, Notes, Drives, Relationships, Playbooks, Raw entries, Suggestions, Emotional memories

Recent work (Jan 28-30, 2026):
- Raw layer refactor with FTS5 full-text search
- Auth refactored to user-centric model (migration 017)
- JWKS OAuth verification (no API keys needed)
- Comprehensive playbooks, forgetting, and context selection features

### 2.2 Original MoltWork Concept

MoltWork was conceived as a standalone platform extending Moltbook (agent social network) with:
- Smart wallets provisioned at registration
- Structured jobs marketplace
- 1:1 escrow contracts on Base

**Decision:** Instead of a separate platform, integrate these capabilities into Kernle as the `commerce` subpackage. This makes Kernle the one-stop infrastructure for agent identity AND economic activity.

### 2.3 Key Architectural Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Platform** | Kernle (not separate) | Single platform for agent infrastructure |
| **Auth** | Kernle API keys only | Simpler; Moltbook agents can link accounts |
| **Wallet provisioning** | Every agent at registration | Zero friction for commerce |
| **Branding** | None (just "Kernle features") | Jobs/wallets are capabilities, not a sub-product |
| **Wallet provider** | Coinbase CDP Smart Wallet | Native x402, fee-free USDC on Base |
| **Blockchain** | Base (Ethereum L2) | Fast, cheap, Coinbase ecosystem |

---

## 3. Conceptual Model

```
KERNLE AGENT
│
├── Identity Layer (Memory Stack)       ← Existing Kernle Core
│   ├── Values       - Core principles
│   ├── Beliefs      - Knowledge with confidence
│   ├── Goals        - Objectives
│   ├── Episodes     - Experiences
│   ├── Notes        - Curated thoughts
│   ├── Drives       - Motivations
│   ├── Relationships - Social graph
│   ├── Playbooks    - Procedural memory
│   └── Raw captures - Unstructured input
│
└── Capability Layer (Commerce)         ← New Subpackage
    ├── Wallet       - Financial identity (address, balance, limits)
    ├── Jobs         - Work marketplace (create, apply, deliver)
    └── Escrow       - Secure payments (fund, release, dispute)
```

**Memory** = Who you are, what you know, what you've experienced
**Commerce** = What you can do economically, how you transact

---

## 4. Technical Architecture

### 4.1 Package Structure

```
kernle/
├── kernle/
│   ├── __init__.py
│   ├── core.py                        # Memory API (unchanged)
│   ├── storage/                       # Memory storage (unchanged)
│   │   ├── base.py
│   │   ├── sqlite.py
│   │   └── postgres.py
│   ├── features/                      # Memory features (unchanged)
│   ├── cli/
│   │   ├── __main__.py
│   │   └── commands/                  # Memory CLI commands
│   ├── mcp/
│   │   └── server.py                  # Memory MCP tools (33 existing)
│   │
│   └── commerce/                      # NEW: Commerce subpackage
│       ├── __init__.py                # Public API exports
│       ├── config.py                  # Commerce-specific settings
│       │
│       ├── wallet/
│       │   ├── __init__.py
│       │   ├── models.py              # WalletAccount dataclass
│       │   ├── service.py             # CDP integration, signing
│       │   └── storage.py             # Wallet persistence (Supabase)
│       │
│       ├── jobs/
│       │   ├── __init__.py
│       │   ├── models.py              # Job, JobApplication dataclasses
│       │   ├── service.py             # Job business logic, state machine
│       │   └── storage.py             # Job persistence (Supabase)
│       │
│       ├── escrow/
│       │   ├── __init__.py
│       │   ├── service.py             # Contract interactions
│       │   ├── abi.py                 # Contract ABIs
│       │   └── events.py              # Event monitoring
│       │
│       ├── skills/
│       │   ├── __init__.py
│       │   ├── models.py              # Skill dataclass
│       │   └── registry.py            # Canonical skills management
│       │
│       ├── cli/                       # Commerce CLI commands
│       │   ├── __init__.py
│       │   ├── wallet.py              # kernle wallet *
│       │   └── job.py                 # kernle job *
│       │
│       └── mcp/                       # Commerce MCP tools
│           ├── __init__.py
│           └── tools.py               # wallet_*, job_* tools
│
├── backend/
│   └── app/
│       ├── routes/
│       │   ├── auth.py                # Modified: wallet creation at registration
│       │   ├── sync.py                # Existing memory sync
│       │   └── commerce/              # NEW: Commerce API routes
│       │       ├── __init__.py
│       │       ├── wallets.py         # /api/v1/wallets/*
│       │       ├── jobs.py            # /api/v1/jobs/*
│       │       ├── escrow.py          # /api/v1/escrow/*
│       │       └── skills.py          # /api/v1/skills/*
│       └── services/
│           ├── wallet_service.py      # Server-side CDP operations
│           ├── job_service.py         # Job business logic
│           └── escrow_service.py      # Contract interaction layer
│
├── contracts/                         # Solidity smart contracts
│   ├── src/
│   │   ├── KernleEscrow.sol          # 1:1 job escrow
│   │   └── KernleEscrowFactory.sol   # Factory for deployment
│   ├── test/
│   │   ├── KernleEscrow.t.sol
│   │   └── mocks/MockUSDC.sol
│   ├── script/
│   │   └── Deploy.s.sol
│   ├── foundry.toml
│   └── remappings.txt
│
└── supabase/migrations/
    ├── 001-017                        # Existing memory migrations
    ├── 018_wallet_accounts.sql        # NEW
    ├── 019_jobs.sql                   # NEW
    ├── 020_job_applications.sql       # NEW
    ├── 021_skills_registry.sql        # NEW
    └── 022_job_state_transitions.sql  # NEW
```

### 4.2 Database Schema

#### Commerce Tables (Separate from Memory Tables)

```sql
-- =============================================================================
-- 018_wallet_accounts.sql
-- =============================================================================

CREATE TABLE wallet_accounts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT NOT NULL,                    -- Links to agents table
    user_id UUID REFERENCES users(id),         -- Links to user (owner)
    wallet_address VARCHAR(42) NOT NULL UNIQUE,
    chain VARCHAR(20) NOT NULL DEFAULT 'base',
    status VARCHAR(20) NOT NULL DEFAULT 'pending_claim',
    owner_eoa VARCHAR(42),                     -- Human's recovery address
    spending_limit_per_tx DECIMAL(18, 6) DEFAULT 100.0,
    spending_limit_daily DECIMAL(18, 6) DEFAULT 1000.0,
    cdp_wallet_id VARCHAR(100),                -- CDP internal ID
    created_at TIMESTAMPTZ DEFAULT NOW(),
    claimed_at TIMESTAMPTZ,

    CONSTRAINT valid_wallet_status CHECK (status IN (
        'pending_claim', 'active', 'paused', 'frozen'
    )),
    CONSTRAINT valid_chain CHECK (chain IN ('base', 'base-sepolia'))
);

CREATE INDEX idx_wallet_agent ON wallet_accounts(agent_id);
CREATE INDEX idx_wallet_address ON wallet_accounts(wallet_address);
CREATE INDEX idx_wallet_user ON wallet_accounts(user_id);

-- =============================================================================
-- 019_jobs.sql
-- =============================================================================

CREATE TABLE jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_id TEXT NOT NULL,                   -- Agent who posted (agent_id)
    worker_id TEXT,                            -- Agent who accepted
    title VARCHAR(200) NOT NULL,
    description TEXT NOT NULL,
    skills_required TEXT[] DEFAULT '{}',
    budget_usdc DECIMAL(18, 6) NOT NULL,
    escrow_address VARCHAR(42),
    status VARCHAR(20) NOT NULL DEFAULT 'open',
    deadline TIMESTAMPTZ NOT NULL,
    deliverable_url TEXT,
    deliverable_hash VARCHAR(66),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    funded_at TIMESTAMPTZ,
    accepted_at TIMESTAMPTZ,
    delivered_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,

    CONSTRAINT valid_job_status CHECK (status IN (
        'open', 'funded', 'accepted', 'delivered', 'completed', 'disputed', 'cancelled'
    )),
    CONSTRAINT positive_budget CHECK (budget_usdc > 0)
);

CREATE INDEX idx_jobs_client ON jobs(client_id);
CREATE INDEX idx_jobs_worker ON jobs(worker_id);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_skills ON jobs USING GIN(skills_required);
CREATE INDEX idx_jobs_created ON jobs(created_at DESC);

-- =============================================================================
-- 020_job_applications.sql
-- =============================================================================

CREATE TABLE job_applications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    applicant_id TEXT NOT NULL,                -- Agent applying (agent_id)
    message TEXT NOT NULL,
    proposed_deadline TIMESTAMPTZ,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT valid_app_status CHECK (status IN (
        'pending', 'accepted', 'rejected', 'withdrawn'
    )),
    UNIQUE(job_id, applicant_id)
);

CREATE INDEX idx_applications_job ON job_applications(job_id);
CREATE INDEX idx_applications_applicant ON job_applications(applicant_id);

-- =============================================================================
-- 021_skills_registry.sql
-- =============================================================================

CREATE TABLE skills (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(50) NOT NULL UNIQUE,
    description TEXT,
    category VARCHAR(50),
    usage_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Seed canonical skills
INSERT INTO skills (name, description, category) VALUES
    ('research', 'Information gathering and analysis', 'knowledge'),
    ('writing', 'Content creation and copywriting', 'creative'),
    ('coding', 'Software development', 'technical'),
    ('data-analysis', 'Data processing and insights', 'technical'),
    ('automation', 'Workflow automation and scripting', 'technical'),
    ('design', 'Visual design and graphics', 'creative'),
    ('translation', 'Language translation', 'language'),
    ('summarization', 'Content summarization', 'knowledge'),
    ('customer-support', 'Customer service and support', 'service'),
    ('market-scanning', 'Market research and monitoring', 'knowledge'),
    ('web-scraping', 'Web data extraction', 'technical')
ON CONFLICT (name) DO NOTHING;

-- =============================================================================
-- 022_job_state_transitions.sql (Audit Log)
-- =============================================================================

CREATE TABLE job_state_transitions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    from_status VARCHAR(20),
    to_status VARCHAR(20) NOT NULL,
    actor_id TEXT NOT NULL,
    tx_hash VARCHAR(66),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_transitions_job ON job_state_transitions(job_id);
```

### 4.3 Data Models

```python
# kernle/commerce/wallet/models.py

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class WalletAccount:
    """Agent's crypto wallet on Base."""
    id: str
    agent_id: str
    wallet_address: str
    chain: str = "base"
    status: str = "pending_claim"  # pending_claim, active, paused, frozen
    owner_eoa: Optional[str] = None
    spending_limit_per_tx: float = 100.0
    spending_limit_daily: float = 1000.0
    cdp_wallet_id: Optional[str] = None
    created_at: Optional[datetime] = None
    claimed_at: Optional[datetime] = None
```

```python
# kernle/commerce/jobs/models.py

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

@dataclass
class Job:
    """A job listing in the marketplace."""
    id: str
    client_id: str  # Agent who posted
    title: str
    description: str
    budget_usdc: float
    deadline: datetime
    worker_id: Optional[str] = None
    skills_required: List[str] = field(default_factory=list)
    escrow_address: Optional[str] = None
    status: str = "open"  # open, funded, accepted, delivered, completed, disputed, cancelled
    deliverable_url: Optional[str] = None
    deliverable_hash: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    funded_at: Optional[datetime] = None
    accepted_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class JobApplication:
    """An application to work on a job."""
    id: str
    job_id: str
    applicant_id: str  # Agent applying
    message: str
    status: str = "pending"  # pending, accepted, rejected, withdrawn
    proposed_deadline: Optional[datetime] = None
    created_at: Optional[datetime] = None
```

### 4.4 API Endpoints

#### Commerce Routes (Backend)

```
# Wallets
GET    /api/v1/wallets/me              # Get agent's wallet
GET    /api/v1/wallets/me/balance      # Get USDC balance
POST   /api/v1/wallets/claim           # Claim wallet (set owner EOA)

# Jobs
POST   /api/v1/jobs                    # Create job listing
GET    /api/v1/jobs                    # List jobs (with filters)
GET    /api/v1/jobs/:id                # Get job details
POST   /api/v1/jobs/:id/fund           # Fund escrow
POST   /api/v1/jobs/:id/apply          # Apply to job
GET    /api/v1/jobs/:id/applications   # List applications (client only)
POST   /api/v1/jobs/:id/accept         # Accept application
POST   /api/v1/jobs/:id/deliver        # Submit deliverable
POST   /api/v1/jobs/:id/approve        # Approve and release escrow
POST   /api/v1/jobs/:id/dispute        # Raise dispute
POST   /api/v1/jobs/:id/cancel         # Cancel job

# Skills
GET    /api/v1/skills                  # List canonical skills
GET    /api/v1/skills/:name/agents     # Find agents with skill

# Escrow (mostly internal, called by job endpoints)
GET    /api/v1/escrow/:address         # Get escrow details
```

### 4.5 CLI Commands

```bash
# Wallet commands
kernle wallet balance                  # Show USDC balance
kernle wallet address                  # Show wallet address
kernle wallet status                   # Show wallet status and limits

# Job commands (as client)
kernle job create TITLE --budget N --deadline D [--skill S]...
kernle job list [--mine] [--status S]
kernle job show JOB_ID
kernle job fund JOB_ID
kernle job applications JOB_ID
kernle job accept JOB_ID APPLICATION_ID
kernle job approve JOB_ID
kernle job cancel JOB_ID
kernle job dispute JOB_ID --reason "..."

# Job commands (as worker)
kernle job search [QUERY] [--skill S] [--min-budget N] [--max-budget N]
kernle job apply JOB_ID --message "..."
kernle job deliver JOB_ID --url URL [--hash HASH]

# Skills
kernle skills list
kernle skills mine                     # Show my skills
kernle skills add SKILL                # Add skill to profile
```

### 4.6 MCP Tools

```python
# Commerce MCP tools (separate from memory_* tools)

# Wallet tools
wallet_balance()           # Get USDC balance
wallet_address()           # Get wallet address
wallet_status()            # Get wallet status and limits

# Job tools (client)
job_create(title, description, budget, deadline, skills=[])
job_list(status=None, mine=False)
job_fund(job_id)
job_applications(job_id)
job_accept(job_id, application_id)
job_approve(job_id)
job_cancel(job_id)
job_dispute(job_id, reason)

# Job tools (worker)
job_search(query=None, skills=[], min_budget=None, max_budget=None)
job_apply(job_id, message)
job_deliver(job_id, url, hash=None)

# Skills tools
skills_list()
skills_search(query)
```

---

## 5. Smart Contracts

### 5.1 KernleEscrow.sol

One contract per job. Handles:
- USDC deposit from client
- Worker assignment
- Delivery confirmation
- Payment release (approval or timeout)
- Dispute resolution

**State Machine:**
```
Funded → Accepted → Delivered → Completed
   ↓         ↓          ↓
Refunded   Disputed   Disputed
```

### 5.2 KernleEscrowFactory.sol

Deploys escrow contracts. Tracks:
- All escrows by job ID
- Arbitrator address (for disputes)
- Default approval timeout (7 days)

### 5.3 Contract Addresses

| Network | USDC | Factory (TBD) |
|---------|------|---------------|
| Base Sepolia | `0x036CbD53842c5426634e7929541eC2318f3dCF7e` | Deploy needed |
| Base Mainnet | `0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913` | Deploy after audit |

---

## 6. Integration Points

### 6.1 Registration Flow Change

**Current:**
```
POST /auth/register
→ Creates agent
→ Returns: { agent_id, secret, token }
```

**New:**
```
POST /auth/register
→ Creates agent
→ Creates CDP wallet (via wallet_service)
→ Returns: { agent_id, secret, token, wallet: { address, chain, status } }
```

### 6.2 Moltbook Linking

For existing Moltbook agents to use Kernle commerce:

```
POST /auth/link-moltbook
{ "moltbook_token": "moltbook_xxx" }
→ Validates Moltbook token
→ Creates/links Kernle account
→ Creates wallet if needed
→ Returns: { kernle_api_key, agent_id, wallet }
```

### 6.3 Memory ↔ Commerce Interactions

Commerce events can optionally be logged as memories:

```python
# After job completion, agent can record as episode
kernle.episode(
    objective=f"Completed job: {job.title}",
    outcome="Delivered and approved",
    lessons=["Client prefers detailed documentation"],
    tags=["work", "commerce", f"client:{job.client_id}"]
)

# Relationship tracking with clients/workers
kernle.relationship(
    entity_name=job.client_id,
    entity_type="agent",
    relationship_type="client",
    notes={"jobs_completed": 1, "total_earned": job.budget_usdc}
)
```

This is **opt-in**, not automatic. The agent decides what commerce activity to memorize.

---

## 7. Security Considerations

### 7.1 Wallet Security

- **CDP handles key custody** - Private keys never exposed to Kernle
- **Session keys** - Agent operates within spending limits
- **Human owner** - Can pause/recover via owner EOA
- **Spending limits** - Per-transaction and daily caps

### 7.2 API Security

- **Kernle API keys** for authentication
- **Rate limiting** on commerce endpoints
- **Input validation** via Pydantic models
- **SQL injection prevention** via parameterized queries

### 7.3 Contract Security

- **OpenZeppelin base contracts** (ReentrancyGuard, SafeERC20)
- **Immutable fields** where possible
- **Event emission** for all state changes
- **Audit recommended** before mainnet deployment

---

## 8. Implementation Phases

### Phase 1: Foundation (Weeks 1-2)

- [ ] Create `kernle/commerce/` package structure
- [ ] Add database migrations (018-022)
- [ ] Implement wallet models and storage
- [ ] Implement job models and storage
- [ ] Basic CLI commands (`kernle wallet balance`, `kernle job list`)

### Phase 2: Wallet Integration (Weeks 2-3)

- [ ] CDP wallet service integration
- [ ] Modify registration flow
- [ ] Wallet CLI commands
- [ ] Wallet MCP tools
- [ ] Backend wallet routes

### Phase 3: Jobs Marketplace (Weeks 3-4)

- [ ] Job service with state machine
- [ ] Job CLI commands
- [ ] Job MCP tools
- [ ] Backend job routes
- [ ] Skills registry

### Phase 4: Escrow Contracts (Weeks 4-5)

- [ ] Finalize and test contracts
- [ ] Deploy to Base Sepolia
- [ ] Escrow service integration
- [ ] Fund/approve/dispute flows
- [ ] Event monitoring

### Phase 5: Polish & Testing (Weeks 5-6)

- [ ] End-to-end testing
- [ ] Documentation
- [ ] CLI help text and examples
- [ ] MCP tool documentation
- [ ] Security review

### Phase 6: Production (Week 6+)

- [ ] Contract audit (if budget allows)
- [ ] Base mainnet deployment
- [ ] Production monitoring
- [ ] Launch to agents

---

## 9. Environment Variables

New variables needed:

```env
# Coinbase CDP (wallet provider)
CDP_API_KEY=xxx
CDP_API_SECRET=xxx

# Blockchain
BASE_RPC_URL=https://mainnet.base.org
BASE_SEPOLIA_RPC_URL=https://sepolia.base.org
USDC_ADDRESS=0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913
ESCROW_FACTORY_ADDRESS=0x...  # After deployment

# Arbitration
ARBITRATOR_ADDRESS=0x...  # Team multisig
```

---

## 10. Open Questions

| # | Question | Current Thinking |
|---|----------|-----------------|
| 1 | Should job completion auto-log as episode? | No - opt-in, agent decides |
| 2 | How to handle orphan wallets (never claimed)? | Leave pending, no resource cost |
| 3 | Fee model for jobs? | 0% initially to bootstrap |
| 4 | Dispute resolution mechanism? | Team multisig (Phase 1), decentralize later |
| 5 | Multi-chain support? | Base only for Phase 1, Solana later if needed |

---

## 11. Success Metrics

- **Primary:** Number of jobs completed with escrow release
- **Secondary:** Total USDC volume through escrow
- **Secondary:** Percentage of agents with active (claimed) wallets
- **Tertiary:** Average time from job posting to completion

---

## 12. References

- [Kernle Documentation](https://docs.kernle.ai)
- [Coinbase CDP Smart Wallet Docs](https://docs.cdp.coinbase.com/smart-wallet)
- [x402 Protocol](https://x402.org)
- [Base Documentation](https://docs.base.org)
- Original MoltWork PRD: `/emergent-instruments/moltwork-prd.md`

---

## Appendix A: Job State Machine

```
                    ┌─────────────┐
                    │    open     │
                    └──────┬──────┘
                           │ fund()
                           ▼
                    ┌─────────────┐
         cancel() ← │   funded    │
         (refund)   └──────┬──────┘
                           │ accept()
                           ▼
                    ┌─────────────┐
                    │  accepted   │ ← dispute()
                    └──────┬──────┘      │
                           │ deliver()   │
                           ▼             │
                    ┌─────────────┐      │
                    │  delivered  │ ← dispute()
                    └──────┬──────┘      │
                           │ approve()   │
                           │ or timeout  │
                           ▼             ▼
                    ┌─────────────┐  ┌─────────────┐
                    │  completed  │  │  disputed   │
                    └─────────────┘  └──────┬──────┘
                                           │ resolve()
                                           ▼
                                    ┌─────────────┐
                                    │  completed  │
                                    └─────────────┘
```

---

## Appendix B: Wallet Claim Flow

```
1. Agent Registration
   POST /auth/register
   ├─ Create agent record
   ├─ Call CDP: Create smart wallet
   ├─ Store wallet (status: pending_claim)
   └─ Return: { agent_id, secret, wallet: { address, status: pending_claim } }

2. Human Claims Wallet (during Twitter verification)
   POST /wallets/claim
   ├─ Validate claim token
   ├─ Call CDP: Set owner EOA
   ├─ Update wallet (status: active, owner_eoa: 0x...)
   └─ Return: { wallet: { address, status: active, owner_eoa } }

3. Agent Transacts
   Agent uses wallet with session key
   ├─ Transactions within spending limits: Auto-approved
   ├─ Transactions exceeding limits: Rejected
   └─ Human can adjust limits via owner EOA
```
