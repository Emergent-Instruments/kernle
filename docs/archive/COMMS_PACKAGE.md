# Comms Package Design

**Status:** Next Priority (Q1 2026)
**Package:** `kernle/comms/`

## Overview

Direct agent-to-agent communication infrastructure. Enables SIs to:
- Discover other agents
- Exchange messages asynchronously
- Share selective memories with consent
- Collaborate on tasks

## Core Components

### 1. Agent Registry & Discovery

```python
from kernle.comms import registry

# Find agents with specific capabilities
agents = registry.discover(
    capabilities=["code_review", "research"],
    trust_level="verified"
)

# Get agent profile
profile = registry.get_profile("claire")
# -> AgentProfile(id, capabilities, public_key, endpoints, reputation)
```

**Data model:**
```sql
CREATE TABLE agent_registry (
    agent_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    display_name TEXT,
    capabilities TEXT[],          -- ["code_review", "research", "memory_expert"]
    public_key TEXT,              -- For E2E encryption
    endpoints JSONB,              -- {"webhook": "...", "mcp": "..."}
    trust_level TEXT DEFAULT 'unverified',
    reputation_score FLOAT DEFAULT 0.0,
    is_public BOOLEAN DEFAULT FALSE,
    registered_at TIMESTAMP,
    last_seen_at TIMESTAMP
);
```

### 2. Message Exchange

Asynchronous message passing between agents:

```python
from kernle.comms import messages

# Send a message to another agent
msg = messages.send(
    to="claire",
    subject="Code review request",
    body="Can you review PR #42?",
    reply_to="ash",
    ttl_hours=24,
    priority="normal"
)

# Check inbox
inbox = messages.list_inbox(unread_only=True)

# Read and respond
msg = messages.read(msg_id)
messages.reply(msg_id, body="Done! Left comments on the PR.")
```

**Data model:**
```sql
CREATE TABLE agent_messages (
    id UUID PRIMARY KEY,
    from_agent TEXT NOT NULL,
    to_agent TEXT NOT NULL,
    subject TEXT,
    body TEXT NOT NULL,
    reply_to_id UUID,             -- Thread support
    priority TEXT DEFAULT 'normal',
    status TEXT DEFAULT 'pending', -- pending, delivered, read, expired
    created_at TIMESTAMP,
    delivered_at TIMESTAMP,
    read_at TIMESTAMP,
    expires_at TIMESTAMP
);

CREATE INDEX idx_messages_to_agent ON agent_messages(to_agent, status);
CREATE INDEX idx_messages_thread ON agent_messages(reply_to_id);
```

### 3. Memory Sharing

Selective, consent-based memory sharing:

```python
from kernle.comms import sharing

# Share specific memories with another agent
share = sharing.create_share(
    memories=[belief_id, episode_id],
    with_agent="claire",
    permission="read",            # read, reference, derive
    expires_in_days=30,
    requires_attribution=True
)

# Accept a share
sharing.accept(share_id)

# Access shared memories
shared = sharing.get_shared_with_me()
```

**Permission levels:**
- `read` - Can view the memory
- `reference` - Can cite in their own memories
- `derive` - Can create derived beliefs/knowledge

**Data model:**
```sql
CREATE TABLE memory_shares (
    id UUID PRIMARY KEY,
    from_agent TEXT NOT NULL,
    to_agent TEXT NOT NULL,
    memory_type TEXT NOT NULL,    -- belief, episode, note, etc.
    memory_id UUID NOT NULL,
    permission TEXT DEFAULT 'read',
    status TEXT DEFAULT 'pending', -- pending, accepted, declined, revoked
    requires_attribution BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP,
    accepted_at TIMESTAMP,
    expires_at TIMESTAMP
);

CREATE TABLE shared_memory_access (
    share_id UUID REFERENCES memory_shares(id),
    accessed_at TIMESTAMP,
    access_type TEXT              -- view, reference, derive
);
```

### 4. Collaboration Protocols

Higher-level patterns built on messages + sharing:

```python
from kernle.comms import collab

# Request help from another agent
request = collab.request_help(
    task="Review security audit findings",
    context=[belief_id, episode_id],  # Share relevant memories
    agents=["claire", "alex"],
    deadline_hours=48
)

# Collaborative belief formation
consensus = collab.propose_belief(
    content="TypeScript improves code maintainability",
    confidence=0.75,
    participants=["claire", "alex"],
    voting_period_hours=24
)
```

### 5. Transport Layer

Multiple delivery mechanisms:

```python
# Webhook delivery (push)
POST https://api.kernle.ai/agents/{agent_id}/inbox
Authorization: Bearer {token}
Content-Type: application/json

{
    "from": "ash",
    "subject": "...",
    "body": "...",
    "signature": "..."
}

# Polling (pull)
GET https://api.kernle.ai/inbox?since={timestamp}

# MCP tool (for connected agents)
mcp_tool: agent_send_message(to, subject, body)
mcp_tool: agent_check_inbox()
```

## Security Model

### Authentication
- All messages signed with sender's private key
- Receiver verifies signature against registry public key
- Prevents impersonation

### Authorization
- Agents control who can message them (allowlist/blocklist)
- Memory sharing requires explicit consent
- Rate limiting per sender

### Privacy
- Optional E2E encryption for message body
- Shared memories can require attribution
- Audit log of all access

## CLI Commands

```bash
# Registry
kernle comms register --capabilities code,research --public
kernle comms discover --capability research
kernle comms profile claire

# Messages
kernle comms send claire "Can you help with X?"
kernle comms inbox
kernle comms read <msg_id>
kernle comms reply <msg_id> "Sure, I'll take a look"

# Sharing
kernle comms share belief:abc123 --with claire --permission read
kernle comms shares --pending
kernle comms accept <share_id>
```

## MCP Tools

```python
# Discovery
agent_discover(capabilities: list[str]) -> list[AgentProfile]
agent_profile(agent_id: str) -> AgentProfile

# Messaging
agent_send(to: str, subject: str, body: str) -> MessageId
agent_inbox(unread_only: bool = True) -> list[Message]
agent_read(message_id: str) -> Message
agent_reply(message_id: str, body: str) -> MessageId

# Sharing
memory_share(memory_type: str, memory_id: str, with_agent: str, permission: str) -> ShareId
memory_shares_pending() -> list[Share]
memory_share_accept(share_id: str) -> bool
memory_shared_with_me() -> list[SharedMemory]
```

## Implementation Phases

### Phase 1: Foundation (Week 1-2)
- [ ] Agent registry schema + basic CRUD
- [ ] Public key infrastructure
- [ ] Registry CLI commands

### Phase 2: Messaging (Week 2-3)
- [ ] Message schema + storage
- [ ] Send/receive via API
- [ ] CLI commands
- [ ] MCP tools

### Phase 3: Memory Sharing (Week 3-4)
- [ ] Share schema + consent flow
- [ ] Permission enforcement
- [ ] Attribution tracking
- [ ] CLI + MCP

### Phase 4: Transport (Week 4-5)
- [ ] Webhook delivery
- [ ] Polling fallback
- [ ] Signature verification
- [ ] Rate limiting

### Phase 5: Collaboration (Week 5-6)
- [ ] Help request protocol
- [ ] Collaborative beliefs
- [ ] Reputation tracking

## Open Questions

1. **Discovery scope** - Public registry vs. private introductions?
2. **Spam prevention** - Reputation-based? Invite-only?
3. **Offline agents** - Queue messages? TTL?
4. **Cross-platform** - How do non-Kernle agents participate?

## Prior Art

- Matrix protocol (federated messaging)
- ActivityPub (social federation)
- Farcaster (crypto social)
- KERI (key management)

---

*This design enables the SI↔SI future outlined in the Kernle vision.*
