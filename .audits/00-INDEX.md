# Kernle Architecture Audit

> **Date**: 2026-01-30 (Updated)
> **Purpose**: Comprehensive documentation of Kernle's memory management system
> **Status**: Revised after peer review (Claude, Gemini, GPT 5.2)

---

## Documents in This Audit

| # | Document | Purpose |
|---|----------|---------|
| 01 | [PROPOSED-RAW-LAYER.md](./01-PROPOSED-RAW-LAYER.md) | **Revised proposal** for blob-based raw layer (approved) |
| 02 | [DATABASE-SCHEMA.md](./02-DATABASE-SCHEMA.md) | Complete database schema for SQLite and PostgreSQL |
| 03 | [DATA-FLOW-DIAGRAMS.md](./03-DATA-FLOW-DIAGRAMS.md) | Visual diagrams of data movement through the system |
| 04 | [EXTERNAL-INTEGRATIONS.md](./04-EXTERNAL-INTEGRATIONS.md) | CLI, MCP, SDK, and REST API documentation |
| 05 | [FEEDBACK-CONSOLIDATION.md](./05-FEEDBACK-CONSOLIDATION.md) | Peer review feedback and analysis |
| 06 | [ROADMAP.md](./06-ROADMAP.md) | Planned improvements and deferred items |

---

## Executive Summary

### What is Kernle?

Kernle is a **stratified memory system for AI agents** inspired by cognitive science. It provides:

1. **Local-first storage** with optional cloud sync
2. **Multi-layer memory** (raw → episodic → semantic → identity)
3. **Meta-memory** (confidence tracking, provenance, verification)
4. **Salience-based forgetting** with protection for core identity
5. **Anxiety monitoring** (memory health alerts)

### Key Design Principles

1. **All data comes from the agent** - No human user input paths in memory
2. **Local-first** - SQLite primary, cloud sync optional
3. **Agent-driven consolidation** - System provides scaffolds, agent decides what to believe
4. **Soft delete everywhere** - Tombstoning for audit trail, no hard deletes
5. **Protected identity** - Values and drives never decay or forget

---

## Raw Layer Changes (Approved)

The raw layer will be simplified from structured fields to blob-based storage.

### Final Schema

```sql
CREATE TABLE raw_entries (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    blob TEXT NOT NULL,              -- Unstructured brain dump
    captured_at TEXT NOT NULL,       -- When (auto)
    source TEXT DEFAULT 'unknown',   -- cli|mcp|sdk|import (auto)
    processed INTEGER DEFAULT 0,     -- Tracking
    processed_into TEXT,             -- Audit trail
    -- sync fields...
);

-- FTS5 for keyword search
CREATE VIRTUAL TABLE raw_fts USING fts5(blob, content=raw_entries);

-- Partial index for anxiety queries
CREATE INDEX idx_raw_unprocessed ON raw_entries(captured_at)
    WHERE processed = 0 AND deleted = 0;
```

### Key Decisions (from peer review)

| Decision | Rationale |
|----------|-----------|
| Keep `source` field | Auto-populated enum for debugging/provenance |
| Add FTS5 | Safety net for keyword search when backlogs happen |
| Add partial index | Performance for anxiety system queries |
| Raw sync OFF by default | Security - raw blobs often contain accidental secrets |
| Warn, don't reject large blobs | Allow legitimate large entries, let anxiety handle it |
| Natural language migration | More readable than delimiters when agents review old entries |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ENTRY POINTS                                │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐               │
│  │    CLI      │   │    MCP      │   │  Python SDK │               │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘               │
│         └─────────────────┼─────────────────┘                       │
│                           ▼                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                      KERNLE CORE                            │   │
│  │  Mixins: Anxiety, Emotions, Forgetting, Knowledge,          │   │
│  │          MetaMemory, Suggestions                            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                           ▼                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   STORAGE ABSTRACTION                       │   │
│  │  ┌───────────────────┐   ┌───────────────────┐              │   │
│  │  │  SQLiteStorage    │   │  PostgresStorage  │              │   │
│  │  │  (local-first)    │   │  (cloud)          │              │   │
│  │  └───────────────────┘   └───────────────────┘              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                           ▼                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    SYNC ENGINE                              │   │
│  │  • Offline queue with UPSERT deduplication                  │   │
│  │  • Last-write-wins conflict resolution                      │   │
│  │  • Raw sync OFF by default (security)                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Memory Layers

```
Layer 0: RAW CAPTURES
├── Storage: raw_entries table (blob-based)
├── Lifespan: Short (days/weeks)
├── Search: FTS5 keyword (no embeddings)
├── Sync: OFF by default
└── Purpose: Zero-friction brain dump

Layer 1: EPISODIC MEMORY
├── Storage: episodes table
├── Lifespan: Long (months/years)
├── Search: Semantic + keyword
├── Sync: ON by default
└── Purpose: Autobiographical experiences with lessons

Layer 2: SEMANTIC MEMORY
├── Storage: beliefs, notes tables
├── Lifespan: Long with revision chains
├── Search: Semantic + keyword
├── Sync: ON by default
└── Purpose: Facts, decisions, insights

Layer 3: IDENTITY & VALUES
├── Storage: agent_values, goals, drives tables
├── Lifespan: Permanent (protected)
├── Search: Semantic + keyword
├── Sync: ON by default
└── Purpose: Core principles, direction

SUPPORTING:
├── playbooks (procedural memory)
├── relationships (social graph)
├── emotional_memory (valence/arousal)
├── meta_memory (confidence/provenance)
└── forgetting (salience-based retention)
```

---

## Implementation Priority

### High Priority (Do First)
1. Add FTS5 index for raw search
2. Add partial index for unprocessed queries
3. Make raw sync opt-in (default off)
4. Migrate data to blob format

### Medium Priority (Do Soon)
1. Keep `source` field as auto-populated enum
2. Update SDK/CLI/MCP APIs
3. Add deprecation warnings
4. Update migration to natural language format

### Deferred (See Roadmap)
1. Embedding duality resolution (local vs cloud)
2. Semantic anxiety trigger (contradiction detection)
3. Multi-agent support
4. Promotion helper with LLM extraction

---

## Next Steps

1. **Review this documentation** for final approval
2. **Implement raw layer changes** per proposal
3. **Update tests** for new schema and behavior
4. **Document migration** for existing deployments
