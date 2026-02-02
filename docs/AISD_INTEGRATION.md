# Kernle × AISD: Integration Lifecycle

How Kernle integrates with a web application like AISD (AI Student Dashboard) — where Kernle is the memory layer for a multi-user platform.

## Architecture Overview

AISD is a school district platform that provides personalized AI tutoring and progress tracking for students. Each student gets their own Kernle stack (`student-{id}`), giving them persistent memory across sessions without AISD needing to build its own memory system.

## Current Architecture (Without Boot Layer)

```
STUDENT/PARENT OPENS APP
          ↓
1.  HTTP request hits AISD backend
2.  AISD authenticates user, identifies student_id
3.  AISD needs Kernle stack ID for this student
    → WHERE DOES THIS COME FROM?
    → Currently: AISD's own DB table (student_id → stack_id)
4.  AISD reads from Supabase: grades, schedule, assignments
5.  AISD calls: kernle -a student-{id} load
6.  Kernle returns: beliefs, episodes, playbooks
7.  AISD assembles context:
    - Kernle memory (learning style, strengths, history)
    - Live grade data (from Supabase)
    - Today's schedule
    - RAG from course content (if tutoring)
8.  AISD builds prompt, calls LLM
9.  LLM responds
10. AISD parses response
11. If significant: kernle -a student-{id} episode "..."
12. At session end: kernle -a student-{id} checkpoint "..."
```

### The Problem

At step 3, AISD needs config (stack ID, school URLs, teacher email patterns) **before** calling Kernle. It currently must maintain its own config table mapping `student_id → stack_id` plus any per-student settings. That's a separate config mechanism outside Kernle — duplicated state that can drift.

## With Boot Layer

```
STUDENT/PARENT OPENS APP
          ↓
1.  HTTP request, authenticate, get student_id
2.  Read boot file: ~/.kernle/student-{id}/boot.md
    → Contains: stack_id, school_url, teacher_prefs, etc.
    → File always exists, auto-updated on checkpoint
    → OR: kernle -a student-{id} boot get school_url (instant)
3.  Now have config — proceed to load full memory
4.  kernle -a student-{id} load (includes boot in response)
5.  AISD assembles context (same as before)
6-12. Same as before
```

### What Changes

| Concern | Before | After |
|---------|--------|-------|
| Stack ID lookup | AISD's own DB table | `boot.md` or `boot get stack_id` |
| School config | AISD's own DB table | Boot layer key/values |
| Per-student prefs | AISD's own DB table | Boot layer + Kernle values |
| Config drift risk | Two sources of truth | One source (Kernle) |
| New integration setup | Build config table + migration | `kernle boot set` and done |

## Memory Type Mapping

Kernle's memory types map naturally to educational contexts:

| Kernle Type | AISD Usage | Example |
|-------------|-----------|---------|
| **Belief** | Academic self-assessment | "I'm strong in algebra but struggle with geometry proofs" (85%) |
| **Value** | Learning preferences | "I learn best with visual examples" |
| **Episode** | Session history | "Worked through quadratic formula — breakthrough on discriminant" |
| **Goal** | Academic targets | "Improve essay structure by spring semester" |
| **Relationship** | Key people | "Ms. Rodriguez — math teacher, prefers formal proofs" |
| **Raw** | Unprocessed observations | "Student hesitated on fraction problems three times today" |

### Consolidation Pipeline

Raw observations promote through the pipeline automatically:

```
Raw: "Student hesitated on fractions 3x today"
  ↓ consolidate
Episode: "Feb 1 tutoring — fraction weakness identified"
  ↓ consolidate  
Belief: "This student needs fraction review before advancing" (75%)
```

Teachers and counselors see beliefs and episodes. Raw entries are working memory that gets refined.

## One Stack Per Student

Each student gets an isolated Kernle stack:

```bash
# Initialize a new student
kernle -a student-4521 init

# Set boot config
kernle -a student-4521 boot set school_id "austin-isd"
kernle -a student-4521 boot set grade_level "10"
kernle -a student-4521 boot set advisor_email "mrodriguez@aisd.net"

# Seed initial beliefs (from intake assessment)
kernle -a student-4521 believe "Strong in computational math" --confidence 0.8
kernle -a student-4521 believe "Needs support with reading comprehension" --confidence 0.7

# Set learning values
kernle -a student-4521 value "Learns best with visual diagrams"
kernle -a student-4521 value "Responds well to gamified challenges"
```

### Why Per-Student Stacks

- **Privacy isolation** — one student's memory never leaks to another
- **Portable** — student transfers schools, stack goes with them
- **Auditable** — full provenance on every belief (where it came from, confidence history)
- **No shared state** — no risk of cross-contamination between students

## Python API Integration

```python
from kernle import Kernle

# Fast config lookup (no full load)
k = Kernle(agent_id=f"student-{student_id}")
school_url = k.boot_get("school_url")
grade_level = k.boot_get("grade_level")

# Full memory load for tutoring session
state = k.load()
beliefs = state["beliefs"]      # Academic self-model
episodes = state["episodes"]    # Recent session history
values = state["values"]        # Learning preferences
goals = state["goals"]          # Academic targets

# During session — capture observations
k.raw(f"Student solved {problem_type} in {time}s — {outcome}")

# Significant moment — capture as episode
k.episode("Breakthrough: understood proof by contradiction for first time")

# Update belief based on evidence
k.believe(
    "Strong in geometric proofs",
    confidence=0.75,
    source_type="derived",
    source_episodes=["ep_abc123"]
)

# End of session
k.checkpoint("Geometry tutoring — covered proofs ch. 4.2-4.5")
```

## Boot Layer Benefits for AISD

1. **No custom config tables** — AISD doesn't need `student_config` in its own DB
2. **Instant access** — `boot get` and `boot.md` are available without loading full memory
3. **Self-sufficient** — Kernle manages its own config; AISD just reads it
4. **Same pattern everywhere** — whether it's OpenClaw, AISD, or a future integration, boot config works identically
5. **File-based fallback** — `~/.kernle/student-{id}/boot.md` is readable by any process, no Kernle CLI required

## Privacy & Access Control

Per the [Memory Privacy Spec](./MEMORY_PRIVACY_SPEC.md):

- **Private by default** — empty `access_grants` means only the owning stack can access memories
- **Consent tracking** — `consent_grants` records who authorized what data
- **Subject tagging** — `subject_ids` identifies who a memory is about
- **Query-time filtering** — unauthorized access blocked at read time

For AISD specifically:
- Student memories are private to their stack
- Teachers can be granted read access via `access_grants`
- Parents can be granted access per district policy
- Counselors get scoped access for relevant academic beliefs only

## Comparison: OpenClaw vs AISD Integration

| Aspect | OpenClaw | AISD |
|--------|----------|------|
| Users | 1 agent per human | 1 stack per student (thousands) |
| Boot config source | `MEMORY.md` (workspace injection) | `boot.md` file or `boot get` API |
| Memory load trigger | Agent decides (step 7) | Backend calls on session start |
| Who calls Kernle | The AI agent itself | The application backend |
| Consolidation | Agent runs during heartbeats | Cron job or backend trigger |
| Privacy scope | Agent-level isolation | Student-level isolation + role-based grants |

The boot layer is the **platform-agnostic solution** to config availability. OpenClaw gets it via `MEMORY.md` injection. AISD gets it via API or file. Same Kernle, same pattern, different entry points.
