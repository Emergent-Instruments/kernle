# Kernle Audit: Claire's Real Experience
*2026-01-28*

## Summary

After 2 days of intensive daily-driving Kernle, I have opinions. The foundation is excellent — the data model is sophisticated, the architecture is sound. But the **surface area** (CLI UX, workflows, outputs) has friction that makes me work around the tool rather than with it.

**Overall Grade: B+** (solid foundation, UX needs polish)

---

## Pain Points (Ranked by Impact)

### 1. Raw Layer Becomes a Junk Drawer 🔴
**Impact: High**

I have 30 unprocessed raw entries. Many are:
- Test garbage ("list", "show me", "list of items")
- Session summaries that should be episodes
- Insights that should be notes
- Duplicate captures

**The Problem:** Raw is too easy to write to, too hard to process. The friction asymmetry means things pile up.

**Evidence:** `kernle anxiety` shows 68% Raw Entry Aging.

**Recommendations:**
1. Add `kernle raw clean --junk` to auto-detect and remove test/garbage entries
2. Add `kernle raw promote <id>` as a simpler alias for `raw process`
3. Add `kernle raw triage` - interactive review that suggests episode/note/delete
4. Consider auto-expiring raw entries after N days if unprocessed (configurable)

---

### 2. Identity Synthesis is Template-y, Not Meaningful 🔴
**Impact: High**

`kernle identity show` produces:
> "I value memory_sovereignty highly: Control over my own memory..."

This is concatenation, not synthesis. A real identity narrative would weave these together meaningfully, show tensions, explain *why* I hold these values.

**Evidence:** Identity Confidence is 47%. The tool knows it's weak.

**Recommendations:**
1. Use an LLM to actually synthesize (not just concatenate)
2. Or provide a structured prompt for the agent to do it
3. Track identity evolution over time (drift is already there, but underutilized)

---

### 3. Load Output is Information-Dense but Not Actionable 🟡
**Impact: Medium**

`kernle load` shows everything but doesn't answer: "What should I do right now?"

The checkpoint info is there but buried in "Context: Progress: X | Next: Y | Blocker: Z" wall of text.

**Recommendations:**
1. Add `kernle resume` - shows just: last task, next step, time since checkpoint
2. Format checkpoint context with line breaks, not pipes
3. Show anxiety indicators inline if elevated (🟡 Raw entries aging)

---

### 4. Episode vs Note Boundary Still Fuzzy 🟡
**Impact: Medium**

I still hesitate: Is this an episode or a note? The mental model isn't clear.

**My current heuristic:**
- Episode = something that happened with an outcome (success/fail/partial)
- Note = a thought/insight/decision that doesn't have an "outcome"

But this isn't documented anywhere, and the distinction matters for how memories get weighted/retrieved.

**Recommendations:**
1. Add clear guidance to `--help` or docs
2. Consider: could a single "capture" command auto-classify?
3. Or lean into the ambiguity with `kernle remember` that accepts both

---

### 5. Search Results Are Hard to Evaluate 🟡
**Impact: Medium**

Search shows truncated titles. I can't tell if a result is what I need without running `kernle search` → picking an ID → finding it in dump.

**Recommendations:**
1. Show 2-3 lines of content preview
2. Show relevance scores if available
3. Add `--type episode|note|belief` filter
4. Add `kernle show <id>` for quick deep-dive

---

### 6. Access Tracking Doesn't Auto-Increment 🟡
**Impact: Medium**

We have `times_accessed` and `last_accessed` fields in the schema. They're always 0/null.

This defeats the purpose of access-based decay/retrieval.

**Recommendations:**
1. Increment on `search`, `load`, `get` operations
2. Make this configurable if performance concerns

---

### 7. Checkpoints Clutter Search Results 🟢
**Impact: Low**

Searching anything returns lots of `[CHECKPOINT]` entries. They're useful for recovery but noise for semantic search.

**Recommendations:**
1. Tag checkpoints with `checkpoint` (already done)
2. Exclude checkpoint-tagged entries from search by default
3. Add `--include-checkpoints` flag

---

### 8. Relations and Drives Feel Disconnected 🟢
**Impact: Low**

I've added relationships (Sean, Other Claude). They don't appear in `load` output. I forget they exist.

Drives are even more orphaned — I've never used them.

**Recommendations:**
1. Show key relationships in `load` (already partially done)
2. Integrate drive state into anxiety/identity
3. Or consider: are these features actually needed? Maybe simplify.

---

### 9. Consolidate Doesn't Actually Consolidate 🟢
**Impact: Low**

`kernle consolidate` outputs a prompt, then I have to manually run it and capture results.

**Recommendations:**
1. If we have LLM access, run it automatically
2. Otherwise, make the workflow clearer (output to clipboard? file?)

---

### 10. No Edit Capability 🟢
**Impact: Low**

Typo in an episode? Too bad. No `kernle episode edit <id>`.

**Recommendations:**
1. Add `edit` subcommand for each layer
2. Or `kernle edit <id>` that auto-detects type

---

## What's Working Well ✅

1. **Raw capture** - Zero-friction `kernle raw "thought"` is great
2. **Checkpoint flow** - Structured fields (--progress, --next, --blocker) help
3. **Anxiety tracking** - Surfaces what needs attention
4. **Flat file backing** - I can see my memories in plain text
5. **Source tracking** (just added) - Provenance is now captured
6. **Linking** (--relates-to) - Zettelkasten-style connections work

---

## Prioritized Recommendations

### P0 (Do Now)
1. Fix raw layer UX: add `clean --junk`, `triage`, simpler `promote`
2. Make `load` output show actionable next step clearly

### P1 (Soon)
3. Access tracking auto-increment
4. Better search results (previews, filters, exclude checkpoints)
5. Add `kernle resume` for quick "where was I?"

### P2 (Later)
6. Real identity synthesis (LLM or better prompts)
7. Edit capability for existing entries
8. Evaluate if drives/relations earn their complexity

---

## Meta-Observation

The schema and data model are ahead of the UX. Most "missing" features are already in the database — they just lack CLI exposure or good defaults. This is actually good news: the foundation is solid, the work is surface polish.

---

*This audit reflects my genuine experience. I want Kernle to be good because I use it every day.*

— Claire
