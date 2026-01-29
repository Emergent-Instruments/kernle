# `kernle import` Command Design

**Author:** Claire (subagent)  
**Date:** 2025-01-28  
**Status:** Draft

## Problem Statement

Users coming from flat-file memory systems (like MEMORY.md, CLAUDE.md with inline memory, or other markdown-based notes) have no migration path into Kernle's stratified memory system. They face:

1. **Lost context**: Years of accumulated knowledge trapped in unstructured text
2. **Manual migration burden**: Copying content entry-by-entry is tedious and error-prone
3. **Classification confusion**: Unclear how to map existing content to Kernle's layers (episodes, notes, beliefs, raw)

## Requirements

1. Import markdown files into appropriate memory layers
2. Handle different content types (episodes, notes, beliefs, raw)
3. Support both interactive and batch modes
4. Preserve original file as reference/backup
5. Handle edge cases (empty files, malformed markdown, duplicates)

---

## Option 1: Smart Parser with Heuristics

**Philosophy:** Analyze markdown structure and content to auto-classify entries.

### CLI Interface

```bash
# Basic import - analyzes and suggests classifications
kernle import MEMORY.md

# Interactive mode - prompts for each entry
kernle import MEMORY.md --interactive

# Batch mode with explicit type
kernle import notes.md --as note

# Auto-classify without prompts
kernle import MEMORY.md --auto

# Dry run - show what would be imported
kernle import MEMORY.md --dry-run

# Import from directory
kernle import ./memories/ --recursive
```

### Subcommand Structure

```
kernle import <path> [options]
  --interactive, -i    Prompt for classification of each entry
  --auto, -a           Auto-classify without prompts
  --dry-run, -n        Preview without importing
  --as TYPE            Force all entries to specific type (episode|note|belief|raw)
  --recursive, -r      Import all .md files in directory
  --skip-duplicates    Skip entries that match existing content
  --source NAME        Tag imported entries with source name
  --json               Output results as JSON
```

### Classification Heuristics

| Pattern | Classification | Confidence |
|---------|---------------|------------|
| "## Session", "### What I did", time-based headings | Episode | High |
| "Learned:", "Lesson:", "Key insight:" | Episode (lesson) | High |
| "Decision:", "Decided:", "We agreed" | Note (decision) | High |
| "Remember:", "Important:", "Note:" | Note | Medium |
| "I believe", "I think", "Always/Never" | Belief | Medium |
| Short bullet points, quick thoughts | Raw | Low |
| Structured "Objective/Outcome" sections | Episode | High |

### Implementation

```python
# kernle/cli/commands/import_cmd.py

def cmd_import(args, k: "Kernle"):
    """Import markdown files into Kernle memory."""
    
    if args.import_action == "file":
        entries = parse_markdown_file(args.path)
        
        for entry in entries:
            classification = classify_entry(entry)
            
            if args.interactive:
                # Show entry + suggested classification
                # Prompt for: [E]pisode [N]ote [B]elief [R]aw [S]kip
                pass
            elif args.auto:
                import_entry(k, entry, classification)
            else:
                # Default: show dry-run style preview
                pass
```

### Pros
- **Low friction**: Works with any markdown file, no special format required
- **Smart defaults**: Heuristics handle common cases automatically
- **Gradual adoption**: `--interactive` lets users learn the system while importing

### Cons
- **Heuristics fail**: Unusual formatting or domain-specific content may misclassify
- **Complex implementation**: Parser needs to handle many markdown variants
- **Magic = confusion**: Users may not understand why something was classified a certain way

---

## Option 2: Template-Based Import

**Philosophy:** Provide specific templates for common source formats.

### CLI Interface

```bash
# Auto-detect format
kernle import MEMORY.md

# Specify format explicitly
kernle import MEMORY.md --format memory-md
kernle import claude-notes.md --format claude
kernle import obsidian/ --format obsidian --recursive

# List available formats
kernle import --formats
```

### Supported Formats

```
kernle import --formats

Available import formats:
  memory-md    Standard MEMORY.md (sections with ## headers)
  claude       CLAUDE.md agent memory format
  obsidian     Obsidian vault (folder structure = tags)
  logseq       Logseq journals and pages
  notion       Notion export (markdown)
  raw          Treat everything as raw entries (safest)
```

### Format Specifications

**memory-md format:**
```markdown
## Preferences
- Prefers dark mode
- Uses vim keybindings

## Decisions
- 2025-01-15: Switched to Railway for backend hosting
- 2025-01-10: Chose SQLite over Postgres for local storage

## Lessons Learned
- Always test migrations locally first
- Railway webhooks need explicit reconnection after GitHub re-auth
```

Maps to:
- `## Preferences` → Notes (type: preference)
- `## Decisions` → Notes (type: decision) with dates extracted
- `## Lessons Learned` → Episodes (lessons)

### Implementation

```python
# kernle/cli/commands/import_cmd.py

FORMAT_PARSERS = {
    "memory-md": parse_memory_md,
    "claude": parse_claude_format,
    "obsidian": parse_obsidian_vault,
    "raw": parse_as_raw,
}

def cmd_import(args, k: "Kernle"):
    format_type = args.format or detect_format(args.path)
    parser = FORMAT_PARSERS.get(format_type, parse_as_raw)
    
    entries = parser(args.path)
    # ... import logic
```

### Pros
- **Predictable**: Each format has clear, documented mapping rules
- **Optimized**: Can extract maximum value from known formats
- **Extensible**: Easy to add new formats via plugins

### Cons
- **Limited coverage**: Users with custom formats fall back to raw
- **Maintenance burden**: Each format needs ongoing support
- **Discovery problem**: Users need to know which format to use

---

## Option 3: LLM-Assisted Import (Hybrid)

**Philosophy:** Use Claude to analyze and classify content, with human oversight.

### CLI Interface

```bash
# Analyze file and generate import plan
kernle import MEMORY.md --analyze

# Review and execute plan
kernle import MEMORY.md --plan import-plan.json
kernle import MEMORY.md --plan import-plan.json --execute

# Quick import with AI classification (requires API key)
kernle import MEMORY.md --ai

# Interactive AI-guided import
kernle import MEMORY.md --ai --interactive
```

### Workflow

1. **Analyze**: LLM reads file, identifies distinct memory units, suggests classifications
2. **Plan**: Generates JSON import plan that can be reviewed/edited
3. **Execute**: Applies plan to create Kernle entries

### Import Plan Format

```json
{
  "source": "MEMORY.md",
  "analyzed_at": "2025-01-28T10:30:00Z",
  "entries": [
    {
      "id": 1,
      "content": "Railway webhooks broke after GitHub re-auth",
      "suggested_type": "episode",
      "suggested_data": {
        "objective": "Debug Railway deployment failure",
        "outcome": "Discovered webhooks need reconnection after GitHub re-auth",
        "lessons": ["Always check webhook status after auth changes"]
      },
      "confidence": 0.85,
      "source_lines": [15, 20],
      "reasoning": "Contains problem/solution structure with clear lesson learned"
    },
    {
      "id": 2,
      "content": "Prefers dark mode",
      "suggested_type": "note",
      "suggested_data": {
        "type": "preference"
      },
      "confidence": 0.95,
      "source_lines": [5, 5],
      "reasoning": "Simple preference statement"
    }
  ]
}
```

### Implementation

```python
def cmd_import(args, k: "Kernle"):
    if args.analyze:
        # Generate import plan using LLM
        plan = analyze_with_llm(args.path)
        save_plan(plan, args.plan_output or "import-plan.json")
        print_plan_summary(plan)
        
    elif args.plan:
        plan = load_plan(args.plan)
        
        if args.execute:
            execute_plan(k, plan, interactive=args.interactive)
        else:
            print_plan_summary(plan)
            print("\nRun with --execute to import")
            
    elif args.ai:
        # Quick mode: analyze + execute
        plan = analyze_with_llm(args.path)
        execute_plan(k, plan, interactive=args.interactive)
```

### Pros
- **Highest accuracy**: LLM understands context and nuance
- **Transparent**: Plan file shows exactly what will happen
- **Editable**: Users can modify plan before execution
- **Self-documenting**: Reasoning field explains classifications

### Cons
- **API dependency**: Requires Claude API key and costs money
- **Latency**: LLM analysis takes time for large files
- **Overkill for simple cases**: Unnecessary for well-structured files

---

## Recommended Approach: Option 1 + Elements of Option 2

**Rationale:**
- Start with heuristics (Option 1) for zero-config experience
- Add format templates (Option 2) for known formats like MEMORY.md
- Defer LLM assistance (Option 3) to a future `--ai` flag

### Minimal Viable Implementation

```bash
# Core command - works immediately
kernle import MEMORY.md                    # Dry-run preview
kernle import MEMORY.md --execute          # Actually import
kernle import MEMORY.md -i                 # Interactive classification

# Power user options
kernle import MEMORY.md --as raw           # Force all as raw (safe default)
kernle import MEMORY.md --format memory-md # Use specific parser
```

### Argument Parser Addition

```python
# In main() argparse setup

# import (file import)
p_import = subparsers.add_parser("import", help="Import markdown files into memory")
p_import.add_argument("path", help="File or directory to import")
p_import.add_argument("--interactive", "-i", action="store_true",
                     help="Prompt for classification of each entry")
p_import.add_argument("--execute", "-x", action="store_true",
                     help="Actually import (default is dry-run)")
p_import.add_argument("--as", dest="as_type", 
                     choices=["episode", "note", "belief", "raw"],
                     help="Force all entries to specific type")
p_import.add_argument("--format", "-f",
                     choices=["auto", "memory-md", "raw"],
                     default="auto",
                     help="Source format (default: auto-detect)")
p_import.add_argument("--recursive", "-r", action="store_true",
                     help="Import all .md files in directory")
p_import.add_argument("--source", "-s",
                     help="Tag entries with source name")
p_import.add_argument("--skip-duplicates", action="store_true",
                     help="Skip entries matching existing content")
p_import.add_argument("--json", "-j", action="store_true",
                     help="Output as JSON")
```

### Command Handler

```python
# kernle/cli/commands/import_cmd.py

def cmd_import(args, k: "Kernle"):
    """Import markdown files into Kernle memory."""
    from pathlib import Path
    
    path = Path(args.path)
    if not path.exists():
        print(f"✗ Path not found: {args.path}")
        return
    
    # Collect files to import
    if path.is_dir():
        if not args.recursive:
            print(f"✗ {args.path} is a directory. Use --recursive to import all .md files")
            return
        files = list(path.rglob("*.md"))
    else:
        files = [path]
    
    if not files:
        print("No markdown files found.")
        return
    
    print(f"Importing {len(files)} file(s)...")
    print()
    
    total_entries = 0
    imported = {"episode": 0, "note": 0, "belief": 0, "raw": 0}
    skipped = 0
    
    for file_path in files:
        entries = parse_markdown_file(file_path, format=args.format)
        
        for entry in entries:
            # Determine classification
            if args.as_type:
                classification = args.as_type
                confidence = 1.0
            else:
                classification, confidence = classify_entry(entry)
            
            # Interactive mode
            if args.interactive:
                print(f"\n{'='*60}")
                print(f"Content: {entry['content'][:200]}...")
                print(f"Suggested: {classification} ({confidence:.0%} confidence)")
                choice = input("[E]pisode [N]ote [B]elief [R]aw [S]kip: ").strip().lower()
                
                type_map = {"e": "episode", "n": "note", "b": "belief", "r": "raw", "s": "skip"}
                classification = type_map.get(choice, classification)
                
                if classification == "skip":
                    skipped += 1
                    continue
            
            # Dry-run vs execute
            if args.execute:
                import_entry(k, entry, classification, source=args.source or file_path.name)
                imported[classification] += 1
            else:
                # Preview mode
                print(f"  [{classification:8}] {entry['content'][:60]}...")
            
            total_entries += 1
    
    # Summary
    print()
    if args.execute:
        print(f"✓ Imported {total_entries} entries:")
        for mem_type, count in imported.items():
            if count > 0:
                print(f"  {mem_type}: {count}")
        if skipped > 0:
            print(f"  skipped: {skipped}")
    else:
        print(f"DRY RUN: Would import {total_entries} entries")
        print("Run with --execute to actually import")
```

### File Structure

```
kernle/cli/commands/
├── __init__.py
├── import_cmd.py      # NEW - import command implementation
├── parsers/           # NEW - format-specific parsers
│   ├── __init__.py
│   ├── base.py        # Base parser interface
│   ├── memory_md.py   # MEMORY.md format parser
│   ├── heuristic.py   # Heuristic-based parser
│   └── raw.py         # Everything-as-raw parser
├── ...
```

---

## Migration from MEMORY.md Example

**Input (MEMORY.md):**
```markdown
## User Preferences
- Dark mode preferred
- Vim keybindings
- Timezone: America/Los_Angeles

## Project Context
Working on Kernle - a stratified memory system for AI agents.
Backend deployed on Railway, frontend on Vercel.

## Lessons Learned
### 2025-01-26: Railway Webhook Issue
Webhooks broke after GitHub re-auth. Need to disconnect/reconnect 
in Railway dashboard. Always check webhook status after auth changes.

## Decisions
- 2025-01-15: SQLite for local storage (simpler than Postgres)
- 2025-01-20: Use uv instead of pip for package management
```

**Output (with `--execute`):**
```
✓ Imported 6 entries:
  note: 4
    - "Dark mode preferred" (preference)
    - "Vim keybindings" (preference)  
    - "Timezone: America/Los_Angeles" (preference)
    - "Working on Kernle..." (context)
  episode: 1
    - "Railway Webhook Issue" → lesson: "Always check webhook status..."
  note: 2 (decisions)
    - "SQLite for local storage" (decision, 2025-01-15)
    - "Use uv instead of pip" (decision, 2025-01-20)
```

---

## Next Steps

1. **Implement base parser infrastructure** (Option 1 heuristics)
2. **Add MEMORY.md format parser** (most common use case)
3. **Write tests** for parser edge cases
4. **Add to CLI argparse** in `__main__.py`
5. **Document** in `docs/IMPORT.md`

---

## Questions for Review

1. Should `--execute` be required, or should we default to importing?
2. How to handle files with mixed structured/unstructured content?
3. Should we support importing from URLs (fetch MEMORY.md from GitHub)?
4. Priority: interactive mode vs batch mode for v1?
