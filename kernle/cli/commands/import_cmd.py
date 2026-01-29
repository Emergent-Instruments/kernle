"""Import command for migrating flat files to Kernle."""

import re
from pathlib import Path
from typing import TYPE_CHECKING, List, Dict, Any, Optional

if TYPE_CHECKING:
    import argparse
    from kernle import Kernle


def cmd_import(args: "argparse.Namespace", k: "Kernle") -> None:
    """Import a markdown file into Kernle memory layers."""
    file_path = Path(args.file).expanduser()
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return
    
    if not file_path.suffix.lower() in ('.md', '.markdown', '.txt'):
        print(f"⚠️  Warning: File is not markdown ({file_path.suffix})")
    
    content = file_path.read_text()
    
    dry_run = getattr(args, 'dry_run', False)
    interactive = getattr(args, 'interactive', False)
    target_layer = getattr(args, 'layer', None)
    
    # Parse the content
    items = _parse_markdown(content)
    
    if not items:
        print("No importable content found in file")
        print("\nExpected formats:")
        print("  ## Episodes / ## Lessons - for episode entries")
        print("  ## Decisions / ## Notes - for note entries")
        print("  ## Beliefs - for belief entries")
        print("  ## Raw / ## Thoughts - for raw entries")
        print("  Freeform paragraphs - imported as raw entries")
        return
    
    # If layer specified, override detected types
    if target_layer:
        for item in items:
            item['type'] = target_layer
    
    # Show what we found
    type_counts = {}
    for item in items:
        t = item['type']
        type_counts[t] = type_counts.get(t, 0) + 1
    
    print(f"Found {len(items)} items to import:")
    for t, count in sorted(type_counts.items()):
        print(f"  {t}: {count}")
    print()
    
    if dry_run:
        print("=== DRY RUN (no changes made) ===\n")
        for i, item in enumerate(items, 1):
            _preview_item(i, item)
        return
    
    if interactive:
        items = _interactive_import(items, k)
    else:
        # Batch import
        _batch_import(items, k)


def _parse_markdown(content: str) -> List[Dict[str, Any]]:
    """Parse markdown content into importable items.
    
    Detects sections like:
    - ## Episodes, ## Lessons -> episode
    - ## Decisions, ## Notes, ## Insights -> note
    - ## Beliefs -> belief
    - ## Raw, ## Thoughts, ## Scratch -> raw
    - Unstructured text -> raw
    """
    items = []
    
    # Split into sections by ## headers
    sections = re.split(r'^## (.+)$', content, flags=re.MULTILINE)
    
    # First section (before any ##) is preamble
    if sections[0].strip():
        # Check if it has bullet points or paragraphs
        preamble = sections[0].strip()
        for para in _split_paragraphs(preamble):
            if para.strip():
                items.append({
                    'type': 'raw',
                    'content': para.strip(),
                    'source': 'preamble'
                })
    
    # Process header sections
    for i in range(1, len(sections), 2):
        if i + 1 >= len(sections):
            break
            
        header = sections[i].strip().lower()
        section_content = sections[i + 1].strip()
        
        if not section_content:
            continue
        
        # Determine type from header
        if any(h in header for h in ['episode', 'lesson', 'experience', 'event']):
            items.extend(_parse_episodes(section_content))
        elif any(h in header for h in ['decision', 'note', 'insight', 'observation']):
            items.extend(_parse_notes(section_content, header))
        elif 'belief' in header:
            items.extend(_parse_beliefs(section_content))
        elif any(h in header for h in ['value', 'principle']):
            items.extend(_parse_values(section_content))
        elif any(h in header for h in ['goal', 'objective', 'todo', 'task']):
            items.extend(_parse_goals(section_content))
        elif any(h in header for h in ['raw', 'thought', 'scratch', 'draft', 'idea']):
            items.extend(_parse_raw(section_content))
        else:
            # Unknown section - treat as raw
            items.extend(_parse_raw(section_content))
    
    return items


def _split_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs."""
    return [p.strip() for p in re.split(r'\n\n+', text) if p.strip()]


def _parse_episodes(content: str) -> List[Dict[str, Any]]:
    """Parse episode entries from section content."""
    items = []
    
    # Look for bullet points or numbered items
    entries = re.split(r'^[-*•]\s+|^\d+\.\s+', content, flags=re.MULTILINE)
    
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        
        # Try to extract lesson (after → or "Lesson:")
        lesson = None
        if '→' in entry:
            parts = entry.split('→', 1)
            entry = parts[0].strip()
            lesson = parts[1].strip()
        elif 'lesson:' in entry.lower():
            match = re.search(r'lesson:\s*(.+)', entry, re.IGNORECASE)
            if match:
                lesson = match.group(1).strip()
                entry = re.sub(r'lesson:\s*.+', '', entry, flags=re.IGNORECASE).strip()
        
        items.append({
            'type': 'episode',
            'objective': entry[:200] if len(entry) > 200 else entry,
            'outcome': entry,
            'lesson': lesson,
            'source': 'episodes section'
        })
    
    return items


def _parse_notes(content: str, header: str) -> List[Dict[str, Any]]:
    """Parse note entries from section content."""
    items = []
    
    # Determine note type from header
    if 'decision' in header:
        note_type = 'decision'
    elif 'insight' in header:
        note_type = 'insight'
    elif 'observation' in header:
        note_type = 'observation'
    else:
        note_type = 'note'
    
    # Split by bullets or paragraphs
    entries = re.split(r'^[-*•]\s+|^\d+\.\s+', content, flags=re.MULTILINE)
    
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        
        items.append({
            'type': 'note',
            'content': entry,
            'note_type': note_type,
            'source': f'{header} section'
        })
    
    return items


def _parse_beliefs(content: str) -> List[Dict[str, Any]]:
    """Parse belief entries from section content."""
    items = []
    
    entries = re.split(r'^[-*•]\s+|^\d+\.\s+', content, flags=re.MULTILINE)
    
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        
        # Try to extract confidence (e.g., "(80%)" or "[0.8]")
        confidence = 0.7  # default
        conf_match = re.search(r'\((\d+)%\)|\[(\d*\.?\d+)\]', entry)
        if conf_match:
            if conf_match.group(1):
                confidence = int(conf_match.group(1)) / 100
            elif conf_match.group(2):
                confidence = float(conf_match.group(2))
            entry = re.sub(r'\(\d+%\)|\[\d*\.?\d+\]', '', entry).strip()
        
        items.append({
            'type': 'belief',
            'statement': entry,
            'confidence': confidence,
            'source': 'beliefs section'
        })
    
    return items


def _parse_values(content: str) -> List[Dict[str, Any]]:
    """Parse value entries from section content."""
    items = []
    
    entries = re.split(r'^[-*•]\s+|^\d+\.\s+', content, flags=re.MULTILINE)
    
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        
        # Check for name: description format
        if ':' in entry:
            name, desc = entry.split(':', 1)
            name = name.strip()
            desc = desc.strip()
        else:
            name = entry[:50]
            desc = entry
        
        items.append({
            'type': 'value',
            'name': name,
            'description': desc,
            'source': 'values section'
        })
    
    return items


def _parse_goals(content: str) -> List[Dict[str, Any]]:
    """Parse goal entries from section content."""
    items = []
    
    entries = re.split(r'^[-*•]\s+|^\d+\.\s+', content, flags=re.MULTILINE)
    
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        
        # Check for [done] or [x] markers
        status = 'active'
        if re.search(r'\[x\]|\[done\]|\[complete\]', entry, re.IGNORECASE):
            status = 'completed'
            entry = re.sub(r'\[x\]|\[done\]|\[complete\]', '', entry, flags=re.IGNORECASE).strip()
        
        items.append({
            'type': 'goal',
            'description': entry,
            'status': status,
            'source': 'goals section'
        })
    
    return items


def _parse_raw(content: str) -> List[Dict[str, Any]]:
    """Parse raw entries from section content."""
    items = []
    
    # Check for bullet points first
    if re.search(r'^[-*•]\s+', content, flags=re.MULTILINE):
        entries = re.split(r'^[-*•]\s+', content, flags=re.MULTILINE)
    else:
        entries = _split_paragraphs(content)
    
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        
        items.append({
            'type': 'raw',
            'content': entry,
            'source': 'raw section'
        })
    
    return items


def _preview_item(index: int, item: Dict[str, Any]) -> None:
    """Print preview of an item."""
    t = item['type']
    content = item.get('content') or item.get('objective') or item.get('statement') or item.get('description', '')
    preview = content[:80] + '...' if len(content) > 80 else content
    
    print(f"{index}. [{t}] {preview}")
    
    if item.get('lesson'):
        print(f"   → Lesson: {item['lesson'][:60]}")
    if item.get('note_type'):
        print(f"   Type: {item['note_type']}")
    if item.get('confidence'):
        print(f"   Confidence: {item['confidence']:.0%}")


def _interactive_import(items: List[Dict[str, Any]], k: "Kernle") -> List[Dict[str, Any]]:
    """Interactive import with user confirmation for each item."""
    imported = []
    
    print("Interactive mode: [y]es / [n]o / [e]dit / [s]kip all / [a]ccept all\n")
    
    accept_all = False
    
    for i, item in enumerate(items, 1):
        if accept_all:
            _import_item(item, k)
            imported.append(item)
            continue
        
        _preview_item(i, item)
        
        choice = input("Import? [y/n/e/s/a]: ").strip().lower()
        
        if choice == 'a':
            accept_all = True
            _import_item(item, k)
            imported.append(item)
        elif choice == 'y':
            _import_item(item, k)
            imported.append(item)
        elif choice == 's':
            print(f"Skipping remaining {len(items) - i + 1} items")
            break
        elif choice == 'e':
            item = _edit_item(item)
            _import_item(item, k)
            imported.append(item)
        else:
            print("  Skipped")
        
        print()
    
    print(f"\n✓ Imported {len(imported)} of {len(items)} items")
    return imported


def _edit_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Allow user to edit an item before import."""
    t = item['type']
    
    if t == 'episode':
        new = input(f"  Objective [{item.get('objective', '')[:50]}]: ").strip()
        if new:
            item['objective'] = new
        new = input(f"  Lesson [{item.get('lesson', '')}]: ").strip()
        if new:
            item['lesson'] = new
    elif t == 'note':
        new = input(f"  Content [{item.get('content', '')[:50]}]: ").strip()
        if new:
            item['content'] = new
        new = input(f"  Type [{item.get('note_type', 'note')}]: ").strip()
        if new:
            item['note_type'] = new
    elif t == 'belief':
        new = input(f"  Statement [{item.get('statement', '')[:50]}]: ").strip()
        if new:
            item['statement'] = new
        new = input(f"  Confidence [{item.get('confidence', 0.7):.0%}]: ").strip()
        if new:
            try:
                item['confidence'] = float(new.replace('%', '')) / 100 if '%' in new else float(new)
            except ValueError:
                pass
    elif t == 'raw':
        new = input(f"  Content [{item.get('content', '')[:50]}]: ").strip()
        if new:
            item['content'] = new
    
    return item


def _batch_import(items: List[Dict[str, Any]], k: "Kernle") -> None:
    """Batch import all items."""
    success = 0
    errors = []
    
    for item in items:
        try:
            _import_item(item, k)
            success += 1
        except Exception as e:
            errors.append(f"{item['type']}: {str(e)[:50]}")
    
    print(f"✓ Imported {success} items")
    if errors:
        print(f"⚠️  {len(errors)} errors:")
        for err in errors[:5]:
            print(f"   {err}")


def _import_item(item: Dict[str, Any], k: "Kernle") -> None:
    """Import a single item into Kernle."""
    t = item['type']
    
    if t == 'episode':
        lessons = [item['lesson']] if item.get('lesson') else None
        k.episode(
            objective=item['objective'],
            outcome=item.get('outcome', item['objective']),
            lessons=lessons
        )
    elif t == 'note':
        k.note(
            content=item['content'],
            note_type=item.get('note_type', 'note')
        )
    elif t == 'belief':
        # Use the belief API
        k._storage.save_belief(
            agent_id=k.agent_id,
            statement=item['statement'],
            confidence=item.get('confidence', 0.7)
        )
    elif t == 'value':
        k.value(
            name=item['name'],
            description=item.get('description', item['name'])
        )
    elif t == 'goal':
        k.goal(
            description=item['description'],
            status=item.get('status', 'active')
        )
    elif t == 'raw':
        k.raw(item['content'])
