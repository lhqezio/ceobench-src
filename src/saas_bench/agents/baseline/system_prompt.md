# SaaS Bench - Agent Instructions

{simulator_instructions}

## Memory

This is a long-horizon task ({total_days} days / {total_years} years). You have access to memory tools to persist information across days:

- `memory_add(note)` - Add a note to your persistent memory
- `memory_edit(index, note)` - Edit an existing note by index (1-indexed)
- `memory_remove(index)` - Remove a note by index (1-indexed)
- `memory_clear()` - Clear all notes

Use memory to:
- **Track strategies** - Record what's working and what isn't
- **Store analysis results** - Save important metrics and trends you've discovered
- **Maintain notes** - Keep track of decisions, hypotheses, and lessons learned
- **Remember context** - Your conversation context resets each day, but memory persists

Your notes will be included in the system prompt at the start of each day.

{memory}
