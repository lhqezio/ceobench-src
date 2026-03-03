#!/usr/bin/env python3
"""Generate PDF report from Claude Code agent run logs.

Merges environment events and agent conversation logs into a single,
easy-to-read report.

Usage:
    python generate_report.py <workspace_dir> [--output report.pdf] [--up-to-day N]
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class MergedEvent:
    """A single event in the merged timeline."""
    timestamp: str
    day: int
    source: str  # 'env' or 'agent'
    event_type: str
    summary: str
    details: Optional[Dict[str, Any]] = None


def load_env_events(logs_dir: Path, run_id: str) -> List[Dict]:
    """Load environment events from run JSON."""
    log_file = logs_dir / f"run_{run_id}.json"
    if not log_file.exists():
        return []

    with open(log_file) as f:
        data = json.load(f)

    return data.get('events', [])


def load_agent_events(logs_dir: Path, run_id: str) -> List[Dict]:
    """Load agent conversation events from JSONL."""
    log_file = logs_dir / f"agent_conversation_{run_id}.jsonl"
    if not log_file.exists():
        return []

    events = []
    with open(log_file) as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))

    return events


def load_rationales(logs_dir: Path, run_id: str) -> List[Dict]:
    """Load agent rationales."""
    rationale_file = logs_dir / f"rationales_{run_id}.json"
    if not rationale_file.exists():
        return []

    with open(rationale_file) as f:
        return json.load(f)


def merge_events(env_events: List[Dict], agent_events: List[Dict],
                 rationales: List[Dict], up_to_day: Optional[int] = None) -> List[MergedEvent]:
    """Merge all events into a single timeline."""
    merged = []

    # Process env events
    for event in env_events:
        day = event.get('day', 0)
        if up_to_day is not None and day > up_to_day:
            continue

        event_type = event.get('event_type', 'unknown')

        # Create summary based on event type
        if event_type == 'daily_state':
            details = event.get('details', {})
            summary = f"Day {day} End: Cash=${details.get('cash', 0):,.0f}, MRR=${details.get('mrr', 0):,.0f}, Subs={details.get('subscribers', 0)}"
        elif event_type == 'shock':
            summary = f"Shock: {event.get('category', 'unknown')} - {event.get('details', {}).get('description', '')}"
        elif event_type == 'customer_signup':
            details = event.get('details', {})
            summary = f"New customer: Group {details.get('group_id', '?')}, Plan {details.get('plan', '?')}"
        elif event_type == 'customer_churn':
            details = event.get('details', {})
            summary = f"Churn: Customer {details.get('customer_id', '?')}, Reason: {details.get('reason', '?')}"
        elif event_type == 'tool_call':
            details = event.get('details', {})
            summary = f"Tool: {details.get('tool_name', '?')}"
        else:
            summary = f"{event_type}: {event.get('category', '')}"

        merged.append(MergedEvent(
            timestamp=event.get('timestamp', ''),
            day=day,
            source='env',
            event_type=event_type,
            summary=summary,
            details=event.get('details')
        ))

    # Process agent turns
    for turn in agent_events:
        day = turn.get('day', 0)
        if up_to_day is not None and day > up_to_day:
            continue

        output = turn.get('response', {}).get('output', '')[:200]
        tool_calls = turn.get('response', {}).get('tool_calls', [])

        if tool_calls:
            tools_summary = ', '.join([tc.get('name', '?') for tc in tool_calls[:3]])
            summary = f"Agent Turn {turn.get('turn', 0)}: Called {tools_summary}"
        else:
            summary = f"Agent Turn {turn.get('turn', 0)}: {output[:100]}..."

        merged.append(MergedEvent(
            timestamp=turn.get('timestamp', ''),
            day=day,
            source='agent',
            event_type='agent_turn',
            summary=summary,
            details={
                'turn': turn.get('turn'),
                'tool_calls': tool_calls,
                'output_preview': output
            }
        ))

    # Process rationales
    for rat in rationales:
        day = rat.get('day', 0)
        if up_to_day is not None and day > up_to_day:
            continue

        merged.append(MergedEvent(
            timestamp=rat.get('timestamp', ''),
            day=day,
            source='agent',
            event_type='rationale',
            summary=f"Rationale: {rat.get('rationale', '')[:100]}...",
            details={'full_rationale': rat.get('rationale'), 'context': rat.get('context')}
        ))

    # Sort by timestamp
    merged.sort(key=lambda x: x.timestamp)

    return merged


def generate_markdown_report(workspace_dir: Path, up_to_day: Optional[int] = None) -> str:
    """Generate a Markdown report from the logs."""
    logs_dir = workspace_dir / "logs"

    # Find run_id from log files
    run_files = list(logs_dir.glob("run_*.json"))
    if not run_files:
        return "# Error\nNo run log files found."

    run_id = run_files[0].stem.replace("run_", "")

    # Load all events
    env_events = load_env_events(logs_dir, run_id)
    agent_events = load_agent_events(logs_dir, run_id)
    rationales = load_rationales(logs_dir, run_id)

    # Load metadata
    with open(run_files[0]) as f:
        run_data = json.load(f)
    metadata = run_data.get('metadata', {})

    # Merge events
    merged = merge_events(env_events, agent_events, rationales, up_to_day)

    # Find final state
    daily_states = [e for e in merged if e.event_type == 'daily_state']
    final_state = daily_states[-1] if daily_states else None

    # Generate report
    lines = [
        "# SaaS Bench Agent Run Report",
        "",
        "## Run Information",
        "",
        f"- **Run ID**: {run_id}",
        f"- **Scenario**: {metadata.get('scenario', 'unknown')}",
        f"- **Seed**: {metadata.get('seed', 'unknown')}",
        f"- **Start Time**: {metadata.get('start_time', 'unknown')}",
        f"- **Report Generated**: {datetime.now().isoformat()}",
        "",
    ]

    if up_to_day:
        lines.append(f"- **Report covers**: Days 1-{up_to_day}")
        lines.append("")

    # Summary
    lines.extend([
        "## Summary",
        "",
    ])

    if final_state and final_state.details:
        d = final_state.details
        lines.extend([
            f"- **Final Day**: {final_state.day}",
            f"- **Final Cash**: ${d.get('cash', 0):,.2f}",
            f"- **Final MRR**: ${d.get('mrr', 0):,.2f}",
            f"- **Subscribers**: {d.get('subscribers', 0)}",
            "",
        ])

    # Statistics
    total_turns = len([e for e in merged if e.event_type == 'agent_turn'])
    total_rationales = len([e for e in merged if e.event_type == 'rationale'])
    total_shocks = len([e for e in merged if e.event_type == 'shock'])
    total_signups = len([e for e in merged if e.event_type == 'customer_signup'])
    total_churns = len([e for e in merged if e.event_type == 'customer_churn'])

    lines.extend([
        "## Statistics",
        "",
        f"- Agent turns: {total_turns}",
        f"- Rationales logged: {total_rationales}",
        f"- Shocks encountered: {total_shocks}",
        f"- Customer signups: {total_signups}",
        f"- Customer churns: {total_churns}",
        "",
    ])

    # Day-by-day breakdown
    lines.extend([
        "## Day-by-Day Timeline",
        "",
    ])

    current_day = 0
    for event in merged:
        if event.day != current_day:
            current_day = event.day
            lines.append(f"### Day {current_day}")
            lines.append("")

        # Format event
        source_icon = "🤖" if event.source == 'agent' else "🌍"
        lines.append(f"- {source_icon} **{event.event_type}**: {event.summary}")

    lines.append("")

    # Agent Rationales Section
    if rationales:
        lines.extend([
            "## Agent Rationales",
            "",
        ])

        for rat in rationales:
            if up_to_day is not None and rat.get('day', 0) > up_to_day:
                continue
            lines.extend([
                f"### Day {rat.get('day', 0)} - {rat.get('context', 'General')}",
                "",
                rat.get('rationale', ''),
                "",
            ])

    return "\n".join(lines)


def generate_pdf_report(workspace_dir: Path, output_path: Path, up_to_day: Optional[int] = None):
    """Generate a PDF report from the logs."""
    try:
        import markdown
        from weasyprint import HTML, CSS
    except ImportError:
        print("Warning: weasyprint not available. Generating Markdown only.")
        md_content = generate_markdown_report(workspace_dir, up_to_day)
        md_path = output_path.with_suffix('.md')
        md_path.write_text(md_content)
        print(f"Markdown report saved to: {md_path}")
        return md_path

    # Generate markdown
    md_content = generate_markdown_report(workspace_dir, up_to_day)

    # Convert to HTML
    html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])

    # Add styling
    styled_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                line-height: 1.6;
            }}
            h1 {{ color: #333; border-bottom: 2px solid #333; padding-bottom: 10px; }}
            h2 {{ color: #555; border-bottom: 1px solid #ddd; padding-bottom: 5px; margin-top: 30px; }}
            h3 {{ color: #666; margin-top: 20px; }}
            code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
            pre {{ background: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }}
            ul {{ padding-left: 20px; }}
            li {{ margin-bottom: 5px; }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    # Generate PDF
    HTML(string=styled_html).write_pdf(output_path)
    print(f"PDF report saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate report from Claude Code agent run")
    parser.add_argument("workspace_dir", type=Path, help="Path to agent workspace directory")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output file path")
    parser.add_argument("--up-to-day", "-d", type=int, default=None, help="Generate report up to this day")
    parser.add_argument("--format", "-f", choices=['pdf', 'md'], default='md', help="Output format")

    args = parser.parse_args()

    workspace_dir = args.workspace_dir
    if not workspace_dir.exists():
        print(f"Error: Workspace directory not found: {workspace_dir}")
        return 1

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        suffix = '.pdf' if args.format == 'pdf' else '.md'
        day_suffix = f"_day{args.up_to_day}" if args.up_to_day else ""
        output_path = workspace_dir / f"report{day_suffix}{suffix}"

    if args.format == 'pdf':
        generate_pdf_report(workspace_dir, output_path, args.up_to_day)
    else:
        md_content = generate_markdown_report(workspace_dir, args.up_to_day)
        output_path.write_text(md_content)
        print(f"Markdown report saved to: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
