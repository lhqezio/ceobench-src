#!/usr/bin/env python3
"""Generate chronological PDF report from tool calls log - FULL OUTPUT VERSION with proper line wrapping."""

import json
import textwrap
from pathlib import Path
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT

def wrap_text(text, width=95):
    """Wrap text to specified character width with proper line breaks."""
    if text is None:
        return ""
    text = str(text)
    lines = text.split('\n')
    wrapped_lines = []
    for line in lines:
        if len(line) > width:
            wrapped = textwrap.fill(line, width=width)
            wrapped_lines.append(wrapped)
        else:
            wrapped_lines.append(line)
    return '\n'.join(wrapped_lines)

def escape_xml(text):
    """Escape XML special characters for reportlab."""
    if text is None:
        return ""
    text = str(text)
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')
    return text

def main():
    run_dir = Path(__file__).parent
    logs_dir = run_dir / "logs"

    # Find tool calls log
    tool_calls_files = list(logs_dir.glob("tool_calls_*.jsonl"))
    if not tool_calls_files:
        print("No tool_calls log found!")
        return

    tool_calls_file = tool_calls_files[0]

    # Load all tool calls
    tool_calls = []
    with open(tool_calls_file, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    tool_calls.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    print(f"Loaded {len(tool_calls)} tool calls")

    # Load file changes if available
    file_changes_files = list(logs_dir.glob("file_changes_*.jsonl"))
    file_changes = []
    if file_changes_files:
        with open(file_changes_files[0], 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        file_changes.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        print(f"Loaded {len(file_changes)} file changes")

    # Load rationales if available
    rationales_files = list(logs_dir.glob("rationales_*.json"))
    rationales = {}
    if rationales_files:
        with open(rationales_files[0], 'r') as f:
            rationales = json.load(f)

    # Get run metadata
    run_id = run_dir.name.replace("run_", "")

    # Get state info
    state_file = run_dir / ".mcp_state.json"
    current_day = 0
    if state_file.exists():
        with open(state_file, 'r') as f:
            state = json.load(f)
            current_day = state.get("current_day", 0)

    # Create PDF
    output_path = run_dir / f"report_{run_id}.pdf"
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=0.5*inch,
        rightMargin=0.5*inch,
        topMargin=0.5*inch,
        bottomMargin=0.5*inch
    )

    # Define styles
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'Title',
        parent=styles['Title'],
        fontSize=18,
        spaceAfter=12
    )

    day_header_style = ParagraphStyle(
        'DayHeader',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#1a237e'),  # dark blue
        spaceAfter=8,
        spaceBefore=16
    )

    tool_name_style = ParagraphStyle(
        'ToolName',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#1b5e20'),  # dark green
        fontName='Helvetica-Bold',
        spaceAfter=4
    )

    timestamp_style = ParagraphStyle(
        'Timestamp',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.gray,
        spaceAfter=2
    )

    code_style = ParagraphStyle(
        'Code',
        parent=styles['Code'],
        fontSize=7,
        fontName='Courier',
        leftIndent=10,
        backColor=colors.HexColor('#f5f5f5'),
        spaceAfter=8
    )

    error_style = ParagraphStyle(
        'Error',
        parent=styles['Code'],
        fontSize=7,
        fontName='Courier',
        leftIndent=10,
        textColor=colors.red,
        backColor=colors.HexColor('#ffebee'),
        spaceAfter=8
    )

    rationale_style = ParagraphStyle(
        'Rationale',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.HexColor('#37474f'),
        leftIndent=10,
        spaceAfter=6,
        fontName='Helvetica-Oblique'
    )

    file_change_style = ParagraphStyle(
        'FileChange',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#6a1b9a'),  # purple
        fontName='Helvetica-Bold',
        spaceAfter=4
    )

    file_content_style = ParagraphStyle(
        'FileContent',
        parent=styles['Code'],
        fontSize=7,
        fontName='Courier',
        leftIndent=10,
        backColor=colors.HexColor('#f3e5f5'),  # light purple
        spaceAfter=8
    )

    # Build content
    content = []

    # Title page
    content.append(Paragraph("SaaS Bench Agent Report", title_style))
    content.append(Spacer(1, 12))
    content.append(Paragraph(f"<b>Run ID:</b> {run_id}", styles['Normal']))
    content.append(Paragraph(f"<b>Model:</b> claude-sonnet-4-20250514", styles['Normal']))
    content.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    content.append(Paragraph(f"<b>Total Tool Calls:</b> {len(tool_calls)}", styles['Normal']))
    content.append(Paragraph(f"<b>Days Simulated:</b> {current_day}", styles['Normal']))
    content.append(Spacer(1, 24))
    content.append(PageBreak())

    # Group tool calls by day
    calls_by_day = {}
    for call in tool_calls:
        day = call.get("day", 0)
        if day not in calls_by_day:
            calls_by_day[day] = []
        calls_by_day[day].append(call)

    # Group file changes by day
    changes_by_day = {}
    for change in file_changes:
        day = change.get("day", 0)
        if day not in changes_by_day:
            changes_by_day[day] = []
        changes_by_day[day].append(change)

    # Add tool calls chronologically by day
    for day in sorted(calls_by_day.keys()):
        content.append(Paragraph(f"Day {day}", day_header_style))

        # Check for rationale for this day
        day_key = f"day_{day}"
        if day_key in rationales:
            rationale_text = rationales[day_key].get("rationale", "")
            if rationale_text:
                wrapped_rationale = wrap_text(rationale_text, width=100)
                content.append(Paragraph(f"<i>Rationale: {escape_xml(wrapped_rationale[:500])}{'...' if len(wrapped_rationale) > 500 else ''}</i>", rationale_style))

        for call in calls_by_day[day]:
            tool_name = call.get("tool", "unknown")
            timestamp = call.get("timestamp", "")

            content.append(Paragraph(f"🔧 {tool_name}", tool_name_style))
            if timestamp:
                content.append(Paragraph(timestamp, timestamp_style))

            # Arguments
            args = call.get("arguments", {})
            if args:
                # Special handling for 'code' argument - show it as actual code
                if "code" in args and isinstance(args["code"], str):
                    # Show code directly with proper newlines
                    code_text = args["code"]
                    # Show other args if any
                    other_args = {k: v for k, v in args.items() if k != "code"}
                    if other_args:
                        other_text = json.dumps(other_args, indent=2, default=str)
                        wrapped_other = wrap_text(other_text, width=95)
                        content.append(Paragraph("<b>Arguments:</b>", styles['Normal']))
                        content.append(Preformatted(wrapped_other, code_style))
                    content.append(Paragraph("<b>Code:</b>", styles['Normal']))
                    wrapped_code = wrap_text(code_text, width=95)
                    content.append(Preformatted(wrapped_code, code_style))
                else:
                    args_text = json.dumps(args, indent=2, default=str)
                    wrapped_args = wrap_text(args_text, width=95)
                    content.append(Paragraph("<b>Arguments:</b>", styles['Normal']))
                    content.append(Preformatted(wrapped_args, code_style))

            # Result
            result = call.get("result", {})
            if result:
                result_text = json.dumps(result, indent=2, default=str) if isinstance(result, dict) else str(result)
                wrapped_result = wrap_text(result_text, width=95)

                # Check if error
                is_error = "error" in str(result).lower()
                style = error_style if is_error else code_style

                content.append(Paragraph("<b>Result:</b>", styles['Normal']))
                # Truncate very long results
                if len(wrapped_result) > 2000:
                    wrapped_result = wrapped_result[:2000] + "\n... [truncated]"
                content.append(Preformatted(wrapped_result, style))

            content.append(Spacer(1, 8))

        # Add file changes for this day
        if day in changes_by_day:
            content.append(Paragraph("📁 File Changes:", tool_name_style))
            for change in changes_by_day[day]:
                change_type = change.get("type", "unknown")
                path = change.get("path", "unknown")

                if change_type == "created":
                    icon = "✨"
                    label = "Created"
                elif change_type == "modified":
                    icon = "📝"
                    label = "Modified"
                elif change_type == "deleted":
                    icon = "🗑️"
                    label = "Deleted"
                else:
                    icon = "❓"
                    label = change_type

                content.append(Paragraph(f"{icon} {label}: {escape_xml(path)}", file_change_style))

                # Show content preview for created/modified files
                content_preview = change.get("content_preview", "")
                if content_preview:
                    wrapped_content = wrap_text(content_preview, width=95)
                    # Truncate if too long
                    if len(wrapped_content) > 3000:
                        wrapped_content = wrapped_content[:3000] + "\n... [truncated]"
                    content.append(Preformatted(wrapped_content, file_content_style))

            content.append(Spacer(1, 8))

    # Build PDF
    doc.build(content)
    print(f"Report generated: {output_path}")
    return output_path

if __name__ == "__main__":
    main()
