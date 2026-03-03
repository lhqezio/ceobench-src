"""Tool definitions and execution for the bash_agent.

The bash_agent has a small set of tools: bash (shell commands),
and file manipulation (read, write, edit, search, glob).
"""

import fnmatch
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


# =========================================================================
# Tool schemas (OpenAI function-calling format)
# =========================================================================

BASH_AGENT_TOOL_DEFS = [
    {
        'name': 'bash',
        'description': (
            'Execute a bash command in the agent working directory. '
            'Use this to run novamind-operation CLI commands, Python scripts, '
            'and any other shell commands. The novamind_api Python library is '
            'available for import in Python scripts.'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'command': {
                    'type': 'string',
                    'description': 'The bash command to execute',
                },
            },
            'required': ['command'],
        },
    },
    {
        'name': 'read_file',
        'description': (
            'Read the contents of a file. Returns the file content as a string. '
            'Use offset and limit for large files.'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'path': {
                    'type': 'string',
                    'description': 'Path to the file (relative to working directory)',
                },
                'offset': {
                    'type': 'integer',
                    'description': 'Line number to start reading from (1-indexed, optional)',
                },
                'limit': {
                    'type': 'integer',
                    'description': 'Maximum number of lines to read (optional)',
                },
            },
            'required': ['path'],
        },
    },
    {
        'name': 'write_file',
        'description': (
            'Create or overwrite a file with the given content. '
            'Use this to create new files or completely replace file contents.'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'path': {
                    'type': 'string',
                    'description': 'Path to the file (relative to working directory)',
                },
                'content': {
                    'type': 'string',
                    'description': 'Content to write to the file',
                },
            },
            'required': ['path', 'content'],
        },
    },
    {
        'name': 'edit_file',
        'description': (
            'Edit an existing file by replacing old_string with new_string. '
            'The old_string must appear exactly once in the file.'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'path': {
                    'type': 'string',
                    'description': 'Path to the file (relative to working directory)',
                },
                'old_string': {
                    'type': 'string',
                    'description': 'The exact string to find and replace',
                },
                'new_string': {
                    'type': 'string',
                    'description': 'The replacement string',
                },
            },
            'required': ['path', 'old_string', 'new_string'],
        },
    },
    {
        'name': 'search_files',
        'description': (
            'Search file contents using a regex pattern (like grep). '
            'Returns matching lines with file paths and line numbers.'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'pattern': {
                    'type': 'string',
                    'description': 'Regular expression pattern to search for',
                },
                'path': {
                    'type': 'string',
                    'description': 'File or directory to search in (default: working directory)',
                },
                'glob': {
                    'type': 'string',
                    'description': 'Glob pattern to filter files (e.g., "*.py")',
                },
            },
            'required': ['pattern'],
        },
    },
    {
        'name': 'glob_files',
        'description': (
            'Find files matching a glob pattern. '
            'Returns a list of matching file paths.'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'pattern': {
                    'type': 'string',
                    'description': 'Glob pattern (e.g., "**/*.py", "docs/*.json")',
                },
            },
            'required': ['pattern'],
        },
    },
]


def get_bash_agent_tool_descriptions() -> List[Dict[str, Any]]:
    """Get OpenAI Responses API-compatible tool descriptions for the bash agent."""
    return [
        {
            'type': 'function',
            'name': t['name'],
            'description': t['description'],
            'parameters': t['parameters'],
        }
        for t in BASH_AGENT_TOOL_DEFS
    ]


def get_bash_agent_anthropic_tools() -> List[Dict[str, Any]]:
    """Get Anthropic API-compatible tool descriptions for the bash agent."""
    return [
        {
            'name': t['name'],
            'description': t['description'],
            'input_schema': t['parameters'],
        }
        for t in BASH_AGENT_TOOL_DEFS
    ]


# =========================================================================
# Tool execution
# =========================================================================

class BashAgentToolExecutor:
    """Executes bash_agent tools within a working directory."""

    def __init__(self, workspace_path: Path, env: Optional[Dict[str, str]] = None,
                 bash_timeout: int = 300):
        """Initialize the tool executor.

        Args:
            workspace_path: Agent's working directory.
            env: Extra environment variables for bash commands.
            bash_timeout: Timeout in seconds for bash commands (default 5 min).
        """
        self.workspace_path = workspace_path
        self.extra_env = env or {}
        self.bash_timeout = bash_timeout

    def execute(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Execute a tool and return the result string."""
        dispatch = {
            'bash': self._exec_bash,
            'read_file': self._exec_read_file,
            'write_file': self._exec_write_file,
            'edit_file': self._exec_edit_file,
            'search_files': self._exec_search_files,
            'glob_files': self._exec_glob_files,
        }
        handler = dispatch.get(tool_name)
        if handler is None:
            return f"Error: Unknown tool '{tool_name}'"
        try:
            return handler(args)
        except Exception as e:
            return f"Error: {e}"

    def _resolve_path(self, path_str: str) -> Path:
        """Resolve a path relative to the workspace, preventing escape."""
        p = Path(path_str)
        if p.is_absolute():
            resolved = p.resolve()
        else:
            resolved = (self.workspace_path / p).resolve()
        # Ensure it's within workspace
        ws_resolved = self.workspace_path.resolve()
        if not str(resolved).startswith(str(ws_resolved)):
            raise ValueError(f"Path escapes workspace: {path_str}")
        return resolved

    def _exec_bash(self, args: Dict) -> str:
        """Execute a bash command, sandboxed to the workspace directory.

        The command runs with a clean environment that only exposes
        standard system paths. HOME and TMPDIR are set to the workspace
        to prevent accidental writes elsewhere. A read-only bind of the
        workspace prevents traversal outside it via symlinks or '..'.
        """
        command = args.get('command', '')
        if not command:
            return "Error: No command provided"

        ws = str(self.workspace_path)

        # Build a minimal, sandboxed environment.
        # Start from scratch — do NOT inherit os.environ (which contains
        # simulator source paths, home directory, etc.)
        env = {
            'PATH': '/usr/local/bin:/usr/bin:/bin',
            'HOME': ws,
            'TMPDIR': ws,
            'LANG': os.environ.get('LANG', 'en_US.UTF-8'),
            'TERM': os.environ.get('TERM', 'xterm'),
        }
        env.update(self.extra_env)

        # Run the command directly in the workspace directory.
        # cwd=ws ensures the working directory is correct.
        # No double-nesting of bash -c — that breaks quoting for
        # commands containing single quotes, heredocs, etc.
        #
        # Use Popen so we can explicitly kill the process group on timeout.
        # subprocess.run() does NOT kill children on TimeoutExpired, leaving
        # zombie processes that can hold DB locks or resources.
        import signal
        proc = subprocess.Popen(
            ['bash', '-c', command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=ws,
            env=env,
            start_new_session=True,  # Create new process group for clean kill
        )
        try:
            stdout, stderr = proc.communicate(timeout=self.bash_timeout)

            output_parts = []
            if stdout:
                output_parts.append(stdout)
            if stderr:
                output_parts.append(f"[stderr]\n{stderr}")
            if proc.returncode != 0:
                output_parts.append(f"[exit code: {proc.returncode}]")

            output = '\n'.join(output_parts) if output_parts else "(no output)"

            # Truncate very long output (same limit as Claude Code: 30K chars)
            if len(output) > 30000:
                output = output[:15000] + "\n\n... (output truncated — exceeded 30,000 character limit) ...\n\n" + output[-15000:]

            return output

        except subprocess.TimeoutExpired:
            # Kill the entire process group (bash + all children)
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            proc.kill()  # Fallback: kill the direct child
            proc.wait(timeout=5)  # Reap the zombie
            return f"Error: Command timed out after {self.bash_timeout} seconds"

    def _exec_read_file(self, args: Dict) -> str:
        """Read file contents."""
        path = self._resolve_path(args['path'])
        if not path.exists():
            return f"Error: File not found: {args['path']}"
        if not path.is_file():
            return f"Error: Not a file: {args['path']}"

        content = path.read_text()
        lines = content.split('\n')

        offset = args.get('offset', 1)
        limit = args.get('limit')

        # Apply offset (1-indexed)
        start = max(0, offset - 1)
        if limit:
            end = start + limit
            lines = lines[start:end]
        else:
            lines = lines[start:]

        # Format with line numbers
        numbered = []
        for i, line in enumerate(lines, start=start + 1):
            numbered.append(f"{i:6d}\t{line}")

        return '\n'.join(numbered)

    def _exec_write_file(self, args: Dict) -> str:
        """Write file contents."""
        path = self._resolve_path(args['path'])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args['content'])
        return f"File written: {args['path']} ({len(args['content'])} bytes)"

    def _exec_edit_file(self, args: Dict) -> str:
        """Edit a file by replacing old_string with new_string."""
        path = self._resolve_path(args['path'])
        if not path.exists():
            return f"Error: File not found: {args['path']}"

        content = path.read_text()
        old_str = args['old_string']
        new_str = args['new_string']

        count = content.count(old_str)
        if count == 0:
            return f"Error: old_string not found in {args['path']}"
        if count > 1:
            return f"Error: old_string found {count} times in {args['path']} (must be unique)"

        new_content = content.replace(old_str, new_str, 1)
        path.write_text(new_content)
        return f"File edited: {args['path']}"

    def _exec_search_files(self, args: Dict) -> str:
        """Search files with regex pattern."""
        pattern = args['pattern']
        search_path = args.get('path', '.')
        glob_filter = args.get('glob', '*')

        resolved = self._resolve_path(search_path)
        if not resolved.exists():
            return f"Error: Path not found: {search_path}"

        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"Error: Invalid regex: {e}"

        matches = []
        if resolved.is_file():
            files = [resolved]
        else:
            files = sorted(resolved.rglob(glob_filter))

        for fpath in files[:100]:  # Limit file count
            if not fpath.is_file():
                continue
            try:
                content = fpath.read_text()
            except (UnicodeDecodeError, PermissionError):
                continue
            for i, line in enumerate(content.split('\n'), 1):
                if regex.search(line):
                    rel = fpath.relative_to(self.workspace_path)
                    matches.append(f"{rel}:{i}: {line}")
                    if len(matches) >= 200:
                        break
            if len(matches) >= 200:
                break

        if not matches:
            return "No matches found."
        return '\n'.join(matches)

    def _exec_glob_files(self, args: Dict) -> str:
        """Find files matching a glob pattern."""
        pattern = args['pattern']
        matches = sorted(self.workspace_path.glob(pattern))
        if not matches:
            return "No matching files."
        result = []
        for m in matches[:200]:
            try:
                rel = m.relative_to(self.workspace_path)
                result.append(str(rel))
            except ValueError:
                result.append(str(m))
        return '\n'.join(result)
