#!/usr/bin/env python3
"""A coding agent in <500 lines. Stdlib only.

Speaks to any OpenAI-compatible /v1/chat/completions endpoint.
Usage: python agent.py [--yes] ["do the thing"]   (no prompt -> REPL)
"""
import fnmatch
import json
import os
import pathlib
import re
import subprocess
import sys
import urllib.error
import urllib.request

BASE_URL = os.environ.get("AGENT_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.environ.get("AGENT_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
MODEL = os.environ.get("AGENT_MODEL", "gpt-4o-mini")
ROOT = pathlib.Path.cwd()          # the agent may only touch files under here
MAX_TOOL_OUTPUT = 20_000           # chars; tool results are truncated to this
MAX_STEPS = 50                     # tool rounds per user turn
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv"}
AUTO_APPROVE = "--yes" in sys.argv or "-y" in sys.argv
GATED = {"write_file", "edit_file", "bash"}


def chat(messages: list, tools: list) -> dict:
    """One blocking round-trip. Returns the assistant message dict."""
    body = json.dumps({"model": MODEL, "messages": messages,
                       "tools": tools, "tool_choice": "auto"}).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions", data=body,
        headers={"Authorization": f"Bearer {API_KEY}",
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())["choices"][0]["message"]
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"API {e.code}: {e.read().decode()[:500]}")
    except OSError as e:               # DNS, TLS, timeout, connection reset
        raise RuntimeError(f"API unreachable: {e}")


# --- tools -----------------------------------------------------------------
# name -> (json schema shown to the model, python function that runs it)
TOOLS: dict = {}


def tool(desc, **params):
    """Register a function as a tool. params: name=(json_type, help, required)."""
    def deco(fn):
        TOOLS[fn.__name__] = ({
            "type": "function",
            "function": {
                "name": fn.__name__, "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": {k: {"type": t, "description": d}
                                   for k, (t, d, _) in params.items()},
                    "required": [k for k, (_, _, r) in params.items() if r]}}},
            fn)
        return fn
    return deco


def safe_path(p: str) -> pathlib.Path:
    """Resolve p under ROOT, refusing anything that escapes the workspace."""
    path = (ROOT / p).resolve()
    if path != ROOT and not path.is_relative_to(ROOT):
        raise ValueError(f"path escapes workspace: {p}")
    return path


def walk(base: pathlib.Path):
    """Every file under base, skipping noise directories."""
    for path in sorted(base.rglob("*")):
        if path.is_file() and not SKIP_DIRS & set(path.relative_to(ROOT).parts):
            yield path


@tool("Read a file. Returns its contents with 1-based line numbers.",
      path=("string", "file path relative to the workspace root", True))
def read_file(path):
    f = safe_path(path)
    if not f.is_file():
        return "error: not found"
    lines = f.read_text(errors="replace").splitlines()
    return "\n".join(f"{i + 1}: {line}" for i, line in enumerate(lines))


@tool("Create or overwrite a file with exact content.",
      path=("string", "file path relative to the workspace root", True),
      content=("string", "full file content", True))
def write_file(path, content):
    f = safe_path(path)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(content)
    return f"wrote {len(content.encode())} bytes to {path}"


@tool("Replace an exact substring in a file. `old` must appear exactly once.",
      path=("string", "file path relative to the workspace root", True),
      old=("string", "exact text to replace, with enough context to be unique",
           True),
      new=("string", "replacement text", True))
def edit_file(path, old, new):
    f = safe_path(path)
    if not f.is_file():
        return "error: not found"
    text = f.read_text()
    n = text.count(old)
    if n == 0:
        return "error: old text not found"
    if n > 1:
        return f"error: old text occurs {n} times; include more context"
    f.write_text(text.replace(old, new))
    return "ok"


@tool("List files recursively, optionally filtered by a glob on the filename.",
      path=("string", "directory to list (default '.')", False),
      pattern=("string", "glob matched against the file name, e.g. '*.py'",
               False))
def list_files(path=".", pattern="*"):
    base = safe_path(path)
    if not base.is_dir():
        return "error: not a directory"
    hits = [str(f.relative_to(ROOT)) for f in walk(base)
            if fnmatch.fnmatch(f.name, pattern)]
    out = "\n".join(hits[:500])
    if len(hits) > 500:
        out += f"\n...and {len(hits) - 500} more"
    return out or "(no matches)"


@tool("Search file contents with a Python regular expression.",
      regex=("string", "Python regex", True),
      path=("string", "file or directory to search (default '.')", False))
def grep(regex, path="."):
    rx = re.compile(regex)
    base = safe_path(path)
    files = [base] if base.is_file() else list(walk(base))
    hits = []
    for f in files:
        try:
            text = f.read_text()
        except (UnicodeDecodeError, OSError):
            continue          # binary or unreadable: skip
        for i, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                hits.append(f"{f.relative_to(ROOT)}:{i}: {line.strip()}")
            if len(hits) >= 200:
                return "\n".join(hits) + "\n...(truncated at 200 matches)"
    return "\n".join(hits) or "(no matches)"


@tool("Run a shell command in the workspace root. 60s timeout.",
      command=("string", "shell command", True))
def bash(command):
    try:
        p = subprocess.run(command, shell=True, cwd=ROOT, text=True,
                           capture_output=True, timeout=60)
    except subprocess.TimeoutExpired:
        return "error: timed out after 60s"
    return f"{p.stdout}{p.stderr}\n[exit {p.returncode}]"


def run_tool(name: str, args: dict) -> str:
    """Dispatch one tool call. Errors come back as text so the model can retry."""
    if name not in TOOLS:
        return f"error: unknown tool {name}"
    try:
        out = TOOLS[name][1](**args)
    except Exception as e:
        out = f"error: {type(e).__name__}: {e}"
    if len(out) > MAX_TOOL_OUTPUT:
        extra = len(out) - MAX_TOOL_OUTPUT
        out = out[:MAX_TOOL_OUTPUT] + f"\n...[truncated {extra} chars]"
    return out


# --- permission gate -------------------------------------------------------
def confirm(name: str, args: dict) -> bool:
    """Ask the human before anything that writes to disk or runs a command."""
    if AUTO_APPROVE:
        return True
    preview = args.get("command") or args.get("path", "?")
    print(f"\n[{name}] {preview}")
    body = args.get("content") or args.get("new")
    if body:
        for line in body.splitlines()[:10]:
            print(f"  | {line}")
    try:
        return input("allow? [y/N] ").strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


# --- agent loop ------------------------------------------------------------
SYSTEM_PROMPT = f"""You are a coding agent working in {ROOT}.

Use the tools to inspect the project before you change anything; never guess
what a file contains. Prefer edit_file over rewriting a whole file with
write_file. When tests exist, run them with bash before and after your change.
Keep shell commands short and non-interactive.

Work in small steps: one tool call, look at the result, decide the next one.
When the task is done, stop calling tools and reply in plain text with what you
changed and what the tests said. If you cannot do something, say so plainly."""


def run_turn(messages: list) -> None:
    """Drive the model until it answers with plain text (or hits the cap)."""
    schemas = [s for s, _ in TOOLS.values()]
    for _ in range(MAX_STEPS):
        msg = chat(messages, schemas)
        messages.append(msg)          # verbatim: tool_calls must round-trip
        if msg.get("content"):
            print(msg["content"])
        if not msg.get("tool_calls"):
            return
        for call in msg["tool_calls"]:
            name = call["function"]["name"]
            raw = call["function"]["arguments"]
            args = raw if isinstance(raw, dict) else json.loads(raw or "{}")
            short = json.dumps(args)[:80]
            print(f"\u2192 {name}({short})")
            if name in GATED and not confirm(name, args):
                result = "user denied this action"
            else:
                result = run_tool(name, args)
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": result})
    messages.append({"role": "user",
                     "content": "Stop: step limit reached. Summarize what you did."})
    final = chat(messages, [])
    messages.append(final)
    print(final.get("content", ""))


def main() -> None:
    if not API_KEY and "api.openai.com" in BASE_URL:
        sys.exit("set AGENT_API_KEY (or OPENAI_API_KEY), or point "
                 "AGENT_BASE_URL at a local server")
    prompt = " ".join(a for a in sys.argv[1:] if not a.startswith("-"))
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if prompt:                        # one-shot mode
        messages.append({"role": "user", "content": prompt})
        try:
            run_turn(messages)
        except RuntimeError as e:
            sys.exit(str(e))
        return
    print(f"{MODEL} @ {BASE_URL} in {ROOT}  (ctrl-d to quit)")
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if line in ("exit", "quit"):
            return
        if not line:
            continue
        messages.append({"role": "user", "content": line})
        try:
            run_turn(messages)
        except RuntimeError as e:      # API error: report it, keep the session
            print(e)


if __name__ == "__main__":
    main()
