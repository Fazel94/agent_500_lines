#!/usr/bin/env python3
"""An agent harness in <500 lines. Stdlib, plus ddgs for web_search.

agent.py plus the machinery that makes it livable: streaming output,
retries, resumable sessions, context compaction, and cancellation.
Usage: python harness.py [--yes] [--resume] ["do the thing"]
"""
import fnmatch
import html
import inspect
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

BASE_URL = os.environ.get("AGENT_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.environ.get("AGENT_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
MODEL = os.environ.get("AGENT_MODEL", "gpt-4o-mini")
ROOT = pathlib.Path.cwd()          # the agent may only touch files under here
MAX_TOOL_OUTPUT = 20_000           # chars; tool results are truncated to this
PAGE_CHARS = 10_000                # chars of page text per fetch_url call
UA = "Mozilla/5.0 (compatible; tiny-agent/1.0)"   # fetch_url identifies itself
MAX_STEPS = 50                     # tool rounds per user turn
MAX_RETRIES = 4                    # attempts per API call before giving up
COMPACT_AT = int(os.environ.get("AGENT_COMPACT_AT", 120_000))  # chars, not tokens
SKIP_DIRS = {".git", ".agent", "node_modules", "__pycache__", ".venv"}
SESSION_DIR = ROOT / ".agent"      # transcripts, one JSON file per session
AUTO_APPROVE = "--yes" in sys.argv or "-y" in sys.argv
GATED = {"write_file", "edit_file", "bash"}
SYSTEM_PROMPT = f"""You are a coding agent working in {ROOT}.

Use the tools to inspect the project before you change anything; never guess
what a file contains. Prefer edit_file over rewriting a whole file with
write_file. When tests exist, run them with bash before and after your change.
Keep shell commands short and non-interactive. For anything outside this
workspace, web_search with keywords, then fetch_url the best hit and cite it.

Work in small steps: one tool call, look at the result, decide the next one.
When the task is done, stop calling tools and reply in plain text with what you
changed and what the tests said. If you cannot do something, say so plainly."""


def post(messages: list, tools: list, stream: bool):
    """Open the HTTP response for one completion request."""
    body = json.dumps({"model": MODEL, "messages": messages, "tools": tools,
                       "tool_choice": "auto", "stream": stream}).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions", data=body,
        headers={"Authorization": f"Bearer {API_KEY}",
                 "Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=120)


def merge_delta(msg: dict, delta: dict) -> None:
    """Fold one SSE delta into the assistant message being assembled."""
    if delta.get("content"):
        print(delta["content"], end="", flush=True)
        msg["content"] += delta["content"]
    for part in delta.get("tool_calls") or []:
        slot = msg["calls"].setdefault(part.get("index", 0),
                                       {"id": "", "type": "function",
                                        "function": {"name": "", "arguments": ""}})
        slot["id"] += part.get("id") or ""
        fn = part.get("function") or {}
        slot["function"]["name"] += fn.get("name") or ""
        slot["function"]["arguments"] += fn.get("arguments") or ""


class Retryable(RuntimeError):
    """A failure worth another attempt: rate limit, 5xx, dropped connection."""

    def __init__(self, message, delay=None):
        super().__init__(message)
        self.delay = delay


def once(messages: list, tools: list, stream: bool = True) -> dict:
    """One round-trip. Returns the assistant message dict, printing as it streams."""
    msg = None
    try:
        with post(messages, tools, stream) as resp:
            if not stream:
                return json.loads(resp.read())["choices"][0]["message"]
            msg = {"role": "assistant", "content": "", "calls": {}}
            for raw in resp:                      # server-sent events, one per line
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                choices = json.loads(data).get("choices") or [{}]
                merge_delta(msg, choices[0].get("delta") or {})
    except urllib.error.HTTPError as e:
        detail = f"API {e.code}: {e.read().decode()[:200]}"
        if e.code == 429 or e.code >= 500:
            raise Retryable(detail, int(e.headers.get("Retry-After") or 0) or None)
        raise RuntimeError(detail)
    except OSError as e:               # DNS, TLS, timeout, connection reset
        if msg and (msg["content"] or msg["calls"]):
            raise RuntimeError(f"stream died mid-message: {e}")
        raise Retryable(f"API unreachable: {e}")
    calls = msg.pop("calls")
    if calls:
        msg["tool_calls"] = [calls[i] for i in sorted(calls)]
    if msg["content"]:
        print()                                   # close the streamed line
    else:
        msg["content"] = None
    return msg


def chat(messages: list, tools: list, stream: bool = True) -> dict:
    """once(), with exponential backoff on the failures that deserve it."""
    for attempt in range(MAX_RETRIES):
        try:
            return once(messages, tools, stream)
        except Retryable as e:
            delay = e.delay or min(2 ** attempt, 30)
            print(f"[{e} | retry {attempt + 1}/{MAX_RETRIES} in {delay}s]")
            time.sleep(delay)
    raise RuntimeError(f"gave up after {MAX_RETRIES} attempts")


# --- tools -----------------------------------------------------------------
TOOLS: dict = {}       # name -> python function
SCHEMAS: list = []     # what the model sees, in registration order
JSON_TYPES = {str: "string", int: "integer", bool: "boolean"}


def tool(description: str, **params: str):
    """Register fn as a tool. The JSON type comes from each parameter's
    annotation, a parameter with no default is required, and params holds
    one line of help per parameter."""
    def register(fn):
        properties, required = {}, []
        for name, p in inspect.signature(fn).parameters.items():
            properties[name] = {"type": JSON_TYPES[p.annotation],
                                "description": params[name]}
            if p.default is p.empty:
                required.append(name)
        SCHEMAS.append({"type": "function", "function": {
            "name": fn.__name__, "description": description,
            "parameters": {"type": "object", "properties": properties,
                           "required": required}}})
        TOOLS[fn.__name__] = fn
        return fn
    return register


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
      path="file path relative to the workspace root",
      start="first line to return (1-based, default 1)",
      end="last line to return (default: end of file)")
def read_file(path: str, start: int = 1, end: int = None):
    f = safe_path(path)
    if not f.is_file():
        return "error: not found"
    lines = f.read_text(errors="replace").splitlines()
    start = max(int(start), 1)
    end = len(lines) if end is None else min(int(end), len(lines))
    out = "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, end + 1))
    if end < len(lines):
        out += f"\n...[{len(lines) - end} more lines; read them with start={end + 1}]"
    return out or "(empty range)"


@tool("Create or overwrite a file with exact content.",
      path="file path relative to the workspace root",
      content="full file content")
def write_file(path: str, content: str):
    f = safe_path(path)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(content)
    return f"wrote {len(content.encode())} bytes to {path}"


@tool("Replace an exact substring in a file. `old` must appear exactly once.",
      path="file path relative to the workspace root",
      old="exact text to replace, with enough context to be unique",
      new="replacement text")
def edit_file(path: str, old: str, new: str):
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
      path="directory to list (default '.')",
      pattern="glob matched against the file name, e.g. '*.py'")
def list_files(path: str = ".", pattern: str = "*"):
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
      regex="Python regex",
      path="file or directory to search (default '.')")
def grep(regex: str, path: str = "."):
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
      command="shell command")
def bash(command: str):
    try:
        p = subprocess.run(command, shell=True, cwd=ROOT, text=True,
                           capture_output=True, timeout=60)
    except subprocess.TimeoutExpired:
        return "error: timed out after 60s"
    return f"{p.stdout}{p.stderr}\n[exit {p.returncode}]"


@tool("Search the web with DuckDuckGo. Returns ranked title/url/snippet "
      "triples; the snippet is a teaser, so fetch_url the ones that matter.",
      query="search query, as you would type it into a search box",
      max_results="how many results to return (default 5, max 25)")
def web_search(query: str, max_results: int = 5):
    try:
        from ddgs import DDGS     # pip install ddgs
    except ImportError:
        return "error: web search needs the ddgs package (pip install ddgs)"
    n = max(1, min(int(max_results), 25))
    hits = []
    for _ in (1, 2):              # a rotated backend comes back empty at random
        try:
            hits = DDGS(timeout=30).text(query, max_results=n) or []
        except Exception as e:    # rate limits, captchas, DNS
            if "no results" not in str(e).lower():   # ddgs raises on empty too
                return f"error: search failed: {type(e).__name__}: {e}"
        if hits:
            break
    out = [f"{i}. {h.get('title', '')}\n   {h.get('href', '')}\n   "
           f"{' '.join((h.get('body') or '').split())}"
           for i, h in enumerate(hits, 1)]
    return "\n".join(out) or "(no results)"


@tool("Fetch a URL and return it as plain text, scripts and tags stripped. "
      "Use it to actually read a page web_search pointed at.",
      url="absolute http:// or https:// URL",
      start="character offset to resume from (default 0)")
def fetch_url(url: str, start: int = 0):
    if not url.startswith(("http://", "https://")):
        return "error: url must start with http:// or https://"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=30) as r:
            kind = r.headers.get_content_type()
            if not (kind.startswith("text/") or "json" in kind or "xml" in kind):
                return f"error: {kind} is not text"
            body = r.read(4_000_000).decode(
                r.headers.get_content_charset() or "utf-8", "replace")
    except Exception as e:        # DNS, TLS, 404, timeout
        return f"error: fetch failed: {type(e).__name__}: {e}"
    if kind.startswith("text/html"):
        body = re.sub(r"(?is)<(script|style|noscript|svg|head|nav|footer|aside"
                      r"|form)\b.*?</\1>", " ", body)
        main = re.search(r"(?is)<(main|article)\b.*?</\1>", body)   # skip the chrome
        body = main.group(0) if main else body
        body = re.sub(r"(?i)<br\s*/?>|</(p|div|li|tr|h[1-6])>", "\n", body)
        body = html.unescape(re.sub(r"(?s)<!--.*?-->|<[^>]+>", " ", body))
    text = "\n".join(ln.strip() for ln in re.sub(r"[ \t]+", " ", body).splitlines()
                     if ln.strip())
    start = max(int(start), 0)
    out = text[start:start + PAGE_CHARS]
    if len(text) > start + PAGE_CHARS:
        out += (f"\n...[{len(text) - start - PAGE_CHARS} more chars; "
                f"continue with start={start + PAGE_CHARS}]")
    return out or "(no text on page)"


@tool("Ask the human a question and wait for their answer. Use only for a "
      "decision the code cannot answer, never to ask permission.",
      question="a single, specific question")
def ask_user(question: str):
    if AUTO_APPROVE:
        return "error: running unattended (--yes); use your best judgement"
    print(f"\n[?] {question}")
    try:
        return input("your answer: ").strip() or "(no answer given)"
    except (EOFError, KeyboardInterrupt):
        return "(no answer given)"


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


# --- dispatch --------------------------------------------------------------
def run_tool(call: dict) -> str:
    """Execute one tool call from the model. Every failure comes back as
    text, so the model can read it and try something else."""
    name = call["function"]["name"]
    raw = call["function"]["arguments"]
    try:
        args = raw if isinstance(raw, dict) else json.loads(raw or "{}")
        print(f"\u2192 {name}({json.dumps(args)[:80]})")
        if name not in TOOLS:
            return f"error: unknown tool {name}"
        if name in GATED and not confirm(name, args):
            return "user denied this action"
        out = TOOLS[name](**args)
    except Exception as e:
        out = f"error: {type(e).__name__}: {e}"
    if len(out) > MAX_TOOL_OUTPUT:
        extra = len(out) - MAX_TOOL_OUTPUT
        out = out[:MAX_TOOL_OUTPUT] + f"\n...[truncated {extra} chars]"
    return out


# --- agent loop ------------------------------------------------------------
def cut_point(messages: list) -> int:
    """Halfway, then forward to a boundary that keeps tool results with their call."""
    i = 1 + (len(messages) - 1) // 2
    while i < len(messages) and messages[i].get("role") == "tool":
        i += 1
    return i


def compact(messages: list) -> None:
    """Replace the older half of the transcript with a summary, in place."""
    if len(json.dumps(messages)) < COMPACT_AT or len(messages) < 8:
        return
    cut = cut_point(messages)
    old = messages[1:cut]
    if not old:
        return
    print(f"[compacting {len(old)} messages]")
    ask = [messages[0], *old, {"role": "user", "content":
           "Summarize the work so far for your own future reference: files "
           "read and changed, decisions made, what is still unfinished. Be "
           "specific about names and line numbers. No preamble."}]
    summary = chat(ask, [], stream=False).get("content") or "(summary failed)"
    messages[1:cut] = [{"role": "user",
                        "content": f"Summary of earlier work:\n{summary}"}]


def seal(messages: list) -> None:
    """Make an interrupted transcript valid: every tool_call needs a result."""
    answered = {m.get("tool_call_id") for m in messages if m.get("role") == "tool"}
    for m in list(messages):
        for call in m.get("tool_calls") or []:
            if call["id"] not in answered:
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": "cancelled by user"})
    messages.append({"role": "user", "content": "[interrupted by the user]"})


def steps(messages: list) -> None:
    """The loop proper: model, tools, repeat."""
    for _ in range(MAX_STEPS):
        compact(messages)
        msg = chat(messages, SCHEMAS)
        messages.append(msg)          # verbatim: tool_calls must round-trip
        if not msg.get("tool_calls"):
            return
        for call in msg["tool_calls"]:
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": run_tool(call)})
    messages.append({"role": "user",
                     "content": "Stop: step limit reached. Summarize what you did."})
    messages.append(chat(messages, []))


def run_turn(messages: list) -> None:
    """One turn, interruptible. Ctrl-C ends the turn, not the session."""
    try:
        steps(messages)
    except KeyboardInterrupt:
        print("\n[cancelled]")
        seal(messages)


# --- sessions --------------------------------------------------------------
def newest_session():
    """The most recently written transcript, or None."""
    files = sorted(SESSION_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime)
    return files[-1] if files else None


def open_session(resume: bool):
    """Return (messages, path). --resume reopens the newest transcript."""
    if resume:
        path = newest_session()
        if path:
            messages = json.loads(path.read_text())
            print(f"[resumed {path.name}: {len(messages)} messages]")
            return messages, path
        print("[no session to resume; starting a new one]")
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    return ([{"role": "system", "content": SYSTEM_PROMPT}],
            SESSION_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}.json")


def save(messages: list, path: pathlib.Path) -> None:
    """Write the whole transcript after every turn. It is just a list of dicts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(messages, indent=1))


def main() -> None:
    if not API_KEY and "api.openai.com" in BASE_URL:
        sys.exit("set AGENT_API_KEY (or OPENAI_API_KEY), or point "
                 "AGENT_BASE_URL at a local server")
    prompt = " ".join(a for a in sys.argv[1:] if not a.startswith("-"))
    messages, path = open_session("--resume" in sys.argv or "-r" in sys.argv)
    if prompt:                        # one-shot mode
        messages.append({"role": "user", "content": prompt})
        try:
            run_turn(messages)
        except RuntimeError as e:
            sys.exit(str(e))
        finally:
            save(messages, path)
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
        save(messages, path)


if __name__ == "__main__":
    main()
