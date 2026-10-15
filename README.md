# A coding agent in under 500 lines of Python

Two single-file coding agents that read, edit, and test the code in your project through any OpenAI-compatible chat API. `agent.py` is the smallest version that works (354 lines, one dependency); `harness.py` is the same agent made usable all day, with streaming, retries, saved sessions, automatic summarizing of long conversations, and a Ctrl-C that stops the command instead of the program (499 lines). If you've wondered what's inside Claude Code or Cursor, or want an agent small enough to read in an afternoon and change to fit your workflow, this is it.

"OpenAI-compatible" means the agent sends the same `/v1/chat/completions` request OpenAI does. OpenRouter, Together, vLLM, llama.cpp's server, and Ollama all accept it, so switching providers is one environment variable.

## What it looks like

Here's `agent.py` fixing an off-by-one bug, from a real run with `gpt-4o-mini`. The scratch directory holds a `fizzbuzz.py` whose loop is `range(1, n)` instead of `range(1, n + 1)`, and a two-assertion pytest file that catches it:

```
$ python ../agent.py --yes "run the tests, fix the bug, run them again"
→ list_files({"path": "tests"})
→ list_files({"path": "."})
→ bash({"command": "pytest"})
→ read_file({"path": "fizzbuzz.py"})
→ edit_file({"path": "fizzbuzz.py", "old": "for i in range(1, n):", "new": "for i in range(1)
→ bash({"command": "pytest"})
I fixed the bug in the `fizzbuzz` function by changing the loop range from
`range(1, n)` to `range(1, n + 1)`, ensuring it includes the number `n`. After
making this change, I ran the tests again, and all of them passed successfully.
```

Six tool calls, seventeen seconds. Each `→` line is the agent printing which tool the model asked for. The first call is a wrong guess (there is no `tests/` directory); the model read the error, looked at `.` instead, and carried on. Without `--yes`, the agent asks `allow? [y/N]` before `edit_file` and both `bash` calls.

## Run it

You need Python 3.9+ and `requests`. The agent works on the directory you run it from, so `cd` into your project first.

```sh
pip install requests          # add ddgs if you want web_search

# OpenAI
export OPENAI_API_KEY=sk-...
python agent.py "add type hints to utils.py and run mypy"

# OpenRouter
export AGENT_BASE_URL=https://openrouter.ai/api/v1
export AGENT_API_KEY=sk-or-...
export AGENT_MODEL=anthropic/claude-sonnet-4
python agent.py

# Ollama, no key needed
AGENT_BASE_URL=http://localhost:11434/v1 AGENT_MODEL=qwen2.5-coder \
  python agent.py "explain what this repo does"
```

With a prompt on the command line, the agent runs one turn and exits. With no prompt you get a `>` prompt and a conversation that remembers earlier turns; Ctrl-D or `exit` ends it. `harness.py` takes the same arguments plus `--resume`.

| Setting | Default | Effect |
| --- | --- | --- |
| `AGENT_BASE_URL` | `https://api.openai.com/v1` | Any OpenAI-compatible API root |
| `AGENT_API_KEY` | falls back to `OPENAI_API_KEY` | API key; empty is fine for local servers |
| `AGENT_MODEL` | `gpt-4o-mini` | Model name as the endpoint knows it |
| `AGENT_COMPACT_AT` | `120000` | `harness.py` only: conversation size, in characters, that triggers summarization |
| `--yes` / `-y` | off | Skip the permission prompts; `ask_user` returns an error instead of waiting |
| `--resume` / `-r` | off | `harness.py` only: reopen the newest saved conversation in `.agent/` |

## What's in the repo

| File | Lines | What it is |
| --- | --- | --- |
| `agent.py` | 354 | The minimal agent: one HTTP call, nine tools, a loop |
| `harness.py` | 499 | `agent.py` plus streaming, retries, sessions, compaction, cancellation |
| `flaky_server.py` | 32 | A fake API for testing retries and streaming without a key |

The two agents share everything except the 145 lines the harness adds, so `diff agent.py harness.py` shows exactly what each fix cost. `git log --reverse` reads in the same order as the blog posts below, roughly one tool or one fix per commit.

## How it works

Chat APIs support *tool calling*: along with your messages you send a list of functions the model may ask you to run. Instead of answering in prose, the model can reply "call `read_file` with `path=fizzbuzz.py`". You run the function, append the result as a `tool` message, and send the whole conversation again. The model never runs anything itself; it only asks.

An agent is that exchange on repeat. This is the loop body in `agent.py`'s [`run_turn`](agent.py#L303-L319):

```python
    for _ in range(MAX_STEPS):
        msg = chat(messages, SCHEMAS)
        messages.append(msg)          # verbatim: tool_calls must round-trip
        if msg.get("content"):
            print(msg["content"])
        if not msg.get("tool_calls"):
            return
        for call in msg["tool_calls"]:
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": run_tool(call)})
```

Everything else supports this loop: one function that makes the HTTP call, a `@tool` decorator that builds the parameter description the API wants (JSON Schema) from each function's Python signature, and the tools themselves. The model learns what a tool does from the decorator's description and sees every failure (unknown tool, bad arguments, missing file, path outside the project) as a short error string it can read and recover from.

| Tool | What it does | Asks first? |
| --- | --- | --- |
| `read_file` | File contents with line numbers; `start`/`end` for big files | |
| `list_files` | Recursive listing, optional glob on the file name | |
| `grep` | Regex search, `path:line: text` output | |
| `write_file` | Create or overwrite a file | yes |
| `edit_file` | Replace a substring that must occur exactly once | yes |
| `bash` | Shell command in the project root, 60 s timeout | yes |
| `ask_user` | Put a question to you and wait for the answer | |
| `web_search` | DuckDuckGo results; needs `pip install ddgs` | |
| `fetch_url` | A page as plain text, scripts and tags stripped | |

Two checks stand between the model and your machine. Every path goes through `safe_path()`, which resolves it (collapsing `..` and symlinks) and refuses anything outside the directory you started in. And the three tools that change things ask you `allow? [y/N]` before running; a "no" goes back to the model as `user denied this action`, and it usually replies with the patch for you to apply by hand. The web tools aren't gated because they only read, but a URL can carry data out, so add them to `GATED` if the code must not leak.

Other limits: 50 tool rounds per turn, after which the agent asks the model to summarize and stop; tool results cut at 20,000 characters; `fetch_url` reads 4 MB at most and refuses non-text responses.

## The harness

A *harness* is everything around the model: the loop, the tools, the network code, the prompts. `harness.py` starts as `agent.py` verbatim and adds five things you notice the first day you use an agent for real:

- **Streaming.** The reply prints as it's generated instead of arriving in one lump after twenty seconds. Tool calls arrive in fragments too and are reassembled by index.
- **Retries.** A 429 or 5xx waits (1, 2, 4… seconds, or the server's `Retry-After`) and tries again, giving up after four attempts. A 400 or 401 fails at once, since retrying a bad key only takes longer to be wrong.
- **Sessions.** The transcript (the message list sent to the model, as plain JSON) is saved to `.agent/` after every turn. `--resume` reopens the newest one.
- **Compaction.** When the transcript passes `AGENT_COMPACT_AT` characters, the older half is replaced with a summary the model writes for itself, so a long task doesn't overflow the model's context window (the most text it can read in one request).
- **Cancellation.** Ctrl-C ends the current turn, not the program. Any tool call it interrupted gets a `cancelled by user` result so the transcript stays valid.

Four of the five share one rule: **the transcript must be valid after every operation.** Every tool call has exactly one `tool` message answering it, and the two are never separated. Break that and the agent works for a few turns, then every request fails with a 400, because the damage is in the history.

### Testing it without an API key

You can't ask a real provider for a rate limit on demand, but the endpoint is just a URL. `flaky_server.py` answers the first request with a 429 and `Retry-After: 1`, and every request after that with a canned streamed reply. Start it in another terminal with `python flaky_server.py`, then:

```
$ AGENT_BASE_URL=http://127.0.0.1:8931/v1 python harness.py --yes "hi"
[API 429: {"error":"slow down"} | retry 1/4 in 1s]
hello from the flaky server
```

Point `AGENT_BASE_URL` at a closed port and you see the give-up path: four backoffs, then `gave up after 4 attempts`. Set `AGENT_COMPACT_AT=1500` and any real task triggers compaction within a few rounds. For cancellation, ask for `sleep 60`, press Ctrl-C, then ask the agent what it was doing; if it answers instead of failing with a 400, the repair worked.

## Not included

What separates `harness.py` from something you'd ship to strangers, none of which needs a different architecture:

- **Concurrent tool calls.** Models often ask for several tools at once; these run them one after another.
- **Permission scopes.** The y/n prompt asks every time; it can't remember "always allow `pytest`".
- **File-change detection.** If you edit a file between the model reading it and writing it, the write wins.
- **A real tokenizer.** Counting characters instead of tokens (the units models actually limit and bill by) is fine for a threshold and bad for a budget.
- **Subagents.** One conversation means one context window.
- **A terminal UI.** Rendered diffs, spinners, syntax highlighting. This is where thousands of lines go.

## Read more

- [Build a coding agent in about 300 lines of Python](https://kialog.nlogn.ir/posts/coding-agent-in-300-lines/): every line of `agent.py`, and why `edit_file`'s "exactly once" rule matters more than any prompt.
- [What makes a coding agent usable: five fixes in 145 lines of Python](https://kialog.nlogn.ir/posts/making-coding-agents-usable/): the harness, one fix at a time.
