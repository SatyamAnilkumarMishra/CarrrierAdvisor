# Architecture

Career Advisor is a single-provider **Retrieval-Augmented Generation (RAG)**
system with one lightweight decision step before retrieval — not a
multi-agent pipeline. This document explains how a request actually flows
through the system and why the code is organized the way it is.

## Data flow

```
                     ┌─────────────────────────┐
                     │   PDF (upload or         │
                     │   default document)      │
                     └────────────┬─────────────┘
                                  │  validate_pdf()
                                  ▼
                     ┌─────────────────────────┐
                     │  PyPDFLoader              │  rag_pipeline.py
                     │  → RecursiveCharacter     │
                     │    TextSplitter           │
                     └────────────┬─────────────┘
                                  │  chunks
                                  ▼
                     ┌─────────────────────────┐
                     │  HuggingFace embeddings   │
                     │  (all-MiniLM-L6-v2)       │
                     │  → Chroma vector store    │
                     │    (persisted to disk)    │
                     └────────────┬─────────────┘
                                  │
   ┌──────────────┐   query      │
   │ User (Streamlit│──────────► │
   │ UI or CLI)     │             │
   └──────────────┘             ▼
                     ┌─────────────────────────┐
                     │  1. decide_needs_        │  llm_providers.py
                     │     retrieval(query)     │  (LLM call — real
                     │                           │   decision, not a
                     │  2. if yes → similarity   │   rule-based stub)
                     │     search + relevance    │  rag_pipeline.py
                     │     threshold filter      │
                     │                           │
                     │  3. build prompt with     │  rag_service.py
                     │     history + context     │
                     │                           │
                     │  4. generate()            │  llm_providers.py
                     └────────────┬─────────────┘
                                  │  answer + sources
                                  ▼
                     ┌─────────────────────────┐
                     │  Streamlit UI / CLI       │
                     │  (renders answer +        │
                     │   source attribution)     │
                     └─────────────────────────┘
```

## Why a query doesn't always trigger retrieval

The original version of this project always ran a similarity search whenever
a document was loaded, even for questions unrelated to it (e.g. "what should
I wear to an interview?" against a resume-writing guide). That pollutes the
prompt with irrelevant context and produces worse answers.

`GeminiProvider.decide_needs_retrieval()` asks the model a small, separate
yes/no question first. Only if the answer is "yes" does `rag_pipeline.py` run
a similarity search — and even then, results below
`Settings.relevance_score_threshold` are dropped before being added to the
prompt. This is the one deliberately "agentic" decision point in an otherwise
straightforward RAG pipeline; see `llm_providers.py` for the reasoning behind
keeping it minimal rather than reaching for a full multi-agent framework.

## Module map

| Module | Responsibility |
|---|---|
| `config.py` | Loads and validates all environment-derived settings. Single source of truth — no other module calls `os.getenv()` directly. |
| `errors.py` | Shared exception types and a retry-with-backoff decorator. Ensures user-facing messages never leak internals. |
| `llm_providers.py` | `LLMProvider` abstract interface + `GeminiProvider` implementation. Swapping models later means adding one class here. |
| `rag_pipeline.py` | PDF validation, chunking, embedding, vector store build/load, relevance-filtered retrieval. |
| `rag_service.py` | Ties retrieval + generation + conversation memory together for chat. **The single place every chat front door calls.** |
| `career_tools.py` | Job search, skill-gap analysis, resume analysis, roadmap generation. **The single place every tool front door calls** (Streamlit "Career Tools" tab, `mcp_server.py`, `evaluation.py`). |
| `resume_pipeline.py` | Resume upload validation + text extraction (PDF/DOCX/TXT), separate from `rag_pipeline.py` since resumes aren't indexed into the vector store. |
| `jobs_data.py` | Small bundled sample job dataset used as a fallback when no live job-search API is configured. |
| `tracing.py` | Configures LangSmith env vars once per process; exposes `traceable`, a no-op pass-through when tracing is disabled/unavailable. |
| `evaluation.py` | LangSmith evaluation harness (`python evaluation.py`) — scores the chat flow and career tools against small hand-curated datasets. |
| `mcp_server.py` | MCP server exposing the four career tools to external MCP clients (Claude Desktop, Claude Code, etc.), built on `career_tools.py` and `resume_pipeline.py`. |
| `app.py` | Streamlit UI: a Chat tab (`rag_service.py`) and a Career Tools tab set (`career_tools.py`). Presentation only — no business logic. |
| `main.py` | CLI. Presentation only — shares chat logic with `app.py` via `rag_service.py`. |
| `run.py` | Developer convenience commands (`install` / `setup` / `status` / `web` / `cli` / `mcp`). |

## Career tools data flow

The four career tools are a second, parallel surface alongside chat — they
don't touch the vector store or conversation history, and each is a single
LLM call (except job search, which isn't an LLM call at all):

```
                     ┌──────────────────────────┐
                     │ Streamlit "Career Tools"  │
                     │ tabs  /  MCP client        │
                     └────────────┬───────────────┘
                                  │
              ┌───────────────────┼────────────────────┬─────────────────────┐
              ▼                   ▼                     ▼                    ▼
    ┌─────────────────┐ ┌──────────────────┐ ┌────────────────────┐ ┌──────────────────┐
    │ search_jobs()     │ │ analyze_skill_gap()│ │ analyze_resume()    │ │ generate_roadmap()│
    │ career_tools.py   │ │ career_tools.py    │ │ career_tools.py     │ │ career_tools.py   │
    │                    │ │                     │ │                      │ │                    │
    │ Adzuna API if      │ │ LLM call →          │ │ resume_pipeline.py   │ │ LLM call →         │
    │ configured, else    │ │ structured JSON     │ │ extracts text first  │ │ structured JSON    │
    │ bundled dataset     │ │                     │ │ (PDF/DOCX/TXT) →     │ │ (milestones)       │
    │ (jobs_data.py)      │ │                     │ │ LLM call → JSON      │ │                    │
    └─────────────────┘ └──────────────────┘ └────────────────────┘ └──────────────────┘
```

Every one of these calls is wrapped with `@traceable` (`tracing.py`), so when
`LANGSMITH_TRACING=true` each run — inputs, outputs, latency — is logged to
LangSmith exactly like the chat flow's `RagService.answer()` and
`GeminiProvider.generate()` calls. `evaluation.py` scores this surface using
the same underlying functions, so a prompt change here is checked the same
way a chat-prompt change is.

## Why the MCP server shares `career_tools.py` instead of reimplementing it

`mcp_server.py` calls the exact same functions the Streamlit "Career Tools"
tabs call — same pattern as `rag_service.py` being the single shared brain
behind `app.py` and `main.py`. This means an MCP client (e.g. Claude Desktop)
and a person using the web UI always get identical tool behavior, and a
prompt fix only needs to happen in one place.

## Conversation memory

Each session keeps a bounded window of prior turns
(`Settings.max_history_turns`, default 6 exchanges). `rag_service.trim_history()`
truncates older turns before they're sent to the model, keeping prompt size —
and cost — predictable regardless of how long a conversation runs.

## Configuration boundary

`config.py` is the only module that reads environment variables directly.
Everything else receives a validated `Settings` object. This makes the rest
of the codebase trivially testable (see `tests/`) without needing a real
`.env` file, and means a misconfigured deployment fails immediately at
startup with a specific, actionable error instead of a confusing failure
three requests later.

## A note on dependency pinning: `mcp`

`requirements.txt` pins `mcp>=1.2.0,<2.0.0`. The `mcp` package's 2.x line
restructured its server API — `FastMCP` (what `mcp_server.py` is built on)
was renamed/moved to `MCPServer` under `mcp.server.mcpserver`. Letting pip
resolve an unconstrained `mcp>=1.2.0` would silently pull in 2.x and break
`mcp_server.py` at import time. If you want to move to the 2.x API, update
the `from mcp.server.fastmcp import FastMCP` import and tool-registration
calls in `mcp_server.py` accordingly, then relax this pin.
