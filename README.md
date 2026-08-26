# 🧭 Career Advisor

> An AI career-guidance assistant that answers questions using your own
> reference documents (RAG), analyzes resumes, searches jobs, finds skill
> gaps, and builds learning roadmaps — available via a Streamlit UI, a CLI,
> and a Model Context Protocol (MCP) server, with LangSmith tracing and
> evaluation across all of it.


![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b)

<!--
  Add a screenshot or short GIF of the app here once you have one, e.g.:
  ![Career Advisor screenshot](docs/screenshot.png)
-->

## What it does

Career Advisor is a Retrieval-Augmented Generation (RAG) assistant, built on
Google Gemini, that gives practical career guidance. Upload a PDF — a
career-guide, a company handbook, your own notes — and it answers questions
grounded in that document, citing which page(s) it drew from. Ask something
unrelated to the document, and it recognizes that and answers from general
knowledge instead of forcing irrelevant excerpts into the response.

Beyond chat, it also has four dedicated career tools:

| Tool | What it does |
|---|---|
| 🔎 **Job Search** | Search jobs by role, skills, location, and experience level |
| 🎯 **Skill Gap Analyzer** | Compare your current skills against a target role |
| 📄 **Resume Analyzer** | Upload a resume (PDF/DOCX/TXT) — extracts skills and flags missing/improvable areas |
| 🗺️ **Career Roadmap Generator** | Generates a structured, milestone-based learning roadmap |

These tools are available in the Streamlit UI's "Career Tools" tab **and**
as an **MCP server** (`mcp_server.py`), so any MCP-compatible client (Claude
Desktop, Claude Code, other agents) can call them directly.

Built for students and early-career professionals who want fast, specific
answers instead of generic advice.

## How it works

```
PDF upload → chunk → embed → Chroma vector store
                                     │
User question ──► retrieval-needed? ─┤ (real LLM decision, not always-on)
                                     │
                          similarity search + relevance filter
                                     │
                    prompt (history + context) → Gemini → answer + sources

Resume upload → text extraction (PDF/DOCX/TXT) ──► Resume Analyzer (LLM, JSON)
Skill list + target role ─────────────────────────► Skill Gap Analyzer (LLM, JSON)
Skills + target role + timeframe ─────────────────► Roadmap Generator (LLM, JSON)
Role + skills + location + experience ────────────► Job Search (Adzuna API, else bundled dataset)

All of the above are callable from: Streamlit UI · CLI · MCP server (mcp_server.py)
Every LLM call and tool invocation is traced to LangSmith when enabled (tracing.py)
```

See **[ARCHITECTURE.md](ARCHITECTURE.md)** for the full data-flow diagram and
module breakdown.

## Tech stack

- **LLM**: Google Gemini (`gemini-flash-latest` by default)
- **Orchestration**: LangChain (document loading, chunking, vector store)
- **Vector store**: ChromaDB, with local HuggingFace embeddings (`all-MiniLM-L6-v2`)
- **UI**: Streamlit (web) + a CLI, sharing one service layer
- **Tool protocol**: MCP (Model Context Protocol) server exposing the four career tools
- **Observability / evaluation**: LangSmith (tracing + `evaluate()` harness)
- **Job data**: Adzuna API (optional) with a bundled sample dataset fallback
- **Resume parsing**: `pypdf` (PDF), `python-docx` (DOCX), plain text
- **Testing**: `pytest` / `unittest`
- **Lint/format**: `ruff` + `black`
- **CI**: GitHub Actions

## Getting started

**Requirements**: Python 3.10+, a [Google Gemini API key](https://aistudio.google.com/).

```bash

pip install -r requirements.txt

cp .env.example .env
# then edit .env and add your GOOGLE_API_KEY

# Web UI
python run.py web        # → http://localhost:8501

# or CLI
python run.py cli

# or the MCP server (for Claude Desktop / Claude Code / other MCP clients)
python run.py mcp
```

`python run.py status` checks your environment and dependencies before you
start. Full command list: `python run.py`.

### Using the MCP server

Run it standalone with `python run.py mcp` (or `python mcp_server.py`), or
register it with an MCP client, e.g. in Claude Desktop's config:

```json
{
  "mcpServers": {
    "career-advisor": {
      "command": "python",
      "args": ["/absolute/path/to/mcp_server.py"]
    }
  }
}
```

It exposes five tools: `job_search`, `skill_gap_analyzer`, `resume_analyzer`,
`career_roadmap_generator`, and `analyze_uploaded_resume` (a convenience
wrapper that extracts text from a resume file path and analyzes it in one
call).

### Enabling LangSmith tracing + evaluation

By default the app runs fully offline with respect to tracing — nothing is
sent to LangSmith unless you opt in. To enable it:

```bash
# in .env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=career-advisor   # optional, this is the default
```

Once enabled, every chat turn and every career-tool call shows up as a traced
run in your LangSmith project. To check for regressions after changing a
prompt or model, run the evaluation harness:

```bash
python evaluation.py
```

This scores the RAG chat flow and the three LLM-backed career tools
(skill-gap, resume, roadmap) against a small hand-curated dataset and
uploads the results to your LangSmith project dashboard. See
[`evaluation.py`](evaluation.py) for the dataset and evaluators.

### Try it online

The app is a plain Streamlit app, so it deploys directly to
[Streamlit Community Cloud](https://streamlit.io/cloud) with no extra setup:
point it at `app.py` and add `GOOGLE_API_KEY` under the app's **Secrets**.
*(Deploy link goes here once published.)*

## Design decisions

A few choices worth calling out, since they're not obvious from the code alone:

- **A real retrieval-decision step, not always-on RAG.** Before searching the
  document, the model is asked a small, separate yes/no question: does this
  query actually need the document? This is the one deliberately "agentic"
  part of an otherwise single-pass RAG pipeline — see
  [`llm_providers.py`](llm_providers.py) for why that scope was chosen over
  pulling in a full multi-agent framework.
- **A relevance-score threshold on retrieval**, not just top-k. Returning the
  top 3 chunks regardless of how relevant they are means occasionally
  grounding an answer in noise. Chunks below a configurable score
  (`RELEVANCE_SCORE_THRESHOLD`) are dropped instead of injected into the prompt.
- **One shared service layer for every front door.** `app.py` (Streamlit),
  `main.py` (CLI), and `mcp_server.py` (MCP) all call into `rag_service.py`
  and `career_tools.py` rather than each reimplementing the underlying logic
  — so behavior can't silently drift between interfaces.
- **A provider abstraction with only one provider implemented.** `LLMProvider`
  is deliberately minimal — it exists so a second model can be added later
  without touching the RAG pipeline or UI, not because multi-provider support
  is needed today.
- **The job search tool degrades gracefully.** With no `ADZUNA_APP_ID` /
  `ADZUNA_APP_KEY` configured, it filters a small bundled dataset
  ([`jobs_data.py`](jobs_data.py)) instead of failing outright — the same
  "work without extra configuration" philosophy as the RAG pipeline's
  general-mode fallback when no document is loaded.
- **Tracing is additive, never required.** `tracing.py`'s `traceable`
  decorator is a no-op pass-through unless `LANGSMITH_TRACING=true` and an
  API key are set, so the app's behavior and dependencies are unchanged for
  anyone who doesn't use LangSmith.

## Testing

```bash
pytest
```

Unit tests cover config validation, retry/error handling, PDF/resume
validation, relevance filtering, the retrieval-decision logic, the shared
service layer, the four career tools (job search, skill-gap, resume, roadmap),
the LangSmith tracing fallback, and MCP tool registration — using
fakes/mocks for the LLM, vector store, and LangSmith client so the suite runs
without network access or real API credentials.

## Project structure

```
career-advisor/
├── app.py                # Streamlit UI: Chat tab + Career Tools tabs (presentation only)
├── main.py                # CLI (presentation only)
├── mcp_server.py          # MCP server exposing the four career tools
├── rag_service.py         # Shared retrieval + generation logic for chat
├── rag_pipeline.py        # PDF ingestion, chunking, vector store, retrieval
├── career_tools.py        # Job search, skill-gap, resume analysis, roadmap generation
├── resume_pipeline.py     # Resume upload validation + text extraction (PDF/DOCX/TXT)
├── jobs_data.py           # Bundled sample job listings (fallback when no job API is configured)
├── llm_providers.py       # LLMProvider interface + Gemini implementation
├── tracing.py             # LangSmith tracing setup + no-op fallback decorator
├── evaluation.py          # LangSmith evaluation harness (python evaluation.py)
├── config.py              # Centralized, validated settings
├── errors.py               # Shared exception types + retry helper
├── run.py                  # install / setup / status / web / cli / mcp commands
├── tests/                  # pytest / unittest suite
├── .github/workflows/      # CI (lint + tests)
└── ARCHITECTURE.md
```
