# 🧭 Career Advisor

> An AI career-intelligence platform that answers questions using your own reference documents (RAG), analyzes resumes, searches live jobs, evaluates skill gaps, and builds milestone learning roadmaps — featuring a modern Next.js web application, a FastAPI backend, an interactive CLI, and a Model Context Protocol (MCP) server with LangSmith observability.

[![CI](https://github.com/<your-username>/<your-repo>/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/<your-repo>/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Next.js](https://img.shields.io/badge/Next.js-16-black)
![React](https://img.shields.io/badge/React-19-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-green)

---

## 🌟 Workspaces & Features

| Workspace / Tool | Capabilities |
|---|---|
| 💬 **AI Career Agent** | Grounded RAG chat with document citation, prompt quick-actions, markdown renderer, and animated mascot |
| 📄 **Resume Analyzer** | Multi-format upload (PDF/DOCX/TXT), skill extraction, ATS summary, strengths, growth areas, and target role suggestions |
| 🎯 **Skill Gap Matrix** | Compare skills against industry standards, high/medium/low readiness indicator, 3-column breakdown |
| 🗺️ **Learning Roadmap** | Milestone-by-milestone curriculum with customizable duration, focus skills, and interactive checklists |
| 🔎 **Live Job Matcher** | Real-time job search via Adzuna API + local verified dataset, filterable by role, location, skills, and experience |

---

## 🏗️ Project Structure

```
career-advisor/
├── backend/                  # Python Backend Package
│   ├── server.py             # FastAPI REST Server
│   ├── career_tools.py       # Core tool algorithms & business logic
│   ├── rag_service.py        # RAG orchestration service
│   ├── rag_pipeline.py       # Chroma vector store & embeddings
│   ├── resume_pipeline.py    # Resume text extraction (PDF, DOCX, TXT)
│   ├── llm_providers.py      # Gemini API provider & ChatMessage types
│   ├── mcp_server.py         # MCP server for external AI agents
│   ├── main.py               # Terminal CLI interface
│   ├── jobs_data.py          # Fallback job listings dataset
│   ├── config.py             # App settings & validation
│   ├── errors.py             # Custom exceptions & backoff retry
│   ├── tracing.py            # LangSmith tracing & telemetry
│   └── evaluation.py         # Evaluation benchmark harness
├── frontend/                 # Next.js 16 Web Application
│   ├── src/
│   │   ├── app/              # App router & global styles
│   │   ├── components/       # Workspace view components & UI widgets
│   │   ├── lib/api.ts        # API client for FastAPI backend
│   │   └── types/index.ts    # Shared TypeScript data models
│   ├── package.json
│   └── tsconfig.json
├── tests/                    # Backend Unit & Integration Tests (87 tests)
├── run.py                    # Unified Runner CLI
├── requirements.txt          # Python dependencies
└── pyproject.toml            # Project configuration & pytest settings
```

---

## 🚀 Quickstart

### 1. Setup Environment
```bash
pip install -r requirements.txt
python run.py setup
# Edit .env and insert your GOOGLE_API_KEY
```

The repo ships with a default reference document, `Career_Advisor_Guide_2025.pdf`, covering
resumes/ATS, interviews, negotiation, skill-building, job search, and career roadmaps. On first
run (CLI) or first `/api/documents/upload` call, it — or whatever PDF you upload — gets chunked,
embedded (`sentence-transformers/all-MiniLM-L6-v2`), and written to a persisted **ChromaDB**
vector store at `CHROMA_PERSIST_DIR` (default `chroma_db/`, gitignored). Subsequent runs reuse
that persisted index instead of rebuilding it. Delete the `chroma_db/` folder to force a rebuild,
or upload a new PDF via the Knowledge & RAG Hub in the frontend to re-index against your own
document.

### 2. Verify your keys actually work

```bash
python run.py doctor
```

Runs a live preflight in the same order a real request travels — config →
Gemini (a real API call with your key) → embeddings → Chroma → a full
end-to-end RAG answer → job search — and prints PASS/FAIL/SKIP for each with
an actionable fix. Exits non-zero on failure, so it works in CI too. If this
passes, the app will generate responses.

`GEMINI_MODEL` defaults to **`gemini-flash-latest`**, a moving alias that always
resolves to the current Flash model, so it stays compatible as individual model
versions are retired. If a model is unavailable for your key, the provider
automatically falls back through other current Flash models.

Only `GOOGLE_API_KEY` is required. `ADZUNA_APP_ID` / `ADZUNA_APP_KEY` are
optional — without them the Job Matcher serves the bundled dataset instead of
live listings. `LANGSMITH_API_KEY` is only needed if you set
`LANGSMITH_TRACING=true`.

### 3. Start Applications
```bash
# Option A: Start FastAPI Backend
python run.py api        # → http://localhost:8000

# Option B: Start Next.js Frontend
python run.py frontend   # → http://localhost:3000

# Option C: Terminal CLI
python run.py cli

# Option D: MCP Server (for Claude Desktop, Claude Code, etc.)
python run.py mcp
```

### 4. Run Test Suite
```bash
python run.py doctor          # live preflight (needs your real API key)
python -m pytest              # backend: 87 tests

cd frontend
npm install
npm run lint                  # eslint — clean
npx tsc --noEmit              # type-check — clean
npm run build                 # production build
```

The web UI is responsive: below 900px the sidebar becomes an off-canvas
drawer opened by the hamburger in the topbar, grids collapse to a single
column, tap targets meet the 44px minimum, and inputs use a 16px font so iOS
Safari doesn't zoom on focus. Verified at 375/393/412/768px with no
horizontal overflow.

> **Frontend note:** `npm install` prints `npm warn deprecated eslint@9.x`.
> This is expected and safe to ignore. ESLint 9 is in maintenance mode now that
> 10 is out, but `eslint-config-next@16.3.3` bundles `eslint-plugin-react@7.37.x`,
> which crashes under ESLint 10 (`contextOrFilename.getFilename is not a function`)
> and takes the whole lint run down. Stay on 9.x until Next ships a config that
> declares ESLint 10 support. See the comment at the top of `eslint.config.mjs`.

---

## 🔌 MCP Integration (Claude Desktop)
Add to your Claude Desktop configuration:
```json
{
  "mcpServers": {
    "career-advisor": {
      "command": "python",
      "args": ["-m", "backend.mcp_server"]
    }
  }
}
```
