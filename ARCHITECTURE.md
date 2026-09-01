# Architecture

Career Advisor is a production-ready, dual-tier application featuring a **Next.js 16 + React 19 + TypeScript** modern web frontend and a **FastAPI / MCP** Python backend.

---

## High-Level System Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    Next.js Frontend (Port 3000)            │
│  - AI Career Agent Chat with Animated Levitating Mascot    │
│  - Resume Analyzer & Verified Skills Extraction             │
│  - Skill Gap Matrix & Readiness Evaluation                 │
│  - Milestone-based Learning Roadmap Generator              │
│  - Live Job Matcher & Opportunities Browser                │
│  - Knowledge & Context Hub (PDF Indexing & Profile)        │
└─────────────────────────────┬──────────────────────────────┘
                              │ REST HTTP Requests (CORS)
                              ▼
┌────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (Port 8000)             │
│                     backend/server.py                      │
├────────────────────────────────────────────────────────────┤
│  Core Backend Services (backend/):                         │
│  ├── rag_service.py       -> RAG & Conversation Engine     │
│  ├── rag_pipeline.py      -> PDF Chunking, Embeddings,     │
│  │                           ChromaDB Vector Store         │
│  ├── career_tools.py      -> Career Intelligence Tools:    │
│  │                           Resume, Skill Gap, Roadmap,   │
│  │                           Job Search                    │
│  ├── resume_pipeline.py   -> Multi-format parser           │
│  │                           (PDF, DOCX, TXT)              │
│  ├── llm_providers.py     -> Google Gemini Model Adapter   │
│  ├── mcp_server.py        -> Model Context Protocol Server │
│  ├── main.py              -> Interactive Terminal CLI      │
│  ├── config.py            -> Single source of truth config │
│  ├── errors.py            -> Safe error handling & retries │
│  ├── tracing.py           -> LangSmith Telemetry & Tracing │
│  └── evaluation.py        -> Benchmark Evaluation Harness  │
└────────────────────────────────────────────────────────────┘
```

---

## Data Flow: RAG Document Chat

```
PDF upload (Reference Doc)
        │ validate_pdf()
        ▼
PyPDFLoader → RecursiveCharacterTextSplitter
        │ chunks
        ▼
HuggingFace embeddings (all-MiniLM-L6-v2) → Chroma vector store
        │
User Query ──► 1. decide_needs_retrieval(query) (LLM decision)
                    │
                    ├── If YES: ChromaDB similarity search + relevance filter
                    │
               2. Build prompt with history + grounded context
                    │
               3. Gemini generate() → answer + source citations
```

---

## Directory Organization

```
career-advisor/
├── backend/                  # Python Backend Package
│   ├── __init__.py
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
├── tests/                    # Backend Unit & Integration Tests
├── run.py                    # Root Unified Runner
├── requirements.txt          # Python dependencies
└── pyproject.toml            # Project configuration & pytest settings
```
