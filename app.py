
from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime

import streamlit as st

from career_tools import analyze_resume, analyze_skill_gap, generate_roadmap, search_jobs
from config import ConfigError, configure_logging, get_settings
from errors import CareerAdvisorError, safe_error_message
from llm_providers import ChatMessage, GeminiProvider
from rag_pipeline import build_vector_store, load_existing_vector_store
from rag_service import RagService
from resume_pipeline import extract_resume_text
from tracing import configure_langsmith

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Career Advisor",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- "Night Navigator" design system: a dark, compass-themed reskin ---
# Token system: void/surface/elevated backgrounds, gold = destination/CTA,
# violet = in-progress/interactive, teal = matched/success, rose = gap/error.
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500&display=swap');

        :root {
            --cd-void: #08080c;
            --cd-surface: #121218;
            --cd-elevated: #191922;
            --cd-elevated-2: #1f1f2b;
            --cd-border: rgba(255,255,255,0.08);
            --cd-border-strong: rgba(255,255,255,0.14);
            --cd-text: #f0f0f5;
            --cd-muted: #9a9aab;
            --cd-faint: #6b6b7d;
            --cd-gold: #f2b705;
            --cd-gold-dim: rgba(242,183,5,0.14);
            --cd-violet: #7c6fff;
            --cd-violet-dim: rgba(124,111,255,0.14);
            --cd-teal: #00d9b5;
            --cd-teal-dim: rgba(0,217,181,0.14);
            --cd-rose: #ff6b6b;
            --cd-rose-dim: rgba(255,107,107,0.14);
        }

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        /* ---- App shell: near-black with a slow-drifting aurora ---- */
        .stApp {
            background: var(--cd-void);
            position: relative;
            overflow-x: hidden;
        }
        .stApp::before {
            content: "";
            position: fixed;
            inset: -20%;
            z-index: 0;
            pointer-events: none;
            background:
                radial-gradient(38% 30% at 15% 8%, rgba(124,111,255,0.16), transparent 60%),
                radial-gradient(32% 26% at 88% 18%, rgba(242,183,5,0.10), transparent 60%),
                radial-gradient(30% 24% at 60% 92%, rgba(0,217,181,0.08), transparent 60%);
            animation: cd-drift 26s ease-in-out infinite alternate;
            filter: blur(10px);
        }
        @keyframes cd-drift {
            0%   { transform: translate(0%, 0%) scale(1); }
            50%  { transform: translate(2%, -3%) scale(1.06); }
            100% { transform: translate(-3%, 2%) scale(1.02); }
        }
        @media (prefers-reduced-motion: reduce) {
            .stApp::before { animation: none; }
        }

        .block-container { padding-top: 2.2rem; max-width: 1020px; position: relative; z-index: 1; }

        /* ---- Hero / header ---- */
        .cd-hero {
            display: flex; align-items: center; gap: 0.9rem;
            padding: 1.4rem 1.6rem;
            margin-bottom: 0.6rem;
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(124,111,255,0.10), rgba(242,183,5,0.06));
            border: 1px solid var(--cd-border);
        }
        .cd-hero-icon {
            font-size: 2.1rem;
            filter: drop-shadow(0 0 14px rgba(242,183,5,0.45));
            flex-shrink: 0;
        }
        .cd-hero-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.9rem; font-weight: 700; color: var(--cd-text);
            letter-spacing: -0.02em; line-height: 1.15; margin: 0;
        }
        .career-advisor-tagline {
            color: var(--cd-muted); font-size: 0.98rem; margin-top: 0.25rem; line-height: 1.5;
        }

        /* ---- Headings ---- */
        h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.01em; color: var(--cd-text) !important; }
        h3 { font-size: 1.15rem !important; }

        /* ---- Badges ---- */
        .career-advisor-badge {
            display: inline-block; padding: 0.22rem 0.75rem; border-radius: 999px;
            font-size: 0.74rem; font-weight: 600; margin-right: 0.4rem;
            letter-spacing: 0.02em; text-transform: uppercase;
            border: 1px solid transparent;
        }
        .badge-ok   { background: var(--cd-teal-dim);   color: var(--cd-teal);   border-color: rgba(0,217,181,0.3); }
        .badge-warn { background: var(--cd-gold-dim);   color: var(--cd-gold);   border-color: rgba(242,183,5,0.3); }
        .badge-off  { background: rgba(255,255,255,0.05); color: var(--cd-faint); border-color: var(--cd-border); }

        /* ---- Sidebar ---- */
        [data-testid="stSidebar"] {
            background: var(--cd-surface);
            border-right: 1px solid var(--cd-border);
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            font-size: 0.95rem !important; color: var(--cd-muted) !important;
            text-transform: uppercase; letter-spacing: 0.06em;
        }

        /* ---- Tabs ---- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px; background: var(--cd-surface); padding: 5px;
            border-radius: 12px; border: 1px solid var(--cd-border);
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 9px; color: var(--cd-muted); font-weight: 500;
            padding: 8px 16px; background: transparent;
        }
        .stTabs [aria-selected="true"] {
            background: var(--cd-elevated-2) !important;
            color: var(--cd-text) !important;
            box-shadow: inset 0 0 0 1px var(--cd-border-strong);
        }

        /* ---- Buttons ---- */
        .stButton > button {
            background: linear-gradient(135deg, var(--cd-gold), #ffce3a);
            color: #1a1200; font-weight: 700; border: none; border-radius: 10px;
            padding: 0.55rem 1.1rem; transition: transform 0.15s ease, box-shadow 0.15s ease;
            box-shadow: 0 4px 14px rgba(242,183,5,0.18);
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(242,183,5,0.3);
        }
        .stButton > button:active { transform: translateY(0); }

        /* ---- Inputs ---- */
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea,
        [data-testid="stNumberInput"] input,
        .stSelectbox div[data-baseweb="select"] > div {
            background: var(--cd-surface) !important;
            border: 1px solid var(--cd-border) !important;
            color: var(--cd-text) !important;
            border-radius: 9px !important;
        }
        [data-testid="stTextInput"] input:focus,
        [data-testid="stTextArea"] textarea:focus {
            border-color: var(--cd-violet) !important;
            box-shadow: 0 0 0 1px var(--cd-violet) !important;
        }

        /* ---- File uploader ---- */
        [data-testid="stFileUploaderDropzone"] {
            background: var(--cd-surface);
            border: 1.5px dashed var(--cd-border-strong);
            border-radius: 12px;
        }

        /* ---- Expanders (used for jobs, roadmap milestones, sources) ---- */
        [data-testid="stExpander"] {
            background: var(--cd-elevated);
            border: 1px solid var(--cd-border);
            border-radius: 12px;
            overflow: hidden;
        }
        [data-testid="stExpander"] summary { color: var(--cd-text) !important; font-weight: 500; }

        /* ---- Alerts (st.info / success / warning / error) ---- */
        [data-testid="stAlert"] {
            background: var(--cd-elevated) !important;
            border: 1px solid var(--cd-border);
            border-radius: 12px;
        }

        /* ---- Chat ---- */
        [data-testid="stChatMessage"] {
            background: var(--cd-elevated);
            border: 1px solid var(--cd-border);
            border-radius: 14px;
        }
        [data-testid="stChatInput"] {
            background: var(--cd-surface);
            border: 1px solid var(--cd-border-strong);
            border-radius: 14px;
        }
        [data-testid="stChatInput"] textarea { color: var(--cd-text) !important; }

        /* ---- Misc ---- */
        hr { border-color: var(--cd-border) !important; }
        [data-testid="stCaptionContainer"] { color: var(--cd-faint) !important; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; }
        code { color: var(--cd-teal) !important; background: rgba(0,217,181,0.08) !important; }

        ::-webkit-scrollbar { width: 10px; height: 10px; }
        ::-webkit-scrollbar-track { background: var(--cd-void); }
        ::-webkit-scrollbar-thumb { background: var(--cd-elevated-2); border-radius: 8px; border: 2px solid var(--cd-void); }
        ::-webkit-scrollbar-thumb:hover { background: var(--cd-border-strong); }
    </style>
    """,
    unsafe_allow_html=True,
)


def _badge(label: str, kind: str) -> str:
    return f'<span class="career-advisor-badge badge-{kind}">{label}</span>'


@st.cache_resource(show_spinner=False)
def _load_settings():
    """Cached so config validation runs once per process, not once per rerun."""
    settings = get_settings()
    configure_logging(settings)
    configure_langsmith(settings)
    return settings


def _init_session_state() -> None:
    defaults = {
        "messages": [],  # list[dict]: role, content, timestamp
        "service": None,
        "vectorstore": None,
        "document_loaded": False,
        "init_error": None,
        "resume_text": "",
        "resume_filename": "",
        "resume_analysis": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _history_as_chat_messages() -> list[ChatMessage]:
    return [
        ChatMessage(role=m["role"], content=m["content"])
        for m in st.session_state.messages
        if m["role"] in ("user", "assistant")
    ]


def _build_service(settings) -> RagService:
    provider = GeminiProvider(
        api_key=settings.google_api_key,
        model_name=settings.gemini_model,
        timeout_seconds=settings.llm_request_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )
    return RagService(settings, provider)


def _get_llm_provider(settings) -> GeminiProvider:
    """The RagService already owns a provider; the career tools reuse it
    directly rather than instantiating a second client."""
    service: RagService = st.session_state.service
    return service.llm


def _render_resume_tab(settings) -> None:
    st.subheader("📄 Resume Analyzer")
    st.caption("Upload your resume (PDF, DOCX, or TXT) to extract skills and get feedback.")

    uploaded_resume = st.file_uploader(
        "Upload your resume", type=["pdf", "docx", "txt"], key="resume_uploader"
    )
    target_role_for_resume = st.text_input(
        "Target role (optional)",
        key="resume_target_role",
        placeholder="e.g. Data Analyst — leave blank to get role suggestions instead",
    )

    if uploaded_resume and st.button("Analyze resume", use_container_width=True):
        with st.spinner("Reading and analyzing your resume…"):
            suffix = os.path.splitext(uploaded_resume.name)[1] or ".pdf"
            try:
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(uploaded_resume.getvalue())
                    tmp_path = tmp.name
                try:
                    text = extract_resume_text(tmp_path, settings)
                    st.session_state.resume_text = text
                    st.session_state.resume_filename = uploaded_resume.name
                    result = analyze_resume(
                        text,
                        _get_llm_provider(settings),
                        target_role=target_role_for_resume or None,
                    )
                    st.session_state.resume_analysis = result
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            except CareerAdvisorError as exc:
                st.error(exc.user_message)
            except Exception as exc:
                st.error(safe_error_message(exc))

    if st.session_state.resume_analysis:
        result = st.session_state.resume_analysis
        st.success(f"Analyzed: {st.session_state.resume_filename}")
        st.markdown(f"**Experience summary:** {result.experience_summary}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Extracted skills**")
            st.write(", ".join(result.extracted_skills) or "—")
            st.markdown("**Strengths**")
            for s in result.strengths:
                st.markdown(f"- {s}")
        with col2:
            st.markdown("**Gaps / improvements**")
            for g in result.gaps_or_improvements:
                st.markdown(f"- {g}")
            st.markdown("**Suggested target roles**")
            for r in result.suggested_target_roles:
                st.markdown(f"- {r}")


def _render_skill_gap_tab(settings) -> None:
    st.subheader("🎯 Skill Gap Analyzer")
    st.caption("Compare your current skills against a target role.")

    default_skills = (
        ", ".join(st.session_state.resume_analysis.extracted_skills)
        if st.session_state.resume_analysis
        else ""
    )
    skills_input = st.text_area(
        "Your current skills (comma-separated)",
        value=default_skills,
        height=80,
        placeholder="e.g. Python, Excel, SQL basics",
    )
    target_role = st.text_input(
        "Target role", placeholder="e.g. Data Analyst", key="gap_target_role"
    )

    if st.button("Analyze skill gap", use_container_width=True):
        skills = [s.strip() for s in skills_input.split(",") if s.strip()]
        with st.spinner("Comparing your skills to the target role…"):
            try:
                result = analyze_skill_gap(skills, target_role, _get_llm_provider(settings))
                st.info(result.summary)
                readiness_kind = {"high": "ok", "medium": "warn", "low": "off"}.get(
                    result.overall_readiness, "off"
                )
                st.markdown(
                    _badge(f"Readiness: {result.overall_readiness}", readiness_kind),
                    unsafe_allow_html=True,
                )

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**✅ Matched**")
                    for s in result.matched_skills:
                        st.markdown(f"- {s}")
                with col2:
                    st.markdown("**🟡 Partial**")
                    for s in result.partially_met_skills:
                        st.markdown(f"- {s}")
                with col3:
                    st.markdown("**❌ Missing**")
                    for s in result.missing_skills:
                        st.markdown(f"- {s}")
            except CareerAdvisorError as exc:
                st.error(exc.user_message)
            except Exception as exc:
                st.error(safe_error_message(exc))


def _render_job_search_tab(settings) -> None:
    st.subheader("🔎 Job Search")
    st.caption(
        "Searches live listings via Adzuna if configured, otherwise a small bundled sample dataset."
        if not settings.has_job_search_api
        else "Searching live listings via Adzuna."
    )

    col1, col2 = st.columns(2)
    with col1:
        role = st.text_input("Role", placeholder="e.g. Frontend Engineer", key="job_role")
        location = st.text_input(
            "Location (optional)", placeholder="e.g. Remote", key="job_location"
        )
    with col2:
        skills_input = st.text_input("Skills (comma-separated, optional)", key="job_skills")
        experience_level = st.selectbox(
            "Experience level (optional)", ["", "entry", "mid", "senior"], key="job_exp"
        )

    if st.button("Search jobs", use_container_width=True):
        skills = [s.strip() for s in skills_input.split(",") if s.strip()]
        with st.spinner("Searching…"):
            try:
                jobs = search_jobs(
                    role,
                    settings,
                    skills=skills,
                    location=location or None,
                    experience_level=experience_level or None,
                )
                if not jobs:
                    st.warning("No matching jobs found. Try broadening your search.")
                for job in jobs:
                    with st.expander(f"{job.title} · {job.company} · {job.location}"):
                        if job.skills:
                            st.caption("Skills: " + ", ".join(job.skills))
                        if job.description:
                            st.write(job.description)
                        if job.url:
                            st.markdown(f"[View listing]({job.url})")
            except CareerAdvisorError as exc:
                st.error(exc.user_message)
            except Exception as exc:
                st.error(safe_error_message(exc))


def _render_roadmap_tab(settings) -> None:
    st.subheader("🗺️ Career Roadmap Generator")
    st.caption("Generate a structured, milestone-based learning roadmap toward a target role.")

    default_skills = (
        ", ".join(st.session_state.resume_analysis.extracted_skills)
        if st.session_state.resume_analysis
        else ""
    )
    skills_input = st.text_area(
        "Your current skills (comma-separated)",
        value=default_skills,
        height=80,
        key="roadmap_skills",
    )
    col1, col2 = st.columns([3, 1])
    with col1:
        target_role = st.text_input(
            "Target role", placeholder="e.g. Frontend Engineer", key="roadmap_target_role"
        )
    with col2:
        timeframe = st.number_input(
            "Months", min_value=1, max_value=36, value=6, key="roadmap_timeframe"
        )

    if st.button("Generate roadmap", use_container_width=True):
        skills = [s.strip() for s in skills_input.split(",") if s.strip()]
        with st.spinner("Building your roadmap…"):
            try:
                result = generate_roadmap(
                    skills,
                    target_role,
                    _get_llm_provider(settings),
                    timeframe_months=int(timeframe),
                )
                st.info(result.summary)
                for i, milestone in enumerate(result.milestones, start=1):
                    with st.expander(
                        f"{i}. {milestone.title} ({milestone.duration})", expanded=(i == 1)
                    ):
                        if milestone.focus_skills:
                            st.caption("Focus skills: " + ", ".join(milestone.focus_skills))
                        for action in milestone.actions:
                            st.markdown(f"- {action}")
            except CareerAdvisorError as exc:
                st.error(exc.user_message)
            except Exception as exc:
                st.error(safe_error_message(exc))


def _render_tools(settings) -> None:
    job_tab, gap_tab, resume_tab, roadmap_tab = st.tabs(
        ["🔎 Job Search", "🎯 Skill Gap", "📄 Resume Analyzer", "🗺️ Roadmap"]
    )
    with job_tab:
        _render_job_search_tab(settings)
    with gap_tab:
        _render_skill_gap_tab(settings)
    with resume_tab:
        _render_resume_tab(settings)
    with roadmap_tab:
        _render_roadmap_tab(settings)


def main() -> None:
    _init_session_state()

    try:
        settings = _load_settings()
    except ConfigError as exc:
        st.title("🧭 Career Advisor")
        st.error(f"Configuration problem: {exc}")
        st.info(
            "Copy **.env.example** to **.env**, add your Google Gemini API key, "
            "and restart the app."
        )
        st.code("cp .env.example .env", language="bash")
        return

    if st.session_state.service is None:
        try:
            st.session_state.service = _build_service(settings)
        except CareerAdvisorError as exc:
            st.session_state.init_error = exc.user_message
        except Exception as exc:  # pragma: no cover - defensive
            st.session_state.init_error = safe_error_message(exc)

    # Attempt to reuse a previously persisted vector store on first load,
    # instead of requiring a re-upload every session.
    if st.session_state.vectorstore is None and not st.session_state.document_loaded:
        existing = load_existing_vector_store(settings)
        if existing is not None:
            st.session_state.vectorstore = existing
            st.session_state.document_loaded = True
            if st.session_state.service:
                st.session_state.service.set_vectorstore(existing)

    with st.sidebar:
        st.header("⚙️ Configuration")

        if settings.google_api_key:
            st.markdown(_badge("API key loaded", "ok"), unsafe_allow_html=True)
        st.caption(f"Environment: `{settings.app_env}` · Model: `{settings.gemini_model}`")

        st.divider()
        st.subheader("📄 Reference Document")

        uploaded_file = st.file_uploader(
            "Upload a PDF for document-grounded answers",
            type=["pdf"],
            help="Optional — the assistant works without a document too.",
        )

        if uploaded_file and st.button("Process document", use_container_width=True):
            with st.spinner("Reading and indexing document…"):
                try:
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    try:
                        vectorstore = build_vector_store(tmp_path, settings)
                        st.session_state.vectorstore = vectorstore
                        st.session_state.document_loaded = True
                        if st.session_state.service:
                            st.session_state.service.set_vectorstore(vectorstore)
                        st.success("Document indexed successfully.")
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                except CareerAdvisorError as exc:
                    st.error(exc.user_message)
                except Exception as exc:
                    st.error(safe_error_message(exc))

        if st.session_state.document_loaded:
            st.markdown(_badge("Document indexed", "ok"), unsafe_allow_html=True)
        else:
            st.markdown(_badge("General mode (no document)", "off"), unsafe_allow_html=True)

        st.divider()
        st.subheader("👤 Your Background")
        student_profile = st.text_area(
            "Helps personalize answers to your situation",
            value=st.session_state.get("student_profile", ""),
            height=90,
            placeholder="e.g. Final-year CS student interested in ML and data careers",
        )
        st.session_state["student_profile"] = student_profile

        st.divider()
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.markdown(
        """
        <div class="cd-hero">
            <div class="cd-hero-icon">🧭</div>
            <div>
                <p class="cd-hero-title">Career Advisor</p>
                <p class="career-advisor-tagline">Practical, grounded career guidance —
                ask a general question, upload a document for answers tied to it, or
                use the Career Tools for job search, resume review, skill gaps, and roadmaps.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.init_error:
        st.error(st.session_state.init_error)
        return

    service: RagService = st.session_state.service
    if service is None:
        st.warning("Initializing…")
        return

    chat_tab, tools_tab = st.tabs(["💬 Chat", "🧰 Career Tools"])

    with tools_tab:
        _render_tools(settings)

    with chat_tab:
        _render_chat(settings, service)


def _render_chat(settings, service: RagService) -> None:
    if not st.session_state.messages:
        st.info(
            '👋 **New here?** Ask something like *"What skills matter most for '
            'an entry-level data analyst role?"* — or upload a PDF in the sidebar '
            "first if you want answers grounded in a specific document."
        )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📚 Sources used ({len(msg['sources'])})"):
                    for s in msg["sources"]:
                        page_label = f"page {s.page}" if s.page is not None else "unknown page"
                        st.caption(f"**{page_label}** · relevance {s.relevance_score:.2f}")
                        preview = s.content[:400] + ("…" if len(s.content) > 400 else "")
                        st.text(preview)
            st.caption(msg.get("timestamp", ""))

    query = st.chat_input("Ask about careers, skills, or your uploaded document…")
    if not query:
        return

    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.messages.append({"role": "user", "content": query, "timestamp": timestamp})
    with st.chat_message("user"):
        st.markdown(query)
        st.caption(timestamp)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            history = _history_as_chat_messages()[:-1]  # exclude the message just added
            try:
                result = service.answer(
                    query,
                    history=history,
                    student_profile=st.session_state.get("student_profile", ""),
                )
                st.markdown(result.text)
                if result.used_retrieval and result.sources:
                    with st.expander(f"📚 Sources used ({len(result.sources)})"):
                        for s in result.sources:
                            page_label = f"page {s.page}" if s.page is not None else "unknown page"
                            st.caption(f"**{page_label}** · relevance {s.relevance_score:.2f}")
                            preview = s.content[:400] + ("…" if len(s.content) > 400 else "")
                            st.text(preview)
                response_timestamp = datetime.now().strftime("%H:%M:%S")
                st.caption(response_timestamp)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result.text,
                        "timestamp": response_timestamp,
                        "sources": result.sources if result.used_retrieval else None,
                    }
                )
            except CareerAdvisorError as exc:
                st.error(exc.user_message)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": exc.user_message,
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                    }
                )
            except Exception as exc:
                message = safe_error_message(exc)
                st.error(message)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": message,
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                    }
                )


if __name__ == "__main__":
    main()
