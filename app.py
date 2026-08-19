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

# --- Minimal, intentional theming (avoids the raw Streamlit-default look) ---
st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; max-width: 1000px; }
        [data-testid="stChatMessage"] { border-radius: 12px; }
        .career-advisor-tagline { color: #6b7280; font-size: 1.05rem; margin-top: -0.6rem; }
        .career-advisor-badge {
            display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px;
            font-size: 0.78rem; font-weight: 600; margin-right: 0.4rem;
        }
        .badge-ok { background: #dcfce7; color: #166534; }
        .badge-warn { background: #fef9c3; color: #854d0e; }
        .badge-off { background: #f3f4f6; color: #6b7280; }
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

    st.title("🧭 Career Advisor")
    st.markdown(
        '<p class="career-advisor-tagline">Practical, grounded career guidance — '
        "ask a general question, upload a document for answers tied to it, or "
        "use the Career Tools for job search, resume review, skill gaps, and roadmaps.</p>",
        unsafe_allow_html=True,
    )
    st.divider()

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
