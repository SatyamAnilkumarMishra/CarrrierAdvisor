"""Bundled sample job listings.

`career_tools.search_jobs()` uses these as a fallback when no live job-search
API is configured (see `Settings.has_job_search_api`), so the Job Search
Tool — and the MCP server that exposes it — works out of the box without
requiring an API key, matching the rest of this app's "degrade gracefully"
philosophy (see `rag_pipeline.py`'s general-mode fallback).

This is a small, static, illustrative dataset — not a live feed. Swap in a
real provider by setting `ADZUNA_APP_ID` / `ADZUNA_APP_KEY` in `.env`.
"""

from __future__ import annotations

SAMPLE_JOBS = [
    {
        "title": "Junior Data Analyst",
        "company": "Northwind Analytics",
        "location": "Remote",
        "experience_level": "entry",
        "skills": ["SQL", "Excel", "Python", "Data Visualization", "Tableau"],
        "description": "Analyze customer datasets, build dashboards, and support reporting for the growth team.",
        "url": "https://example.com/jobs/junior-data-analyst",
    },
    {
        "title": "Data Analyst",
        "company": "Harborline Retail",
        "location": "Chicago, IL",
        "experience_level": "mid",
        "skills": ["SQL", "Python", "Power BI", "Statistics"],
        "description": "Own weekly sales reporting and ad-hoc analysis for merchandising leadership.",
        "url": "https://example.com/jobs/data-analyst-harborline",
    },
    {
        "title": "Frontend Engineer",
        "company": "Bluepeak Software",
        "location": "Remote",
        "experience_level": "mid",
        "skills": ["JavaScript", "React", "CSS", "TypeScript", "Testing"],
        "description": "Build and maintain customer-facing React features in a small product squad.",
        "url": "https://example.com/jobs/frontend-engineer-bluepeak",
    },
    {
        "title": "Junior Frontend Developer",
        "company": "Coastline Media",
        "location": "Austin, TX",
        "experience_level": "entry",
        "skills": ["HTML", "CSS", "JavaScript", "React"],
        "description": "Support the web team building marketing pages and internal tools.",
        "url": "https://example.com/jobs/junior-frontend-coastline",
    },
    {
        "title": "Machine Learning Engineer",
        "company": "Fernwood AI",
        "location": "Remote",
        "experience_level": "mid",
        "skills": ["Python", "PyTorch", "Machine Learning", "MLOps", "SQL"],
        "description": "Train and ship ML models supporting recommendation features.",
        "url": "https://example.com/jobs/ml-engineer-fernwood",
    },
    {
        "title": "Software Engineer, Backend",
        "company": "Ridgeline Systems",
        "location": "New York, NY",
        "experience_level": "mid",
        "skills": ["Python", "Django", "PostgreSQL", "APIs", "Docker"],
        "description": "Design and maintain backend services powering our platform.",
        "url": "https://example.com/jobs/backend-engineer-ridgeline",
    },
    {
        "title": "Product Manager, Associate",
        "company": "Lumen Health",
        "location": "Boston, MA",
        "experience_level": "entry",
        "skills": ["Product Strategy", "SQL", "Communication", "Roadmapping"],
        "description": "Support feature discovery and prioritization for a clinical-tools product line.",
        "url": "https://example.com/jobs/associate-pm-lumen",
    },
    {
        "title": "UX Designer",
        "company": "Coastline Media",
        "location": "Remote",
        "experience_level": "mid",
        "skills": ["Figma", "User Research", "Prototyping", "Design Systems"],
        "description": "Own end-to-end design for two product surfaces alongside engineering.",
        "url": "https://example.com/jobs/ux-designer-coastline",
    },
    {
        "title": "DevOps Engineer",
        "company": "Ridgeline Systems",
        "location": "Remote",
        "experience_level": "senior",
        "skills": ["AWS", "Kubernetes", "Terraform", "CI/CD", "Python"],
        "description": "Own infrastructure reliability and deployment pipelines for a growing platform team.",
        "url": "https://example.com/jobs/devops-ridgeline",
    },
    {
        "title": "Marketing Data Analyst",
        "company": "Northwind Analytics",
        "location": "Remote",
        "experience_level": "entry",
        "skills": ["Excel", "Google Analytics", "SQL", "A/B Testing"],
        "description": "Measure campaign performance and build self-serve reporting for marketing.",
        "url": "https://example.com/jobs/marketing-analyst-northwind",
    },
]
