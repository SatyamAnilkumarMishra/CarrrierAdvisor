#!/usr/bin/env python3
"""Convenience runner for Career Advisor: api / frontend / cli / mcp / install / setup / status."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def check_requirements() -> bool:
    try:
        import fastapi  # noqa: F401
        import google.genai  # noqa: F401
        import langchain  # noqa: F401
        import uvicorn  # noqa: F401

        return True
    except ImportError as exc:
        print(f"❌ Missing required packages: {exc}")
        print("📦 Install them with: pip install -r requirements.txt")
        return False


def check_env() -> bool:
    if not os.path.exists(".env"):
        print("❌ .env file not found")
        print("🔧 Run: python run.py setup   (then edit .env with your API key)")
        return False

    try:
        from backend.config import ConfigError, get_settings

        get_settings()
    except ConfigError as exc:
        print(f"❌ Configuration problem: {exc}")
        return False
    except Exception as exc:  # pragma: no cover
        print(f"❌ Unexpected configuration error: {exc}")
        return False

    print("✅ Environment configured correctly")
    return True


def run_api() -> None:
    print("🚀 Starting FastAPI backend server on http://localhost:8000…")
    try:
        subprocess.run(
            [sys.executable, "-m", "uvicorn", "backend.server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"❌ Failed to start FastAPI server: {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 FastAPI server stopped")


def run_frontend() -> None:
    print("🚀 Starting Next.js frontend on http://localhost:3000…")
    frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
    try:
        subprocess.run(["npm", "run", "dev"], cwd=frontend_dir, shell=True, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"❌ Failed to start Next.js frontend: {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Frontend stopped")


def run_mcp() -> None:
    print("🚀 Starting Career Advisor MCP server…")
    try:
        subprocess.run([sys.executable, "-m", "backend.mcp_server"], check=True)
    except subprocess.CalledProcessError as exc:
        print(f"❌ Failed to start MCP server: {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 MCP server stopped")


def run_cli() -> None:
    print("🚀 Starting CLI interface…")
    try:
        subprocess.run([sys.executable, "-m", "backend.main"], check=True)
    except subprocess.CalledProcessError as exc:
        print(f"❌ Failed to start CLI: {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 CLI stopped")


def install_requirements() -> None:
    print("📦 Installing backend requirements…")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True
        )
        print("✅ Backend requirements installed successfully")
    except subprocess.CalledProcessError as exc:
        print(f"❌ Failed to install requirements: {exc}")
        sys.exit(1)


def create_sample_env() -> None:
    if os.path.exists(".env"):
        print("⚠️  .env already exists")
        if input("Overwrite it? (y/N): ").strip().lower() != "y":
            print("❌ Cancelled")
            return

    with open(".env.example") as src, open(".env", "w") as dst:
        dst.write(src.read())

    print("✅ Created .env from .env.example")
    print("🔧 Edit .env and add your Google API key")


def show_status() -> None:
    print("📊 Career Advisor Project Status")
    print("=" * 40)

    for file in [
        "requirements.txt",
        "backend/__init__.py",
        "backend/server.py",
        "backend/main.py",
        "backend/llm_providers.py",
        "backend/rag_pipeline.py",
        "backend/rag_service.py",
        "backend/config.py",
        "backend/career_tools.py",
        "backend/resume_pipeline.py",
        "backend/mcp_server.py",
        "backend/tracing.py",
        "backend/evaluation.py",
        "frontend/package.json",
    ]:
        print(f"✅ {file}" if os.path.exists(file) else f"❌ {file}")

    if os.path.exists(".env"):
        print("✅ .env")
        check_env()
    else:
        print("❌ .env (run: python run.py setup)")

    print("\n📦 Package status:")
    for pkg in [
        "fastapi",
        "uvicorn",
        "google-genai",
        "langchain",
        "chromadb",
        "python-dotenv",
        "langsmith",
        "mcp",
        "docx",
    ]:
        module = {"google-genai": "google.genai", "python-dotenv": "dotenv"}.get(
            pkg, pkg.replace("-", "_")
        )
        try:
            __import__(module)
            print(f"✅ {pkg}")
        except ImportError:
            print(f"❌ {pkg}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Career Advisor project runner")
    parser.add_argument(
        "command",
        nargs="?",
        choices=["api", "frontend", "cli", "mcp", "install", "setup", "status", "doctor"],
    )
    args = parser.parse_args()

    if not args.command:
        print("🧭 Career Advisor")
        print("Usage: python run.py [api|frontend|cli|mcp|install|setup|status]")
        print("  - api:      Launch FastAPI backend server (http://localhost:8000)")
        print("  - frontend: Launch Next.js web application (http://localhost:3000)")
        print("  - cli:      Launch interactive terminal CLI")
        print("  - mcp:      Launch Model Context Protocol server")
        print("  - status:   Verify environment and package dependencies")
        print("  - doctor:   Live preflight — calls Gemini with your key to prove")
        print("              the app can actually generate responses")
        print("  - install:  Install Python dependencies from requirements.txt")
        print("  - setup:    Initialize .env file from .env.example")
        return

    if args.command == "doctor":
        from backend.doctor import main as doctor_main

        sys.exit(doctor_main())
    elif args.command == "install":
        install_requirements()
    elif args.command == "setup":
        create_sample_env()
    elif args.command == "status":
        show_status()
    elif args.command == "api":
        if not check_requirements() or not check_env():
            sys.exit(1)
        run_api()
    elif args.command == "frontend":
        run_frontend()
    elif args.command == "cli":
        if not check_requirements() or not check_env():
            sys.exit(1)
        run_cli()
    elif args.command == "mcp":
        if not check_requirements() or not check_env():
            sys.exit(1)
        run_mcp()


if __name__ == "__main__":
    main()
