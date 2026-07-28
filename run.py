#!/usr/bin/env python3
"""
run.py
======
Q-SonicFX Unified Cloud Launcher for Railway, Render, and Fly.io
=================================================================

Launches FastAPI backend engine on internal port 8001 (127.0.0.1:8001)
and replaces the launcher process with Streamlit dashboard on public $PORT
using os.execvp so Streamlit becomes the primary container process.
"""
import os
import sys
import subprocess

def main():
    port = os.getenv("PORT", "8000")
    backend_port = "8001"
    
    os.environ["API_URL"] = f"http://127.0.0.1:{backend_port}"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    
    # 1. Start FastAPI backend engine in background on port 8001
    print(f"[Q-SonicFX Launcher] Spawning FastAPI backend on 127.0.0.1:{backend_port} ...", flush=True)
    subprocess.Popen([
        sys.executable, "-m", "uvicorn", "api_server:app",
        "--host", "127.0.0.1",
        "--port", backend_port,
        "--log-level", "info",
    ])

    # 2. Exec Streamlit directly to replace this process on public $PORT
    print(f"[Q-SonicFX Launcher] Executing Streamlit dashboard on 0.0.0.0:{port} ...", flush=True)
    cmd = [
        sys.executable, "-m", "streamlit", "run", "dashboard.py",
        "--server.address=0.0.0.0",
        f"--server.port={port}",
        "--server.headless=true",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false",
    ]
    
    # Replaces current process with streamlit binary/module
    os.execvp(cmd[0], cmd)

if __name__ == "__main__":
    main()
