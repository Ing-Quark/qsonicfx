#!/usr/bin/env python3
"""
run.py
======
Q-SonicFX Unified Cloud Launcher for Railway, Render, and Fly.io

Launches FastAPI backend engine on internal port 8001 and
Streamlit institutional dashboard on public $PORT in 1 clean process tree.
"""
import os
import sys
import time
import subprocess

def main():
    port = os.getenv("PORT", "8000")
    print(f"[Q-SonicFX Launcher] Starting FastAPI backend on 127.0.0.1:8001...")
    
    # 1. Start FastAPI backend engine on internal port 8001
    os.environ["API_URL"] = "http://127.0.0.1:8001"
    backend_proc = subprocess.Popen([
        sys.executable, "-m", "uvicorn", "api_server:app",
        "--host", "127.0.0.1", "--port", "8001", "--log-level", "info"
    ])

    time.sleep(1.5)

    # 2. Start Streamlit UI on Railway public $PORT
    print(f"[Q-SonicFX Launcher] Launching Streamlit dashboard on 0.0.0.0:{port}...")
    streamlit_cmd = [
        sys.executable, "-m", "streamlit", "run", "dashboard.py",
        "--server.address=0.0.0.0",
        f"--server.port={port}",
        "--server.headless=true"
    ]
    
    streamlit_proc = subprocess.Popen(streamlit_cmd)
    
    try:
        streamlit_proc.wait()
    except KeyboardInterrupt:
        streamlit_proc.terminate()
        backend_proc.terminate()

if __name__ == "__main__":
    main()
