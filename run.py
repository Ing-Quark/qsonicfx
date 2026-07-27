#!/usr/bin/env python3
"""
run.py
======
Q-SonicFX Unified Cloud Launcher for Railway, Render, and Fly.io
=================================================================

Railway routes public web traffic to $PORT.
This launcher launches BOTH FastAPI (internal) and Streamlit (public $PORT)
concurrently without blocking loops, ensuring Railway's health check
passes instantly (<1s).
"""
import os
import sys
import time
import subprocess
import signal

def main():
    port = os.getenv("PORT", "8000")
    backend_port = "8001"
    
    os.environ["API_URL"] = f"http://127.0.0.1:{backend_port}"
    
    # 1. Start FastAPI backend engine on internal port 8001
    print(f"[Q-SonicFX Launcher] Starting FastAPI engine on 127.0.0.1:{backend_port} ...", flush=True)
    backend_proc = subprocess.Popen([
        sys.executable, "-m", "uvicorn", "api_server:app",
        "--host", "127.0.0.1",
        "--port", backend_port,
        "--log-level", "info",
    ])

    # 2. Start Streamlit UI IMMEDIATELY on Railway public $PORT (no blocking delay!)
    print(f"[Q-SonicFX Launcher] Launching Streamlit dashboard on 0.0.0.0:{port} ...", flush=True)
    streamlit_cmd = [
        sys.executable, "-m", "streamlit", "run", "dashboard.py",
        "--server.address=0.0.0.0",
        f"--server.port={port}",
        "--server.headless=true",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false",
    ]
    streamlit_proc = subprocess.Popen(streamlit_cmd)
    
    procs = [backend_proc, streamlit_proc]

    def _shutdown(sig, frame):
        print(f"\n[Q-SonicFX Launcher] Shutdown signal {sig} received. Terminating processes...", flush=True)
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)

    try:
        # Wait on Streamlit process (the main web process bound to $PORT)
        streamlit_proc.wait()
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass

if __name__ == "__main__":
    main()
