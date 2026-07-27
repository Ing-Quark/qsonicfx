#!/usr/bin/env python3
"""
run.py
======
Q-SonicFX Unified Cloud Launcher for Railway, Render, and Fly.io
=================================================================

Railway exposes ONE public port ($PORT) and routes external web traffic to it.
This launcher:
  1. Starts FastAPI engine on internal port 8001 (127.0.0.1:8001).
  2. Waits for FastAPI to be fully online.
  3. Starts Streamlit institutional dashboard on public $PORT (0.0.0.0:$PORT).

This ensures visitors opening https://qsonicfx.up.railway.app see the full Streamlit UI!
"""
import os
import sys
import time
import subprocess
import urllib.request
import signal

def wait_for_backend(url: str, timeout: float = 25.0) -> bool:
    """Poll backend until it responds with HTTP 200."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(f"{url}/status", headers={"User-Agent": "Q-SonicFX-Launcher"})
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False

def main():
    port = os.getenv("PORT", "8000")
    backend_port = "8001"
    
    print(f"[Q-SonicFX Launcher] Starting FastAPI engine on 127.0.0.1:{backend_port} ...", flush=True)
    os.environ["API_URL"] = f"http://127.0.0.1:{backend_port}"
    
    # 1. Start FastAPI backend engine on internal port 8001
    backend_proc = subprocess.Popen([
        sys.executable, "-m", "uvicorn", "api_server:app",
        "--host", "127.0.0.1",
        "--port", backend_port,
        "--log-level", "info",
    ])

    print("[Q-SonicFX Launcher] Waiting for FastAPI backend to initialize...", flush=True)
    if wait_for_backend(f"http://127.0.0.1:{backend_port}"):
        print("[Q-SonicFX Launcher] FastAPI engine is online & healthy!", flush=True)
    else:
        print("[Q-SonicFX Launcher] FastAPI engine initialization continuing, proceeding with Streamlit launch...", flush=True)

    # 2. Start Streamlit UI on Railway public $PORT
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
        # Streamlit is the main public process on $PORT
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
