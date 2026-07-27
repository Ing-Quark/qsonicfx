#!/usr/bin/env python3
"""
run.py
======
Q-SonicFX Unified Cloud Launcher — Railway / Render / Fly.io
=============================================================

Railway exposes ONE public port ($PORT) and runs a health check against it.
This launcher:
  1. Starts FastAPI engine on 0.0.0.0:$PORT  (Railway health check passes here)
  2. Starts Streamlit UI on 0.0.0.0:8501    (accessible via custom domain or /streamlit proxy)

If DASHBOARD_ONLY=true is set, only Streamlit runs (for separate Railway services).
If API_ONLY=true is set, only FastAPI runs.
"""
import os
import sys
import time
import subprocess
import signal

def main():
    port        = os.getenv("PORT", "8000")
    api_only    = os.getenv("API_ONLY", "").lower() in ("1", "true", "yes")
    dash_only   = os.getenv("DASHBOARD_ONLY", "").lower() in ("1", "true", "yes")

    procs = []

    if not dash_only:
        # ── 1. FastAPI engine on the PUBLIC $PORT (Railway health check) ──
        print(f"[Q-SonicFX] Starting FastAPI engine on 0.0.0.0:{port} ...", flush=True)
        os.environ["API_URL"] = f"http://127.0.0.1:{port}"
        backend_proc = subprocess.Popen([
            sys.executable, "-m", "uvicorn", "api_server:app",
            "--host", "0.0.0.0",
            "--port", port,
            "--log-level", "info",
            "--timeout-keep-alive", "30",
        ])
        procs.append(backend_proc)
        # Give FastAPI time to bind before Streamlit starts
        time.sleep(3.0)

    if not api_only:
        # ── 2. Streamlit dashboard on internal port 8501 ──────────────────
        # Accessible externally only if you add a second Railway service
        # or configure a reverse proxy.  Internal use: http://localhost:8501
        dash_port = os.getenv("STREAMLIT_PORT", "8501")
        print(f"[Q-SonicFX] Starting Streamlit dashboard on 0.0.0.0:{dash_port} ...", flush=True)
        streamlit_proc = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "dashboard.py",
            "--server.address=0.0.0.0",
            f"--server.port={dash_port}",
            "--server.headless=true",
            "--server.enableCORS=false",
            "--server.enableXsrfProtection=false",
        ])
        procs.append(streamlit_proc)

    if not procs:
        print("[Q-SonicFX] ERROR: No processes configured. Check API_ONLY/DASHBOARD_ONLY env vars.", flush=True)
        sys.exit(1)

    # ── Wait for any child to exit; kill siblings on exit ──────────────
    def _shutdown(sig, frame):
        print(f"\n[Q-SonicFX] Received signal {sig} — shutting down all processes.", flush=True)
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)

    try:
        # Wait on the first process (FastAPI = most critical)
        procs[0].wait()
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
