#!/bin/bash
# Dedicated FastAPI Start Script for Railway
PORT_TO_USE="${PORT:-8000}"
echo "[Q-SonicFX API] Launching Uvicorn on 0.0.0.0:$PORT_TO_USE..."
exec uvicorn api_server:app --host 0.0.0.0 --port $PORT_TO_USE --log-level info
