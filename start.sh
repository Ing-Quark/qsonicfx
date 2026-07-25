#!/bin/bash
# Q-SonicFX Production Startup Script for Railway & Cloud Containers

PORT_TO_USE="${PORT:-8000}"
RAW_SERVICE="${RAILWAY_SERVICE_NAME:-$SERVICE_TYPE}"
SERVICE_LOWER=$(echo "$RAW_SERVICE" | tr '[:upper:]' '[:lower:]')

echo "[Q-SonicFX] Starting process... (Detected Service: '$RAW_SERVICE', Port: $PORT_TO_USE)"

if [[ "$SERVICE_LOWER" == *"dashboard"* ]]; then
    echo "[Q-SonicFX] Mode: Streamlit Dashboard on port $PORT_TO_USE"
    exec streamlit run dashboard.py --server.address 0.0.0.0 --server.port $PORT_TO_USE --server.headless true
elif [[ "$SERVICE_LOWER" == *"api"* ]]; then
    echo "[Q-SonicFX] Mode: FastAPI Core Server on port $PORT_TO_USE"
    exec uvicorn api_server:app --host 0.0.0.0 --port $PORT_TO_USE --log-level info
else
    echo "[Q-SonicFX] Mode: Combined Server (FastAPI on internal port 8000, Streamlit on $PORT_TO_USE)"
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --log-level info &
    exec streamlit run dashboard.py --server.address 0.0.0.0 --server.port $PORT_TO_USE --server.headless true
fi
