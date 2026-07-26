#!/bin/bash
# Q-SonicFX Production Startup Script for Railway & Cloud Containers

PORT_TO_USE="${PORT:-8000}"
RAW_SERVICE="${RAILWAY_SERVICE_NAME:-$SERVICE_TYPE}"
SERVICE_LOWER=$(echo "$RAW_SERVICE" | tr '[:upper:]' '[:lower:]')

echo "[Q-SonicFX] Starting process... (Detected Service: '$RAW_SERVICE', Port: $PORT_TO_USE)"

if [[ "$SERVICE_LOWER" == *"dashboard"* ]]; then
    exec bash start_dashboard.sh
elif [[ "$SERVICE_LOWER" == *"api"* ]]; then
    exec bash start_api.sh
else
    echo "[Q-SonicFX] Combined Mode: Starting FastAPI backend on port 8000 and Streamlit on port $PORT_TO_USE..."
    uvicorn api_server:app --host 127.0.0.1 --port 8000 --log-level info &
    sleep 2
    export API_URL="http://127.0.0.1:8000"
    exec bash start_dashboard.sh
fi
