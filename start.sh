#!/bin/bash
# Q-SonicFX Production Startup Script for Railway & Cloud Containers

PORT_TO_USE="${PORT:-8501}"
RAW_SERVICE="${RAILWAY_SERVICE_NAME:-$SERVICE_TYPE}"
SERVICE_LOWER=$(echo "$RAW_SERVICE" | tr '[:upper:]' '[:lower:]')

echo "[Q-SonicFX] Starting process... (Detected Service: '$RAW_SERVICE', Port: $PORT_TO_USE)"

if [[ "$SERVICE_LOWER" == *"dashboard"* ]]; then
    exec bash start_dashboard.sh
elif [[ "$SERVICE_LOWER" == *"api"* ]]; then
    exec bash start_api.sh
else
    echo "[Q-SonicFX] Combined Mode: Starting FastAPI backend on internal port 127.0.0.1:8001..."
    export API_URL="http://127.0.0.1:8001"
    uvicorn api_server:app --host 127.0.0.1 --port 8001 --log-level info &
    sleep 1
    echo "[Q-SonicFX] Launching Streamlit dashboard on public port $PORT_TO_USE..."
    exec streamlit run dashboard.py \
        --server.address 0.0.0.0 \
        --server.port $PORT_TO_USE \
        --server.headless true \
        --server.enableCORS false \
        --server.enableXsrfProtection false
fi
