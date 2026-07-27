#!/bin/bash
# Q-SonicFX Production Startup Script for Railway & Cloud Containers

PORT_TO_USE="${PORT:-8000}"

echo "[Q-SonicFX] Starting combined engine... (Public Port: $PORT_TO_USE)"

# 1. Start FastAPI backend on internal port 8001
export API_URL="http://127.0.0.1:8001"
uvicorn api_server:app --host 127.0.0.1 --port 8001 --log-level info &

# 2. Launch Streamlit dashboard on Railway public PORT
echo "[Q-SonicFX] Launching Streamlit dashboard on 0.0.0.0:$PORT_TO_USE..."
exec streamlit run dashboard.py \
    --server.address 0.0.0.0 \
    --server.port $PORT_TO_USE \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false
