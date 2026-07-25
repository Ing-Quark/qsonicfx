#!/bin/bash
# Dedicated Streamlit Dashboard Start Script for Railway
PORT_TO_USE="${PORT:-8501}"
echo "[Q-SonicFX Dashboard] Launching Streamlit on 0.0.0.0:$PORT_TO_USE..."
exec streamlit run dashboard.py \
    --server.address 0.0.0.0 \
    --server.port $PORT_TO_USE \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false
