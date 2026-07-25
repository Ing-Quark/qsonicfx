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
    # Fallback when RAILWAY_SERVICE_NAME is unpopulated
    if [ "$PORT_TO_USE" = "8000" ]; then
        exec bash start_api.sh
    else
        exec bash start_dashboard.sh
    fi
fi
