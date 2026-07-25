#!/bin/bash
# Start both services or single service on Railway $PORT

# Start FastAPI in background (if running both)
uvicorn api_server:app --host 0.0.0.0 --port ${API_PORT:-8000} --log-level info &

# Start Streamlit in foreground on $PORT for Railway proxy routing
streamlit run dashboard.py --server.address 0.0.0.0 --server.port ${PORT:-8501} --server.headless true
