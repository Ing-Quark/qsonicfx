#!/bin/bash
# Start both services in parallel

# Start FastAPI in background
uvicorn api_server:app --host 0.0.0.0 --port ${PORT:-8000} --log-level info &

# Start Streamlit in foreground (keeps the container alive)
streamlit run dashboard.py --server.address 0.0.0.0 --server.port ${STREAMLIT_PORT:-8501} --server.headless true
