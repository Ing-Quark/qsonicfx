web: bash start.sh
api: uvicorn api_server:app --host 0.0.0.0 --port $PORT --log-level info
dashboard: streamlit run dashboard.py --server.address 0.0.0.0 --server.port $PORT --server.headless true --server.enableCORS false --server.enableXsrfProtection false
