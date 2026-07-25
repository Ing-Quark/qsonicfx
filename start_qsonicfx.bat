@echo off
TITLE Q-SonicFX -- Institutional Quantitative Trading System
COLOR 0A
CLS

echo =========================================================================
echo                   ⚡ Q-SONICFX QUANTITATIVE TRADING ENGINE
echo =========================================================================
echo.
echo  [1/2] Launching FastAPI Orchestration Core (http://localhost:8000)...
start "Q-SonicFX Engine Core" cmd /k "python api_server.py --port 8000 --autostart"

echo  [2/2] Launching Streamlit HFT Command Terminal (http://localhost:8501)...
start "Q-SonicFX Terminal UI" cmd /k "streamlit run dashboard.py --server.port 8501"

echo.
echo =========================================================================
echo  SUCCESS: Q-SonicFX system launched natively in background windows.
echo  Access HFT Terminal in Browser at: http://localhost:8501
echo =========================================================================
echo.
pause
