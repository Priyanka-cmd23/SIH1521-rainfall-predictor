# Start the SIH1521 dashboard.
# Usage:  powershell -ExecutionPolicy Bypass -File run_demo.ps1
.\.venv\Scripts\python.exe -m uvicorn api.app:app --host 127.0.0.1 --port 8000