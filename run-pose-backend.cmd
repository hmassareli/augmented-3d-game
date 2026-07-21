@echo off
cd /d "%~dp0"
python -m pip install -r pose_backend\requirements.txt
python -m pip install --no-deps -r pose_backend\requirements-acceleration.txt
python -m uvicorn pose_backend.server:app --host 127.0.0.1 --port 8000