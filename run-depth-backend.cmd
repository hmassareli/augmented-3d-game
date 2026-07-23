@echo off
cd /d "%~dp0"
echo Installing DA3 depth backend dependencies...
python -m pip install -r depth_backend\requirements.txt
echo.
echo Make sure DA3 is installed:
echo   git clone https://github.com/ByteDance-Seed/depth-anything-3.git
echo   cd depth-anything-3 && pip install -e .
echo.
echo Starting depth backend on port 8001...
python -m uvicorn depth_backend.server:app --host 127.0.0.1 --port 8001
