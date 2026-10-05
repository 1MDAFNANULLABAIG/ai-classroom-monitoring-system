@echo off
echo ======================================================================
echo Starting AI Smart Classroom Monitoring, Attendance & Malpractice Detection
echo ======================================================================
echo.
cd /d "%~dp0"

echo [1/3] Checking environment & database...
python -c "import app; app.init_db(); app.ensure_face_dirs()"

echo [2/3] Building latest React frontend production bundle...
if not exist "frontend\dist\index.html" (
    echo Building frontend...
    cd frontend && call npm run build && cd ..
)

echo [3/3] Launching Production Backend & Web Dashboard...
echo Local Web Dashboard:  http://127.0.0.1:5000/app/
echo REST API Base:        http://127.0.0.1:5000/api
echo WebSocket Feed:       ws://127.0.0.1:5000/ws/classroom
echo Legacy Flask UI:      http://127.0.0.1:5000/
echo.
python app.py
pause
