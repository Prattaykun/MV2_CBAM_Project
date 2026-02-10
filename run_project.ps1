# Start Backend
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd backend; ..\venv\Scripts\uvicorn main:app --reload --host 0.0.0.0 --port 8000"

# Start Frontend
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd frontend; npm run dev"

Write-Host "Project started! Access the dashboard at http://localhost:3000"
