@echo off
echo Starting RagBot...

:: Get IP address
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /r /c:"IPv4"') do (
    set IP=%%a
    goto :found
)
:found
set IP=%IP:~1%

:: Start the backend server
cd ragbot-backend
call venv\Scripts\activate
start cmd /k "python app.py"

:: Start the frontend server
cd ..\ragbot-frontend
set PORT=3000
set HOST=0.0.0.0
set DANGEROUSLY_DISABLE_HOST_CHECK=true
start cmd /k "npm start"

echo RagBot is starting up...
echo Backend will be available at http://%IP%:5000
echo Frontend will be available at http://%IP%:3000
echo.
echo Other computers on the network can access the application using these URLs
echo Make sure your firewall allows connections on ports 3000 and 5000 