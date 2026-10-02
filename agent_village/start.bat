@echo off
rem Double-click to install the requirements and run the village for 5 rounds.
cd /d "%~dp0"
python -m pip install -q -r requirements.txt
if errorlevel 1 (
  echo.
  echo Could not install the requirements. Is Python installed and on PATH?
  pause
  exit /b 1
)
python -m village run --rounds 5 %*
echo.
echo Done. Open the newest .html file in the "reports" folder in your browser.
pause
