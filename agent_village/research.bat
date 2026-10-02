@echo off
rem Your full research run. Put prices in data\, PDFs/notes/Pine in papers\, videos in videos\.
rem Change --tz to the time zone of your data files (e.g. America/Chicago).
cd /d "%~dp0"
python -m pip install -q -r requirements.txt
if errorlevel 1 (
  echo Could not install the requirements. Is Python installed and on PATH?
  pause
  exit /b 1
)
rem Optional speed-up; the village still works if this fails.
python -m pip install -q numba || echo numba not available for this Python, running without it.
if exist videos\*.* (
  python -m pip install -q faster-whisper && python -m village transcribe videos --out papers\transcripts
)
python -m village run --data data --timeframe 15min --tz America/New_York --team mixed --scout 3M --expand 6M,1Y,3Y,10Y,all --rounds 20 %*
echo.
echo Done. Open the newest .html file in "reports" and paste reports\best_strategy.pine into TradingView.
pause
