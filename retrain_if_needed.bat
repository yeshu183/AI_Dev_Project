@echo off
setlocal enabledelayedexpansion

:: Navigate to the script directory
cd /d %~dp0

:: Set threshold
set THRESHOLD=1000
set LOG_FILE=cron_log.txt

echo [%date% %time%] Starting DVC repro... >> %LOG_FILE%
cd dvc_retraining
call dvc repro >> %LOG_FILE% 2>&1
cd ..

:: Count number of files in feedback_data
set COUNT=0
for /R "feedback_data" %%F in (*) do (
    set /a COUNT+=1
)

echo [%date% %time%] Feedback data file count: !COUNT! >> %LOG_FILE%

:: Check threshold
if !COUNT! GEQ %THRESHOLD% (
    echo [%date% %time%] Threshold exceeded. Rebuilding backend... >> %LOG_FILE%
    docker compose up backend --build -d >> %LOG_FILE% 2>&1
) else (
    echo [%date% %time%] Below threshold. Skipping Docker rebuild. >> %LOG_FILE%
)

endlocal
