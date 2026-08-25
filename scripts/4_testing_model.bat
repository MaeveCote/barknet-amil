@echo off
REM Move to root directory
cd /d "%~dp0.."

echo Running Model Testing...
call python src/test_model.py -c "%cd%/configs/config.yaml"
echo Done.

pause
