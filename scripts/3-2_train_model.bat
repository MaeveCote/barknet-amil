@echo off
REM Move to root directory
cd /d "%~dp0.."

echo Running Model Training...
call python src/train_model.py -c "%cd%/config/config.yaml"
echo Done.

pause
