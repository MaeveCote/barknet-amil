@echo off
REM Move to root directory
cd /d "%~dp0.."

echo Running Batch Training...
call python src/batch_training.py -c "%cd%/config/batch_training_config.yaml"
echo Done.

pause