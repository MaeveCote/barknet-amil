@echo off
REM Move to root directory
cd /d "%~dp0.."

echo Running Model Hyperparameter tuning.
call python src/hyperparameter_tuning.py -c "%cd%/configs/config.yaml" --stage board
echo Done.

pause