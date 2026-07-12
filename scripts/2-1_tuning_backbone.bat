@echo off
REM Move to root directory
cd /d "%~dp0.."

echo Running Backbone Pretrain Hyperparameter tuning.
call python src/hyperparameter_tuning.py -c "%cd%/config/config.yaml" --stage pretrain
echo Done.

pause