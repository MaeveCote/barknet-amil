@echo off
REM Move to root directory
cd /d "%~dp0.."

echo Training the backbone...
call python src/pretrain_backbone.py -c "%cd%/config/config.yaml"
echo Done.

pause