@echo off
REM Move to root directory
cd /d "%~dp0.."

echo Cutting dataset into patches...
call python src/data_preparation/cut_patches.py "%cd%/data/barknet/dataset" "%cd%/data/barknet/patches_224" --patch-size 224 --test-ratio 0.0
echo Done.

pause