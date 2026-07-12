@echo off
REM Move to root directory
cd /d "%~dp0.."

echo Cutting dataset into patches...
call python src/data_preparation/cut_patches.py "%cd%/data/barknet/dataset" "%cd%/data/barknet/patches_384" --patch-size 384 --test-ratio 0.20
echo Done.

pause