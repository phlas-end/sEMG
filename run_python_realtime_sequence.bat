@echo off
setlocal
cd /d d:\Project\emg-esp32
set PY=C:\Users\phlas\miniconda3\envs\myo\python.exe
set COMMON=python_realtime_infer.py --device cuda --max-windows 1000000 --legacy-output --disable-rest-gate

echo [1/4] baseline: E2_light_final_pool_48x1
start "PyTorch 1 baseline" /wait cmd /k "%PY% %COMMON% --config runs\E2_light_final_pool_48x1_20260409-222302\checkpoints\config.yaml --checkpoint runs\E2_light_final_pool_48x1_20260409-222302\checkpoints\best.pt"

echo [2/4] fp48_fc768
start "PyTorch 2 fp48_fc768" /wait cmd /k "%PY% %COMMON% --config runs\E2_iter_fp48_fc768_20260412-220518\checkpoints\config.yaml --checkpoint runs\E2_iter_fp48_fc768_20260412-220518\checkpoints\best.pt"

echo [3/4] fp24_fc1024
start "PyTorch 3 fp24_fc1024" /wait cmd /k "%PY% %COMMON% --config runs\E2_iter_fp24_fc1024_20260412-220753\checkpoints\config.yaml --checkpoint runs\E2_iter_fp24_fc1024_20260412-220753\checkpoints\best.pt"

echo [4/4] fp24_fc768
start "PyTorch 4 fp24_fc768" /wait cmd /k "%PY% %COMMON% --config runs\E2_iter_fp24_fc768_20260412-220921\checkpoints\config.yaml --checkpoint runs\E2_iter_fp24_fc768_20260412-220921\checkpoints\best.pt"

echo All 4 models finished.
pause
