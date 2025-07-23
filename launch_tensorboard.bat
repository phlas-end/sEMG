@echo off
REM 获取当前批处理文件所在目录
set "CURRENT_DIR=%~dp0"

REM 切换到该目录
cd /d "%CURRENT_DIR%"

REM 激活 conda 环境
call conda activate metabci

REM 如果你希望指定日志目录，可以改这里
set "LOGDIR=%CURRENT_DIR%runs"

REM 设置端口（可修改）
set "PORT=6006"

echo Launching TensorBoard in environment 'metabci'...
tensorboard --logdir="%LOGDIR%" --port=%PORT%

pause
