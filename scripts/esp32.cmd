@echo off
setlocal
set "EMG_ROOT=%~dp0.."
if exist "%EMG_ROOT%\local_settings.cmd" call "%EMG_ROOT%\local_settings.cmd"
if not defined EMG_PYTHON set "EMG_PYTHON=%USERPROFILE%\miniconda3\envs\myo\python.exe"
set "EMG_ACTION=%~1"
if "%EMG_ACTION%"=="configure" (
  call "%~dp0configure.cmd"
  exit /b
)
if "%EMG_ACTION%"=="build" goto check_wifi
if "%EMG_ACTION%"=="flash" goto get_port
if "%EMG_ACTION%"=="probe" goto get_port
if "%EMG_ACTION%"=="monitor" goto get_port
if "%EMG_ACTION%"=="menuconfig" goto prepare
echo Usage: scripts\esp32.cmd configure ^| build ^| menuconfig ^| probe [PORT] ^| flash [PORT] ^| monitor [PORT]
exit /b 2
:get_port
set "EMG_PORT=%~2"
if not defined EMG_PORT for /f "delims=" %%P in ('""%EMG_PYTHON%" "%EMG_ROOT%\connection_config.py" --get serial_port"') do set "EMG_PORT=%%P"
if not defined EMG_PORT (
  echo Set serial_port in local_connection.json or pass the actual COM port.
  exit /b 2
)
if "%EMG_ACTION%"=="flash" goto check_wifi
goto prepare
:check_wifi
"%EMG_PYTHON%" "%EMG_ROOT%\connection_config.py" --check-wifi "%EMG_ROOT%\C++\main\network_config.h"
if errorlevel 1 exit /b 2
:prepare
if not defined EMG_IDF_DIR set "EMG_IDF_DIR=%USERPROFILE%\esp\v5.5\esp-idf"
if exist "%USERPROFILE%\.espressif\python_env\idf5.5_py3.11_env\Scripts\python.exe" set "PATH=%USERPROFILE%\.espressif\python_env\idf5.5_py3.11_env\Scripts;%PATH%"
if not exist "%EMG_IDF_DIR%\export.bat" (
  echo ESP-IDF not found. Configure EMG_IDF_DIR in local_settings.cmd.
  exit /b 1
)
call "%EMG_IDF_DIR%\export.bat"
if errorlevel 1 exit /b 1
if "%EMG_ACTION%"=="probe" goto probe
if "%EMG_ACTION%"=="build" goto build
if "%EMG_ACTION%"=="menuconfig" goto menuconfig
if "%EMG_ACTION%"=="flash" goto flash
if "%EMG_ACTION%"=="monitor" goto monitor
echo Usage: scripts\esp32.cmd configure ^| build ^| menuconfig ^| probe [PORT] ^| flash [PORT] ^| monitor [PORT]
exit /b 2
:probe
python -m esptool --port "%EMG_PORT%" chip_id
if errorlevel 1 exit /b 1
python -m esptool --port "%EMG_PORT%" flash_id
exit /b %errorlevel%
:build
cd /d "%EMG_ROOT%\C++"
idf.py build
exit /b %errorlevel%
:menuconfig
cd /d "%EMG_ROOT%\C++"
idf.py menuconfig
exit /b %errorlevel%
:flash
cd /d "%EMG_ROOT%\C++"
idf.py -p "%EMG_PORT%" flash
exit /b %errorlevel%
:monitor
cd /d "%EMG_ROOT%\C++"
idf.py -p "%EMG_PORT%" monitor
exit /b %errorlevel%
