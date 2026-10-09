@echo off
setlocal
set "EMG_ROOT=%~dp0.."
if exist "%EMG_ROOT%\local_settings.cmd" call "%EMG_ROOT%\local_settings.cmd"
if not defined EMG_IDF_DIR set "EMG_IDF_DIR=%USERPROFILE%\esp\v5.5\esp-idf"
if exist "%USERPROFILE%\.espressif\python_env\idf5.5_py3.11_env\Scripts\python.exe" set "PATH=%USERPROFILE%\.espressif\python_env\idf5.5_py3.11_env\Scripts;%PATH%"
if not exist "%EMG_IDF_DIR%\export.bat" (
  echo ESP-IDF not found. Configure EMG_IDF_DIR in local_settings.cmd.
  exit /b 1
)
call "%EMG_IDF_DIR%\export.bat"
if errorlevel 1 exit /b 1
set "EMG_ACTION=%~1"
if "%EMG_ACTION%"=="probe" goto probe
if "%EMG_ACTION%"=="build" goto build
if "%EMG_ACTION%"=="menuconfig" goto menuconfig
if "%EMG_ACTION%"=="flash" goto flash
if "%EMG_ACTION%"=="monitor" goto monitor
echo Usage: scripts\esp32.cmd build ^| menuconfig ^| probe PORT ^| flash PORT ^| monitor PORT
exit /b 2
:probe
if "%~2"=="" exit /b 2
python -m esptool --port %2 chip_id
if errorlevel 1 exit /b 1
python -m esptool --port %2 flash_id
exit /b %errorlevel%
:build
cd /d "%EMG_ROOT%\C++"
if not exist "main\network_config.h" (
  echo Copy main\network_config.h.example to main\network_config.h and configure Wi-Fi.
  exit /b 1
)
idf.py build
exit /b %errorlevel%
:menuconfig
cd /d "%EMG_ROOT%\C++"
idf.py menuconfig
exit /b %errorlevel%
:flash
if "%~2"=="" exit /b 2
cd /d "%EMG_ROOT%\C++"
idf.py -p %2 flash
exit /b %errorlevel%
:monitor
if "%~2"=="" exit /b 2
cd /d "%EMG_ROOT%\C++"
idf.py -p %2 monitor
exit /b %errorlevel%
