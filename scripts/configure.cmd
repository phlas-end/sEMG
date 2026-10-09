@echo off
setlocal
set "EMG_ROOT=%~dp0.."
if not exist "%EMG_ROOT%\local_settings.cmd" (
  copy "%EMG_ROOT%\local_settings.example.cmd" "%EMG_ROOT%\local_settings.cmd" >nul
  if errorlevel 1 exit /b 1
)
if not exist "%EMG_ROOT%\local_connection.json" (
  copy "%EMG_ROOT%\local_connection.example.json" "%EMG_ROOT%\local_connection.json" >nul
  if errorlevel 1 exit /b 1
)
if not exist "%EMG_ROOT%\C++\main\network_config.h" (
  copy "%EMG_ROOT%\C++\main\network_config.h.example" "%EMG_ROOT%\C++\main\network_config.h" >nul
  if errorlevel 1 exit /b 1
)
echo Local configuration is ready. Existing files were NOT overwritten.
echo 1. Edit local_settings.cmd: ESP-IDF and Python tool paths.
echo 2. Edit C++\main\network_config.h: REQUIRED Wi-Fi SSID and password.
echo 3. Edit local_connection.json: actual serial_port and server_ip.
echo    Find the serial port in Device Manager. Read the board IP from its startup log.
echo    Keep server_port at 3333 unless you also change the firmware TCP port.
echo These local files are ignored by Git. Never upload credentials or local paths.
exit /b 0
