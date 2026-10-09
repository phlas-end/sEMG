@echo off
rem Run scripts\configure.cmd, then edit paths in local_settings.cmd.
rem Board IP and serial port belong in local_connection.json, not this file.
set "EMG_IDF_DIR=%USERPROFILE%\esp\v5.5\esp-idf"
set "EMG_PYTHON=%USERPROFILE%\miniconda3\envs\myo\python.exe"
set "EMG_EXPORT_PYTHON=%USERPROFILE%\miniconda3\envs\espdl-int8\python.exe"
