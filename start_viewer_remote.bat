@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_viewer_remote.ps1" %*
if errorlevel 1 pause
