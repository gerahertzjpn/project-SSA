@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_viewer.ps1"
if errorlevel 1 pause
