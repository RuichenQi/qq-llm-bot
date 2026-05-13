@echo off
REM Entry point that bypasses PowerShell's default execution policy.
REM Double-click this on a fresh machine; it forwards everything to run.ps1.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
exit /b %ERRORLEVEL%
