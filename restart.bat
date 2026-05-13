@echo off
REM Stop any running bot python.exe (matches by command line containing main.py),
REM then re-run setup.bat. NapCat (QQ.exe) is left alone.
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*main.py*' } | ForEach-Object { Write-Host ('[restart.bat] stopping pid=' + $_.ProcessId); Stop-Process -Id $_.ProcessId -Force }"
call "%~dp0setup.bat" %*
