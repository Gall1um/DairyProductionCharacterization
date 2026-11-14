@echo off
cd /d "%~dp0"
where py >nul 2>nul
if not errorlevel 1 (
  py -3.13 --version >nul 2>nul
  if not errorlevel 1 (py -3.13 bootstrap.py & goto finish)
  py -3.12 --version >nul 2>nul
  if not errorlevel 1 (py -3.12 bootstrap.py & goto finish)
  py -3.11 --version >nul 2>nul
  if not errorlevel 1 (py -3.11 bootstrap.py & goto finish)
  py -3.10 --version >nul 2>nul
  if not errorlevel 1 (py -3.10 bootstrap.py & goto finish)
  py -3 bootstrap.py
  goto finish
)
where python >nul 2>nul
if not errorlevel 1 (python bootstrap.py & goto finish)
echo Python is needed to run MilkWise. Install Python 3.12 or 3.13 from python.org.
set CODE=1
goto fail
:finish
set CODE=%ERRORLEVEL%
if %CODE% EQU 0 exit /b 0
:fail
echo.
echo MilkWise could not start. Read the error above.
pause
exit /b %CODE%
