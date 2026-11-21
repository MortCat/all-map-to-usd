@echo off
rem AllMapToUSD - the single entry point.
rem  * First run (or after an update): scripts\bootstrap.py shows a small setup window that creates
rem    .venv and installs everything, then starts the app.
rem  * Later runs: starts the app directly. Extra arguments go to run_app.py (e.g. --browser).
rem Only prerequisite: Python 3.11+ for Windows (python.org installer, includes tkinter).
setlocal
cd /d "%~dp0"

set "PYW="
rem 1) An existing, working .venv runs the launcher itself (no system Python needed afterwards)
if exist ".venv\Scripts\pythonw.exe" (
  ".venv\Scripts\python.exe" -c "import sys" >nul 2>&1 && set "PYW=.venv\Scripts\pythonw.exe"
)
rem 2) Otherwise find a system Python 3.11+ to create .venv with
if not defined PYW call :find_python
if not defined PYW goto :no_python

start "" "%PYW%" "scripts\bootstrap.py" %*
exit /b 0


:find_python
set "PYEXE="
for /f "delims=" %%P in ('py -3 -c "import sys; print(sys.executable)" 2^>nul') do set "PYEXE=%%P"
if not defined PYEXE for /f "delims=" %%P in ('python -c "import sys; print(sys.executable)" 2^>nul') do set "PYEXE=%%P"
if not defined PYEXE exit /b 0
"%PYEXE%" -c "import sys; sys.exit(sys.version_info < (3, 11))" >nul 2>&1 || (set "PYEXE=" & exit /b 0)
rem pythonw = no console window; without tkinter fall back to python.exe (setup runs in a console)
set "PYW=%PYEXE:python.exe=pythonw.exe%"
"%PYEXE%" -c "import tkinter" >nul 2>&1 || set "PYW=%PYEXE%"
exit /b 0


:no_python
powershell -NoProfile -Command "Add-Type -AssemblyName PresentationFramework; [void][System.Windows.MessageBox]::Show('AllMapToUSD needs Python 3.11 or newer. The python.org download page will open now: install Python (keep the default options), then run start_app.cmd again.', 'AllMapToUSD', 'OK', 'Information')"
start "" "https://www.python.org/downloads/windows/"
exit /b 1
