@echo off

verify other 2>nul
setlocal enableextensions enabledelayedexpansion

set OUTPUT=D:\Tmp\TestStandalone_デテクター

if not defined CONTEXT_INSIGHTS (
   echo CONTEXT_INSIGHTS environment variable should be set
   exit /b 1
)

if not defined CONTEXT_INSIGHTS_VENV (
   echo CONTEXT_INSIGHTS_VENV environment variable should be set
   exit /b 1
)

if not defined CONTEXT_INSIGHTS_JOBQUEUE (
   echo CONTEXT_INSIGHTS_JOBQUEUE environment variable should be set
   exit /b 1
)

if not defined CONTEXT_INSIGHTS_PYTHON  (
   echo CONTEXT_INSIGHTS_PYTHON  environment variable should be set
   exit /b 1
)

set FRESH_VENV=0
if not exist %CONTEXT_INSIGHTS_VENV% set FRESH_VENV=1
if %FRESH_VENV% equ 1 (
    call "%CONTEXT_INSIGHTS%\sdk\context_insights\create_venv"
    if ERRORLEVEL 1 exit /b 1
)

set PATH=%CONTEXT_INSIGHTS%\bin;%PATH%
set PYTHONPATH=%CONTEXT_INSIGHTS%\sdk\context_insights;%PYTHONPATH%

call %CONTEXT_INSIGHTS_VENV%\Scripts\activate.bat
if ERRORLEVEL 1 exit /b 1

python %CONTEXT_INSIGHTS%\sdk\samples\context_insights_sample.py --output %OUTPUT%
if ERRORLEVEL 1 exit /b 1

call %CONTEXT_INSIGHTS_VENV%\Scripts\deactivate.bat
if ERRORLEVEL 1 exit /b 1
