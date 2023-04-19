@echo off
call activate Pytorch_1-12-1
setlocal

echo ==ME621==


:loop
for %%I in ("%~dp0main.py") do (
    python "%%~fI"
    timeout /t 86400 /nobreak
)
goto loop

pause
call deactivate
