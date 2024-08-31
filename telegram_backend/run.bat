@echo off
call activate Pytorch_1-12-1
setlocal

for %%I in ("%~dp0main.py") do (
    python "%%~fI"
)

call deactivate
exit
