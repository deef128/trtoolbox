REM path of the anaconda installation
set anaconda=C:\Users\Dave\Anaconda3\

REM anaconda environment
set anaenv=data_science

CALL  %anaconda%\Scripts\activate.bat %anaconda%\envs\%anaenv%
python gui.py
