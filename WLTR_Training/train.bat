@ECHO OFF

ECHO Starting WLTR Training

CALL .\env\Scripts\activate.bat

START cmd /k python train.py
START cmd /k tensorboard --logdir logs

CALL deactivate