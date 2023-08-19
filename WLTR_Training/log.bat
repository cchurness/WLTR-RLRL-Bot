@ECHO OFF

ECHO Starting Tensorboard Server

CALL .\env\Scripts\activate.bat

CALL tensorboard --logdir logs

PAUSE

CALL deactivate