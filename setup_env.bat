@echo off
virtualenv venv --python=python3.12
call venv\Scripts\activate
pip install -r requirements.txt
cmd /k