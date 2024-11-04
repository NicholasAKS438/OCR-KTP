@echo off
python -m venv venv
call venv\Scripts\activate
pip install google.generativeai fastapi dotenv
cmd /k