@echo off
python -m venv venv
call venv\Scripts\activate
pip install google.generativeai fastapi python-dotenv pillow fastapi[standard]
cmd /k