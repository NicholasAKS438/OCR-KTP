FROM python:3.10


WORKDIR /code


COPY ./OCR-KTP-Refactored/requirements.txt /code/requirements.txt


RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt


COPY ./OCR-KTP-Refactored /code/app


CMD ["fastapi", "run", "app/main.py", "--port", "8080"]