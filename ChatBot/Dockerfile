FROM python:3.8.0
FROM tensorflow/tensorflow:1.15.2-py3

WORKDIR /application

COPY './requirements.txt' .

RUN pip install -r requirements.txt

COPY . .

ENTRYPOINT [ "python", "app.py"]
