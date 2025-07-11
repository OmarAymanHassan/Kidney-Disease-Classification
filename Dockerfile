FROM python:3.12-slim-buster

RUN apt update -y && apt install awscli -y

WORKDIR /app
COPY . /app


RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]

