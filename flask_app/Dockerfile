FROM python:3.8.0
FROM ubuntu:latest

ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /usr/src/flask_app
COPY requirements.txt .
RUN apt-get update && \
  apt-get install -y --no-install-recommends tzdata g++ curl
RUN apt-get install -y python3-pip

# install java
RUN  apt-get install -y g++ openjdk-8-jdk

RUN apt-get install -y python3-dev
RUN pip install --upgrade pip
RUN pip install konlpy


# copy resources
COPY . .

RUN echo server will be running on 5000

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3", "app.py"]                    