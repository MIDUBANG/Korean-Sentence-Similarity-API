FROM python:3.8.0
FROM ubuntu:latest

ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
  apt-get install -y --no-install-recommends tzdata g++ curl


# install java
RUN  apt-get install g++ openjdk-8-jdk
RUN apt-get install python3-dev
RUN pip3 install konlpy


# copy resources
COPY . .

RUN echo server will be running on 5000


RUN pip install -r requirements.txt



CMD python app.py