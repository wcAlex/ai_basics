FROM ubuntu:18.04
#FROM alpine:3.8

RUN apt-get update -y && \
    apt-get install -y python3.8 python3.8-dev python3.8-distutils python3-pip

# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip3 install -r requirements.txt

EXPOSE 8080

COPY . /app

ENTRYPOINT bash run.sh
