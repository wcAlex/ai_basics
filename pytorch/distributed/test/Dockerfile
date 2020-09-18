# Use the official image as a parent image.
FROM pytorch/pytorch

RUN apt-get update
RUN apt-get install -y vim
RUN apt-get install -y net-tools
RUN apt-get install -y iputils-ping
RUN apt-get install -y iproute2

RUN mkdir -p /usr/src/app

# Set the working directory.
WORKDIR /usr/src/app

ADD aws_dist_train_tuto_cpu.py .

EXPOSE 12355




