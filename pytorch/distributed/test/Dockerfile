# Use the official image as a parent image.
FROM pytorch/pytorch

RUN apt-get update
RUN mkdir -p /usr/src/app

# Set the working directory.
WORKDIR /usr/src/app

ADD aws_dist_train_tuto_cpu.py .

EXPOSE 12355


